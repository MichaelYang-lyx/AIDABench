# lite_code_execution_toolkit.py
# A lightweight, stdlib-only CodeExecutionToolkit with a "jupyter-like" persistent sandbox.
# - No CAMEL dependency
# - Persistent state across runs per (namespace, session_id)
# - Thread-safe isolation between concurrent sessions
#
# Usage:
#   run_code = CodeExecutionToolkit(sandbox="jupyter", namespace="model-A").get_tools()[0]
#   print(run_code(code="x=1\nx+41"))  # -> 42
#
# Notes:
# - This is "Jupyter-like" persistence (shared env dict), not an actual Jupyter kernel process.
# - State persists in-memory for the lifetime of the Python process.
# - For real kernel-level isolation/timeouts, you'd typically use jupyter_client, but that adds deps.

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import ctypes
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class FunctionTool:
    """A tiny callable tool wrapper (CAMEL-like)."""
    name: str
    description: str
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


@dataclasses.dataclass
class _Session:
    env: Dict[str, Any]
    workdir: Path
    created_at: float
    last_used_at: float
    exec_count: int
    lock: threading.RLock


# ---------------------------------------------------------------------------
# Thread-safe stdout/stderr capture using thread-local storage
# ---------------------------------------------------------------------------
class _ThreadLocalStream:
    """
    A stream wrapper that uses thread-local storage for per-thread capture.
    When a thread sets a buffer via set_buffer(), all writes from that thread
    go to the buffer instead of the original stream. Other threads are unaffected.
    """
    def __init__(self, original):
        self._original = original
        self._local = threading.local()

    def write(self, s):
        buf = getattr(self._local, 'buffer', None)
        if buf is not None:
            return buf.write(s)
        return self._original.write(s)

    def flush(self):
        buf = getattr(self._local, 'buffer', None)
        if buf is not None:
            return buf.flush()
        return self._original.flush()

    def set_buffer(self, buf):
        """Set a capture buffer for the current thread."""
        self._local.buffer = buf

    def clear_buffer(self):
        """Remove the capture buffer for the current thread (writes go to original)."""
        self._local.buffer = None

    def __getattr__(self, name):
        # Delegate all other attributes (encoding, fileno, etc.) to the original stream
        return getattr(self._original, name)


# Save process-level original cwd at import time. All threads restore to this
# after exec(), avoiding the race where thread B saves thread A's temp workdir.
_ORIGINAL_CWD = os.getcwd()

# Install thread-local stream wrappers (once at module level)
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_tl_stdout = _ThreadLocalStream(_original_stdout)
_tl_stderr = _ThreadLocalStream(_original_stderr)
sys.stdout = _tl_stdout  # type: ignore[assignment]
sys.stderr = _tl_stderr  # type: ignore[assignment]


class _ChdirGuard:
    """Process-wide chdir guard (since os.chdir is global).

    NOTE: This is only used by execute_command now. execute_code uses
    code injection to avoid the global lock serialization bottleneck.
    """
    _lock = threading.RLock()

    def __init__(self, target: Path):
        self._target = str(target)
        self._prev: Optional[str] = None

    def __enter__(self):
        self._lock.acquire()
        self._prev = os.getcwd()
        os.chdir(self._target)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._prev is not None:
                os.chdir(self._prev)
        finally:
            self._lock.release()


class _SessionRegistry:
    """Global registry so multiple toolkit instances can still share the same session if desired."""
    _lock = threading.RLock()
    _sessions: Dict[Tuple[str, str], _Session] = {}

    @classmethod
    def get_or_create(cls, namespace: str, session_id: str) -> _Session:
        key = (namespace, session_id)
        with cls._lock:
            sess = cls._sessions.get(key)
            if sess is not None:
                sess.last_used_at = time.time()
                return sess

            # per-session working directory
            base = Path(tempfile.gettempdir()) / "lite_code_exec"
            base.mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix=f"{namespace}__{session_id}__", dir=str(base)))

            env: Dict[str, Any] = {
                "__name__": "__main__",
                "__package__": None,
                "__builtins__": __builtins__,  # keep python behavior
                "_": None,   # last result (like IPython)
                "__": None,
                "___": None,
            }
            now = time.time()
            sess = _Session(
                env=env,
                workdir=workdir,
                created_at=now,
                last_used_at=now,
                exec_count=0,
                lock=threading.RLock(),
            )
            cls._sessions[key] = sess
            return sess

    @classmethod
    def reset(cls, namespace: str, session_id: str) -> None:
        key = (namespace, session_id)
        with cls._lock:
            sess = cls._sessions.pop(key, None)
        if sess is not None:
            # best-effort cleanup
            try:
                for p in sess.workdir.rglob("*"):
                    if p.is_file():
                        p.unlink(missing_ok=True)
                for p in sorted(sess.workdir.rglob("*"), reverse=True):
                    if p.is_dir():
                        p.rmdir()
                sess.workdir.rmdir()
            except Exception:
                pass  # ignore cleanup errors


def _split_last_expr(code: str) -> Tuple[ast.AST, Optional[ast.AST]]:
    """
    If last statement is an expression, split it out so we can eval it (Jupyter-like display).
    Returns: (exec_tree, eval_expr_or_None)
    """
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        return tree, None
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        expr = last.value
        tree.body = tree.body[:-1]
        exec_tree = tree
        eval_tree = ast.Expression(expr)
        return exec_tree, eval_tree
    return tree, None


def _collect_files(root: Path) -> List[str]:
    files: List[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                files.append(str(p.relative_to(root)))
            except Exception:
                files.append(str(p))
    files.sort()
    return files


@contextlib.contextmanager
def _timeout_guard(seconds: Optional[float]):
    """
    Best-effort timeout guard.
    - Uses signal.alarm on Main Thread (POSIX only).
    - Uses ctypes async exception injection on Worker Threads (with retry).
    """
    if seconds is None:
        yield
        return

    # 1. Main Thread + POSIX: Use Signal (Classic efficient method)
    if os.name == "posix" and threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {seconds} seconds.")

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
        return

    # 2. Worker Thread or Non-POSIX: Use ctypes async exception with retry
    # This allows interrupting threads in ThreadPoolExecutor
    target_tid = threading.get_ident()
    timed_out = threading.Event()

    def _interrupt():
        timed_out.set()
        # Retry multiple times — a single injection may not take effect if the thread
        # is blocked inside a C extension; retrying increases the chance of catching
        # it between bytecode instructions.
        for _attempt in range(5):
            ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(target_tid),
                ctypes.py_object(TimeoutError)
            )
            if ret == 0:
                break  # Thread already finished
            elif ret > 1:
                # Something went wrong, try to revert
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), 0)
                break
            time.sleep(0.5)

    timer = threading.Timer(seconds, _interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    except TimeoutError:
        raise TimeoutError(f"Code execution timed out after {seconds} seconds.")
    finally:
        timer.cancel()
        # If timeout was triggered but the exception wasn't delivered (e.g. thread was
        # stuck in a C extension that swallowed it), raise now that we're back in Python.
        if timed_out.is_set():
            raise TimeoutError(f"Code execution timed out after {seconds} seconds.")


class CodeExecutionToolkit:
    """
    Lightweight CodeExecutionToolkit (stdlib-only).

    Key params (subset of CAMEL):
      - sandbox: 'jupyter' (persistent) or 'subprocess' (stateless per call)
      - verbose: print execution output to host stdout/stderr
      - unsafe_mode: if False and import_white_list is set, we enforce an import allowlist (basic AST check)
      - import_white_list: allowed top-level module names (e.g. ['math','json'])
      - timeout: per-call timeout (best-effort)
      - namespace: isolation prefix (use model_id / agent_id here!)
    """

    def __init__(
        self,
        sandbox: Literal["jupyter", "subprocess"] = "subprocess",
        verbose: bool = False,
        unsafe_mode: bool = False,
        import_white_list: Optional[List[str]] = None,
        require_confirm: bool = False,
        timeout: Optional[float] = None,
        namespace: str = "default",
        default_session_id: str = "default",
    ):
        self.sandbox = sandbox
        self.verbose = verbose
        self.unsafe_mode = unsafe_mode
        self.import_white_list = import_white_list
        self.require_confirm = require_confirm
        self.timeout = timeout
        self.namespace = namespace
        self.default_session_id = default_session_id

    def get_tools(self) -> List[FunctionTool]:
        # Keep ordering similar to CAMEL: execute_code first, then execute_command
        return [
            FunctionTool(
                name="execute_code",
                description="Execute a code snippet (python). In 'jupyter' sandbox, state persists per session.",
                func=self.execute_code,
            ),
            FunctionTool(
                name="execute_command",
                description="Execute a shell command (best-effort). In 'jupyter' sandbox, uses the session workdir.",
                func=self.execute_command,
            ),
        ]

    def reset_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.default_session_id
        _SessionRegistry.reset(self.namespace, sid)
        return f"Session reset: namespace={self.namespace}, session_id={sid}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()


    # -----------------------
    # Core APIs
    # -----------------------
    def execute_code(
        self,
        code: str,
        code_type: str = "python",
        *,
        session_id: Optional[str] = None,
        confirm: bool = False,
    ) -> str:
        """
        Execute code. Returns text output.

        Extra kwargs:
          - session_id: isolates persistent state in 'jupyter' sandbox
          - confirm: for require_confirm flow
        """
        # --- Security Guard ---
        # Prevent access to OneDrive or system directories that might trigger privacy prompts or freezing
        FORBIDDEN_KEYWORDS = ["OneDrive", "Library", "System", "Applications"]
        for kw in FORBIDDEN_KEYWORDS:
            if kw in code:
                return f"Security Error: Access to path containing '{kw}' is restricted."
        # ----------------------

        if self.require_confirm and not confirm:
            return "Execution requires confirm=True (require_confirm is enabled)."

        if code_type.lower() not in {"python", "py"}:
            return f"Unsupported code_type={code_type!r}. Only 'python' is supported in this lightweight version."

        if self.sandbox == "subprocess":
            return self._execute_code_subprocess(code)

        if self.sandbox != "jupyter":
            return f"Unsupported sandbox={self.sandbox!r}. Use 'jupyter' or 'subprocess'."

        sid = session_id or self.default_session_id
        sess = _SessionRegistry.get_or_create(self.namespace, sid)

        with sess.lock:
            sess.exec_count += 1
            sess.last_used_at = time.time()

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            before_files = _collect_files(sess.workdir)

            try:
                if not self.unsafe_mode and self.import_white_list is not None:
                    self._enforce_import_allowlist(code, self.import_white_list)

                # Inject os.chdir(workdir) into the code instead of using the process-wide
                # _ChdirGuard lock. The old approach held a global RLock for the ENTIRE
                # duration of exec(), which serialized all concurrent jupyter sessions and
                # caused deadlocks when the timeout mechanism failed to fire.
                # The code injection approach has a minor cwd race condition between threads,
                # but this is far better than complete serialization / deadlock.
                workdir_path = str(sess.workdir)
                code_with_chdir = (
                    f"import os as __os_wd; __os_wd.chdir({workdir_path!r}); del __os_wd\n"
                    + code
                )

                exec_tree, eval_tree = _split_last_expr(code_with_chdir)

                last_result = None

                with _timeout_guard(self.timeout):
                    # Use thread-local stream capture instead of redirect_stdout/stderr.
                    # contextlib.redirect_stdout modifies the global sys.stdout, which
                    # is NOT thread-safe — concurrent threads would garble each other's
                    # output and potentially fail to restore the original stream.
                    _tl_stdout.set_buffer(stdout_buf)
                    _tl_stderr.set_buffer(stderr_buf)
                    try:
                        # execute statements
                        if getattr(exec_tree, "body", None):
                            compiled = compile(exec_tree, f"<jupyter:{self.namespace}:{sid}#{sess.exec_count}>", "exec")
                            exec(compiled, sess.env, sess.env)

                        # eval last expression (jupyter-like)
                        if eval_tree is not None:
                            compiled_expr = compile(eval_tree, f"<jupyter:{self.namespace}:{sid}#{sess.exec_count}>", "eval")
                            last_result = eval(compiled_expr, sess.env, sess.env)

                            # update _, __, ___ like IPython
                            sess.env["___"] = sess.env.get("__")
                            sess.env["__"] = sess.env.get("_")
                            sess.env["_"] = last_result
                    finally:
                        _tl_stdout.clear_buffer()
                        _tl_stderr.clear_buffer()
                        # Restore cwd to the process's original working directory
                        # (saved once at module import time), not a per-call snapshot
                        # which is unreliable under concurrent threads.
                        try:
                            os.chdir(_ORIGINAL_CWD)
                        except Exception:
                            pass

            except Exception:
                tb = traceback.format_exc()
                stderr_buf.write(tb)
                last_result = None

            after_files = _collect_files(sess.workdir)
            new_files = [f for f in after_files if f not in before_files]

            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()

            parts: List[str] = []
            if stdout.strip():
                parts.append(stdout.rstrip())

            if last_result is not None:
                parts.append(f"[result]\n{repr(last_result)}")

            if new_files:
                parts.append("[new_files]\n" + "\n".join(new_files))

            if stderr.strip():
                parts.append("[stderr]\n" + stderr.rstrip())

            if not parts:
                parts.append("(no output)")

            out = "\n\n".join(parts)

            if self.verbose:
                _original_stdout.write(out + "\n")

            return out

    def execute_command(
        self,
        command: str,
        *,
        session_id: Optional[str] = None,
        confirm: bool = False,
    ) -> str:
        """
        Execute a shell command (best-effort).
        - In 'jupyter' sandbox: runs in the session workdir
        - In 'subprocess' sandbox: runs in current working dir
        """
        if self.require_confirm and not confirm:
            return "Command execution requires confirm=True (require_confirm is enabled)."

        cwd = None
        if self.sandbox == "jupyter":
            sid = session_id or self.default_session_id
            sess = _SessionRegistry.get_or_create(self.namespace, sid)
            cwd = str(sess.workdir)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = (proc.stdout or "").rstrip()
            stderr = (proc.stderr or "").rstrip()

            if proc.returncode == 0:
                return stdout if stdout else "(no output)"

            # non-zero
            if stdout and stderr:
                return f"[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n\n[returncode]\n{proc.returncode}"
            if stderr:
                return f"[stderr]\n{stderr}\n\n[returncode]\n{proc.returncode}"
            return f"[returncode]\n{proc.returncode}"

        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout} seconds."
        except Exception as e:
            return f"Command failed: {e!r}"

    # -----------------------
    # Helpers
    # -----------------------
    def _execute_code_subprocess(self, code: str) -> str:
        """
        Stateless execution: each call uses a fresh python process.
        """
        try:
            proc = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = (proc.stdout or "").rstrip()
            stderr = (proc.stderr or "").rstrip()

            if proc.returncode == 0:
                return stdout if stdout else "(no output)"

            if stdout and stderr:
                return f"[stdout]\n{stdout}\n\n[stderr]\n{stderr}\n\n[returncode]\n{proc.returncode}"
            if stderr:
                return f"[stderr]\n{stderr}\n\n[returncode]\n{proc.returncode}"
            return f"[returncode]\n{proc.returncode}"
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {self.timeout} seconds."
        except Exception as e:
            return f"Subprocess execution failed: {e!r}"

    def _enforce_import_allowlist(self, code: str, allow: List[str]) -> None:
        """
        Very basic import allowlist enforcement via AST.
        Only checks `import x` and `from x import y`.
        """
        allow_set = set(allow)
        tree = ast.parse(code, mode="exec")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = (alias.name.split(".", 1)[0] or "").strip()
                    if top and top not in allow_set:
                        raise PermissionError(f"Import '{top}' is not allowed. allowlist={sorted(allow_set)}")
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = (node.module.split(".", 1)[0] or "").strip()
                if top and top not in allow_set:
                    raise PermissionError(f"ImportFrom '{top}' is not allowed. allowlist={sorted(allow_set)}")
