import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "workspaces"
HERMES_DIR = Path.home() / ".hermes"


class HermesAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        data_root_path: str,
        save_name: str,
        max_rounds: int = 90,
        provider: str = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.save_name = save_name
        self.profile_name = re.sub(r'[^a-z0-9_-]', '-', save_name.lower())[:64]
        self.max_rounds = max_rounds
        self.timeout = max_rounds * 120
        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider or "custom"

    # ------------------------------------------------------------------
    # Profile setup
    # ------------------------------------------------------------------

    def _run_hermes_cmd(self, args: list[str], check: bool = True):
        result = subprocess.run(
            ["hermes"] + args,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"hermes command failed: {' '.join(args)}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result

    def _setup_profile(self, api_key: str, base_url: str, model_name: str, profile: str, provider: str):
        # Create profile (ignore error if already exists)
        self._run_hermes_cmd(["profile", "create", profile], check=False)

        # For non-custom providers, strip trailing /v1 from base_url to avoid double path
        effective_url = base_url
        if provider != "custom" and effective_url:
            effective_url = effective_url.rstrip("/")
            if effective_url.endswith("/v1"):
                effective_url = effective_url[:-3]

        configs = [
            ("model.default", model_name),
            ("model.provider", provider),
            ("model.api_key", api_key),
            ("terminal.timeout", "1200"),
        ]
        if effective_url:
            configs.append(("model.base_url", effective_url))

        for key, value in configs:
            self._run_hermes_cmd(["--profile", profile, "config", "set", key, value])

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _prepare_workspace(self, workspace_dir: Path, real_input_files: list[str]) -> Path:
        workspace_dir.mkdir(parents=True, exist_ok=True)

        inputs_dir = workspace_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)

        for src in real_input_files:
            src_path = Path(src)
            if src_path.exists():
                dst = inputs_dir / src_path.name
                if not dst.exists() and not dst.is_symlink():
                    dst.symlink_to(src_path.resolve())

        return workspace_dir

    # ------------------------------------------------------------------
    # Session trace
    # ------------------------------------------------------------------

    def _load_session(self, session_id: str, profile_name: str = None) -> Any:
        """Load session from Hermes SQLite store (v0.14+) or legacy JSON files."""
        import sqlite3
        profile = profile_name or self.profile_name

        # Try SQLite first (new Hermes)
        for db_path in [
            HERMES_DIR / "profiles" / profile / "state.db",
            HERMES_DIR / "state.db",
        ]:
            if not db_path.exists():
                continue
            try:
                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, model, started_at, ended_at, system_prompt, message_count "
                    "FROM sessions WHERE id=?", (session_id,)
                )
                row = cur.fetchone()
                if not row:
                    conn.close()
                    continue

                from datetime import datetime
                session_data = {
                    "session_id": row[0],
                    "model": row[1],
                    "session_start": datetime.fromtimestamp(row[2]).isoformat() if row[2] else "",
                    "last_updated": datetime.fromtimestamp(row[3]).isoformat() if row[3] else "",
                    "system_prompt": row[4] or "",
                    "message_count": row[5] or 0,
                    "messages": [],
                }
                # Fallback: derive last_updated from message timestamps if ended_at is NULL
                if not session_data["last_updated"]:
                    cur.execute(
                        "SELECT MAX(timestamp) FROM messages WHERE session_id=?", (session_id,)
                    )
                    max_ts = cur.fetchone()
                    if max_ts and max_ts[0]:
                        session_data["last_updated"] = datetime.fromtimestamp(max_ts[0]).isoformat()

                cur.execute(
                    "SELECT role, content, tool_calls, tool_name, finish_reason "
                    "FROM messages WHERE session_id=? ORDER BY id", (session_id,)
                )
                for msg_row in cur.fetchall():
                    role, content, tool_calls_json, tool_name, finish_reason = msg_row
                    msg = {"role": role, "content": content or "", "finish_reason": finish_reason}
                    if tool_calls_json:
                        try:
                            msg["tool_calls"] = json.loads(tool_calls_json)
                        except Exception:
                            pass
                    if tool_name:
                        msg["tool_name"] = tool_name
                    session_data["messages"].append(msg)

                conn.close()
                return session_data
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                continue

        # Fallback: legacy JSON files
        profile_path = HERMES_DIR / "profiles" / profile / "sessions" / f"session_{session_id}.json"
        global_path = HERMES_DIR / "sessions" / f"session_{session_id}.json"
        src = profile_path if profile_path.exists() else global_path
        if not src.exists():
            return None
        try:
            return json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _extract_session_trace(history: list) -> Dict[str, Any]:
        """Extract rounds, total_tokens, timing, and tool_calls from session data."""
        empty = {"rounds": 0, "total_tokens": 0, "duration_seconds": 0, "tool_calls": []}
        if not history:
            return empty
        session = history[0] if isinstance(history[0], dict) else {}
        messages = session.get("messages", [])
        if not messages:
            return empty

        rounds = len(messages)
        total_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4

        start = session.get("session_start", "")
        end = session.get("last_updated", "")
        duration = 0.0
        if start and end:
            from datetime import datetime
            try:
                t0 = datetime.fromisoformat(start)
                t1 = datetime.fromisoformat(end)
                duration = (t1 - t0).total_seconds()
            except Exception:
                pass

        tool_calls = {}
        for m in messages:
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    tool_calls[name] = tool_calls.get(name, 0) + 1

        return {
            "rounds": rounds,
            "total_tokens": total_tokens,
            "duration_seconds": round(duration, 2),
            "tool_calls": tool_calls,
        }

    # ------------------------------------------------------------------
    # interact
    # ------------------------------------------------------------------

    def interact(
        self,
        query: str,
        system_prompt: str,
        run_code_func: Any,
        path_info: Dict[str, str],
    ) -> Dict[str, Any]:
        real_input_dir = path_info.get("real_input_dir", self.data_root_path)

        # task_id: prefer explicit field, then derive from workspace_dir last segment
        task_id = path_info.get("task_id")
        if not task_id and path_info.get("workspace_dir"):
            task_id = Path(path_info["workspace_dir"]).name
        if not task_id:
            task_id = str(abs(hash(query)) % 10**9)

        # Determine workspace directory
        if path_info.get("workspace_dir"):
            workspace_dir = Path(path_info["workspace_dir"])
        else:
            workspace_dir = DEFAULT_WORKSPACE_DIR / self.save_name / str(task_id)

        # Collect input files to symlink: real_input_dir is already the task-level dir
        # (e.g. data/data_visualization/Q001/), so take all files directly inside it
        real_input_files = []
        if real_input_dir and os.path.isdir(real_input_dir):
            real_input_files = [
                str(p) for p in Path(real_input_dir).iterdir() if p.is_file()
            ]

        work_dir = self._prepare_workspace(workspace_dir, real_input_files)

        # Rewrite virtual mount paths in query to workspace-relative paths.
        # hermes runs with cwd=work_dir, so ./inputs and ./outputs resolve correctly.
        hermes_query = query
        hermes_query = hermes_query.replace(
            path_info.get("mnt_input_dir", "/mnt/data"), "./inputs"
        )
        hermes_query = hermes_query.replace(
            path_info.get("mnt_output_dir", "/mnt/result"), "./outputs"
        )
        # fallback: some runners use /mnt/output instead of /mnt/result
        hermes_query = hermes_query.replace("/mnt/output", "./outputs")

        # Per-task profile to avoid SQLite lock contention under concurrency
        task_profile = re.sub(r'[^a-z0-9_-]', '-', f"{self.save_name}_{task_id}".lower())[:64]
        self._setup_profile(self._api_key, self._base_url, self.model_name, task_profile, self._provider)

        cmd = [
            "hermes",
            "--profile", task_profile,
            "chat",
            "--query", hermes_query,
            "--yolo",
            "-Q",
            "--max-turns", str(self.max_rounds),
            "--model", self.model_name,
        ]

        session_id = None
        model_response = ""
        history = []

        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = self._api_key
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            m = re.search(r"session_id:\s*(\S+)", stderr)
            if m:
                session_id = m.group(1)
            if not session_id:
                m = re.search(r"session_id:\s*(\S+)", stdout)
                if m:
                    session_id = m.group(1)
            # Also match "session=<id>" format used in some hermes versions
            if not session_id:
                m = re.search(r"\bsession=([0-9a-f_]+)\b", stderr + "\n" + stdout)
                if m:
                    session_id = m.group(1)

            if session_id:
                session_data = self._load_session(session_id, task_profile)
                if session_data:
                    history = session_data if isinstance(session_data, list) else [session_data]
                    # Extract model_response from last assistant message in history
                    messages = session_data.get("messages", [])
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant" and msg.get("content", "").strip():
                            model_response = msg["content"].strip()
                            break

            # Fallback to stdout if history extraction failed
            if not model_response:
                model_response = stdout.strip()

            if proc.returncode != 0 and not model_response:
                model_response = f"Error: hermes exited with code {proc.returncode}. stderr: {stderr[:500]}"

        except subprocess.TimeoutExpired:
            model_response = f"Error: hermes timed out after {self.timeout}s"
        except Exception as e:
            model_response = f"Error: {e}"

        trace = self._extract_session_trace(history)

        return {
            "model_response": model_response,
            "history": history,
            "total_tokens": trace["total_tokens"],
            "rounds": trace["rounds"],
            "duration_seconds": trace["duration_seconds"],
            "tool_calls": trace["tool_calls"],
            "session_id": session_id,
            "profile": task_profile,
        }

    # ------------------------------------------------------------------
    # continue_session — resume an existing session with a new query
    # ------------------------------------------------------------------

    def continue_session(
        self,
        session_id: str,
        query: str,
        work_dir: str,
        profile: str = None,
    ) -> Dict[str, Any]:
        use_profile = profile or self.profile_name
        cmd = [
            "hermes",
            "--profile", use_profile,
            "--resume", session_id,
            "chat",
            "--query", query,
            "--yolo",
            "-Q",
            "--max-turns", str(self.max_rounds),
            "--model", self.model_name,
        ]

        model_response = ""
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = self._api_key
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url

        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            model_response = (proc.stdout or "").strip()

            if proc.returncode != 0 and not model_response:
                model_response = f"Error: hermes exited with code {proc.returncode}. stderr: {(proc.stderr or '')[:500]}"

        except subprocess.TimeoutExpired:
            model_response = f"Error: hermes timed out after {self.timeout}s"
        except Exception as e:
            model_response = f"Error: {e}"

        return {
            "model_response": model_response,
            "session_id": session_id,
        }
