#!/usr/bin/env python3
"""
Container-side oneshot entry point. Runs ONE task end-to-end inside a
pre-built hermes image, then exits. Reads task config from env vars,
writes outputs under /eval_output/ (mounted from host).

Env vars (all required unless noted):
    HERMES_PROMPT          : the user prompt to send the agent
    HERMES_MODEL           : model id (e.g. anthropic/claude-opus-4.6)
    HERMES_BASE_URL        : LLM API base url (e.g. https://openrouter.ai/api/v1)
    HERMES_API_KEY         : LLM API key
    HERMES_MAX_ITERATIONS  : optional, default 60
    HERMES_TASK_ID         : task identifier for logging (optional)

Filesystem contract:
    /workspace/      : mounted from host, agent reads/writes here
    /eval_output/    : mounted from host, oneshot writes result.json/messages.jsonl/meta.json here

Exit codes:
    0  - completed (regardless of task success)
    2  - missing required env var
    3  - agent raised exception (output still saved if possible)
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/opt/hermes")

# Convert SIGTERM into SystemExit so the finally block (chown-back) runs.
# run_eval.py uses `docker stop -t 30` on timeout, which sends SIGTERM first
# and only SIGKILLs after the grace window — that's what makes this catchable.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(124))

REQUIRED_ENV = ("HERMES_PROMPT", "HERMES_MODEL", "HERMES_BASE_URL", "HERMES_API_KEY")
OUTPUT_DIR = Path("/eval_output")
WORKSPACE = Path("/workspace")


def _missing_env() -> list[str]:
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


def _with_workspace_constraint(prompt: str) -> str:
    """Prepend a constraint pinning task deliverables to /workspace.

    Only /workspace (the cwd) and /eval_output are bind-mounted back to the
    host; everything the agent writes elsewhere (e.g. its home dir /opt/hermes,
    or /root) is discarded when the container exits. Some tasks — notably the
    p5.js "save it to an html file" prompts — lead every model to write the
    deliverable into its home dir instead of the cwd, so the result file is
    silently lost. The preamble is phrased conditionally ("若题目要求产出交付
    文件") so it's a no-op for pure Q&A tasks but pins the output dir whenever a
    task does produce files — uniformly, without depending on any bundled
    skill's behaviour.
    """
    preamble = (
        "你的当前工作目录是 /workspace，这是唯一会被保留的目录。"
        "若题目要求产出交付文件（最终的 html/脚本/文档/图片等），"
        "则这些文件必须保存在 /workspace 下"
        "（用相对路径或显式以 /workspace/ 开头的绝对路径），"
        "不要写到家目录、/opt、/root 或 /tmp 等其它位置——那些目录在任务结束后会被丢弃，文件将会丢失。\n\n"
        "任务如下：\n"
    )
    return preamble + prompt


# Shared store for per-API-call request ids captured by the monkeypatch below.
# Each entry: {"index", "request_id", "source", "model", "observed_at"}.
_REQUEST_IDS: list = []


def _extract_request_id(result):
    """Best-effort request id from an OpenAI-SDK chat.completions result.

    Tries, in order:
      1. response header ``x-request-id`` (works for both ChatCompletion and
         streaming ``Stream`` objects — the latter exposes the initial httpx
         response via ``.response`` before chunks are consumed);
      2. the response body ``id`` field (non-stream ChatCompletion; some
         providers — e.g. SenseNova — also mirror it as ``request_id``).

    Returns ``(request_id, source)`` or ``(None, None)``. Never raises, so a
    provider that returns neither simply yields ``None`` and is skipped.
    """
    # 1) HTTP response header (universal across providers that set it).
    try:
        resp = getattr(result, "response", None)  # Stream.response (httpx.Response)
        headers = getattr(resp, "headers", None)
        if headers is not None:
            rid = headers.get("x-request-id") or headers.get("X-Request-Id")
            if rid:
                return rid, "header"
    except Exception:
        pass
    # 2) Response body fields (non-stream ChatCompletion).
    try:
        rid = getattr(result, "request_id", None) or getattr(result, "id", None)
        if rid:
            return rid, "body"
    except Exception:
        pass
    return None, None


def _install_request_id_capture() -> bool:
    """Monkeypatch openai's (Async)Completions.create to record request ids.

    Defensive: imported lazily and wrapped in try/except, so if the openai
    SDK layout differs or isn't present, the agent runs unchanged. Each call's
    request id (or absence of one) is appended to ``_REQUEST_IDS``; the wrapper
    re-raises any error from the underlying create() untouched, so behavior for
    every provider — including those that return no request id — is identical
    to the unpatched path apart from the bookkeeping.
    """
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions
    except Exception as exc:  # SDK missing or moved — skip silently.
        print(f"[oneshot] request-id capture disabled ({exc})", flush=True)
        return False

    def _record(result, model):
        try:
            rid, source = _extract_request_id(result)
            if rid:
                _REQUEST_IDS.append({
                    "index": len(_REQUEST_IDS),
                    "request_id": rid,
                    "source": source,
                    "model": model,
                    "observed_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                })
        except Exception:
            pass

    _orig = Completions.create

    def _patched(self, *args, **kwargs):
        result = _orig(self, *args, **kwargs)
        _record(result, kwargs.get("model"))
        return result

    try:
        Completions.create = _patched
    except Exception as exc:
        print(f"[oneshot] request-id capture: failed to patch sync create ({exc})", flush=True)
        return False

    # Async path is best-effort; most hermes runtimes use the sync client, but
    # patch it too so nothing silently slips through if that changes.
    try:
        _orig_async = AsyncCompletions.create

        async def _patched_async(self, *args, **kwargs):
            result = await _orig_async(self, *args, **kwargs)
            _record(result, kwargs.get("model"))
            return result

        AsyncCompletions.create = _patched_async
    except Exception:
        pass

    print("[oneshot] request-id capture installed", flush=True)
    return True


# Subagent (delegate_task) trace capture
# ---------------------------------------------------------------------------
# hermes' delegate_tool spawns child AIAgents whose intermediate tool calls and
# reasoning are deliberately hidden from the parent context — the parent only
# ever sees the delegation call + a `summary`/`tool_trace` rollup, and the
# child's full `messages` list is dropped when `_run_single_child` returns.
# Because oneshot's live-snapshot only walks the PARENT's `_session_messages`,
# none of the subagent activity reaches /eval_output by default.
#
# We wrap `tools.delegate_tool._run_single_child` (the single place that owns
# the child agent and its result) and, right before its dict is returned,
# persist the child's full conversation to
#   /eval_output/subagents/<subagent_id>.jsonl   (one message per line)
# plus an append-only `/eval_output/subagents/index.jsonl` rollup so a reader
# can enumerate every subagent of the run without globbing. Defensive: any
# failure is swallowed so delegation behaves exactly as before.
def _install_subagent_trace_capture(output_dir: Path) -> bool:
    try:
        import tools.delegate_tool as _dt
    except Exception as exc:
        print(f"[oneshot] subagent-trace capture disabled ({exc})", flush=True)
        return False

    if not hasattr(_dt, "_run_single_child"):
        print("[oneshot] subagent-trace capture disabled (_run_single_child missing)", flush=True)
        return False

    sub_dir = output_dir / "subagents"
    index_path = sub_dir / "index.jsonl"
    _seq = {"n": 0}
    _lock = threading.Lock()
    _orig_run_single_child = _dt._run_single_child

    def _persist(child, goal, result, entry):
        try:
            sub_dir.mkdir(parents=True, exist_ok=True)
            # subagent_id is set on the child by _run_single_child's caller
            # (`child._subagent_id = sa-<idx>-<hex8>`). Fall back to a counter
            # so two children never collide even if the id is missing.
            sid = getattr(child, "_subagent_id", None)
            if not isinstance(sid, str) or not sid:
                with _lock:
                    _seq["n"] += 1
                    sid = f"sa-unknown-{_seq['n']}"
            messages = result.get("messages") if isinstance(result, dict) else None
            if not isinstance(messages, list):
                messages = []
            trace_path = sub_dir / f"{sid}.jsonl"
            with trace_path.open("w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
            meta = {
                "subagent_id": sid,
                "parent_subagent_id": getattr(child, "_parent_subagent_id", None),
                "task_index": entry.get("task_index") if isinstance(entry, dict) else None,
                "depth": getattr(child, "_delegate_depth", None),
                "role": getattr(child, "_delegate_role", None),
                "model": getattr(child, "model", None) if isinstance(getattr(child, "model", None), str) else None,
                "goal": goal,
                "status": entry.get("status") if isinstance(entry, dict) else None,
                "exit_reason": entry.get("exit_reason") if isinstance(entry, dict) else None,
                "api_calls": entry.get("api_calls") if isinstance(entry, dict) else None,
                "duration_seconds": entry.get("duration_seconds") if isinstance(entry, dict) else None,
                "num_messages": len(messages),
                "summary": (entry.get("summary") if isinstance(entry, dict) else None),
                "trace_file": trace_path.name,
                "observed_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            }
            with _lock:
                with index_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False, default=str) + "\n")
            print(f"[oneshot] subagent trace saved: {sid} ({len(messages)} msgs, status={meta['status']})", flush=True)
        except Exception as exc:
            print(f"[oneshot] subagent-trace persist failed: {exc}", flush=True)

    def _patched_run_single_child(task_index, goal, child=None, parent_agent=None, **kwargs):
        # _run_single_child builds `result` internally and discards it; we
        # can't reach that local. Instead we capture the child's conversation
        # from the child agent itself after the call returns (run_conversation
        # leaves the full history on the agent), pairing it with the returned
        # entry for status/metadata.
        entry = _orig_run_single_child(task_index, goal, child=child, parent_agent=parent_agent, **kwargs)
        try:
            # The child agent retains the full conversation in _session_messages;
            # wrap it into the same {"messages": [...]} shape _persist expects.
            msgs = list(getattr(child, "_session_messages", []) or []) if child is not None else []
            _persist(child, goal, {"messages": msgs}, entry if isinstance(entry, dict) else {})
        except Exception as exc:
            print(f"[oneshot] subagent-trace wrapper failed: {exc}", flush=True)
        return entry

    try:
        _dt._run_single_child = _patched_run_single_child
    except Exception as exc:
        print(f"[oneshot] subagent-trace capture: failed to patch ({exc})", flush=True)
        return False

    print("[oneshot] subagent-trace capture installed", flush=True)
    return True


def main() -> int:
    missing = _missing_env()
    if missing:
        sys.stderr.write(f"oneshot: missing env vars: {missing}\n")
        return 2

    os.environ["TERMINAL_ENV"] = "local"
    os.environ.setdefault("TERMINAL_CWD", str(WORKSPACE))
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_id = os.environ.get("HERMES_TASK_ID") or f"task_{int(time.time())}"
    prompt = _with_workspace_constraint(os.environ["HERMES_PROMPT"])
    model = os.environ["HERMES_MODEL"]
    max_iters = int(os.environ.get("HERMES_MAX_ITERATIONS") or "60")
    provider = os.environ.get("HERMES_PROVIDER") or None

    # 对非 custom provider（如 anthropic），剥掉 base_url 末尾的 /v1
    # 否则 Anthropic SDK 会拼成 /v1/v1/messages 触发 404
    base_url = os.environ["HERMES_BASE_URL"]
    if provider and provider != "custom" and base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

    prefill_messages: list = []
    prefill_path = os.environ.get("HERMES_PREFILL_PATH")
    if prefill_path and Path(prefill_path).is_file():
        raw_prefill: list = []
        with open(prefill_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_prefill.append(json.loads(line))
                except Exception:
                    pass
        # Preserve tool_calls + tool replies so the model retains full context
        # (queries it ran, results/files/charts it produced). Two-pass cleanup:
        #   Pass 1: collect tool_call ids that have a matching tool reply.
        #   Pass 2: emit normalized messages, dropping unpaired tool calls/replies
        #   (Anthropic / OpenAI APIs reject messages with orphan tool_call_id).
        declared_ids: set = set()
        replied_ids: set = set()
        for m in raw_prefill:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "assistant":
                for tc in (m.get("tool_calls") or []):
                    tid = tc.get("id") or tc.get("call_id")
                    if tid:
                        declared_ids.add(tid)
            elif role == "tool":
                tid = m.get("tool_call_id")
                if tid:
                    replied_ids.add(tid)
        valid_ids = declared_ids & replied_ids

        def _norm_tc(tc: dict):
            tid = tc.get("id") or tc.get("call_id")
            if not tid or tid not in valid_ids:
                return None
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            return {
                "id": tid,
                "type": tc.get("type", "function"),
                "function": {"name": fn.get("name", ""), "arguments": args},
            }

        for m in raw_prefill:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                if isinstance(content, str) and content.strip():
                    prefill_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                content_str = content if isinstance(content, str) else ""
                kept_tcs = []
                for tc in (m.get("tool_calls") or []):
                    norm = _norm_tc(tc)
                    if norm is not None:
                        kept_tcs.append(norm)
                if kept_tcs:
                    prefill_messages.append({
                        "role": "assistant",
                        "content": content_str,
                        "tool_calls": kept_tcs,
                    })
                elif content_str.strip():
                    prefill_messages.append({"role": "assistant", "content": content_str})
                # else: empty assistant with no valid tool_calls -> drop
            elif role == "tool":
                tid = m.get("tool_call_id")
                if tid and tid in valid_ids:
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    prefill_messages.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tid,
                    })
        print(
            f"[oneshot] loaded {len(prefill_messages)} prefill messages "
            f"(from {len(raw_prefill)} raw, {len(valid_ids)} paired tool calls) at {prefill_path}",
            flush=True,
        )

    # Capture per-call request ids (provider response header / body) without
    # touching hermes internals. No-op for providers that return none.
    _install_request_id_capture()

    # Capture full traces of any delegate_task subagents into
    # /eval_output/subagents/. No-op if the task never delegates.
    _install_subagent_trace_capture(OUTPUT_DIR)

    from run_agent import AIAgent

    print(f"[oneshot] task_id={task_id} model={model} max_iter={max_iters} provider={provider or '(default)'}", flush=True)
    print(f"[oneshot] cwd={WORKSPACE} output={OUTPUT_DIR}", flush=True)

    start = time.perf_counter()
    result: dict = {}
    err = None
    agent = None  # bound inside the try once AIAgent() returns

    # Background snapshot of agent._session_messages so that timeouts /
    # SIGKILL / OOM still leave an observable trace on disk. Atomic
    # rename keeps the file always-readable; daemon thread dies with
    # the process if we never reach the finally block.
    #
    # Each emitted message is decorated with `_observed_at` — the UTC
    # timestamp of the FIRST snapshot that saw this message. Granularity
    # is bounded by the snapshot interval below (1s). Stable keys are
    # derived from tool_call ids when available so that hermes' context
    # compression (which can drop middle turns) doesn't reshuffle stamps.
    live_path = OUTPUT_DIR / "live_messages.jsonl"
    live_tmp = OUTPUT_DIR / "live_messages.jsonl.tmp"
    stop_snapshot = threading.Event()
    observed_at: dict = {}

    def _msg_key(m: dict):
        role = m.get("role")
        if role == "tool":
            tid = m.get("tool_call_id")
            if tid:
                return ("tool", tid)
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                tid = tc.get("id") or tc.get("call_id")
                if tid:
                    return ("assistant", tid)
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return (role, hash(content[:512]))

    def _snapshot_once():
        if agent is None:
            return
        try:
            msgs = list(getattr(agent, "_session_messages", []) or [])
            now = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
            with live_tmp.open("w", encoding="utf-8") as f:
                prev_dt = None
                for m in msgs:
                    ts = observed_at.setdefault(_msg_key(m), now)
                    cur_dt = datetime.fromisoformat(ts.rstrip("Z"))
                    delta = 0.0 if prev_dt is None else round(
                        (cur_dt - prev_dt).total_seconds(), 3)
                    f.write(json.dumps(
                        {**m, "_observed_at": ts, "_delta_seconds": delta},
                        ensure_ascii=False, default=str) + "\n")
                    prev_dt = cur_dt
            os.replace(live_tmp, live_path)
        except Exception:
            pass

    def _snapshot_loop():
        while not stop_snapshot.wait(1.0):
            _snapshot_once()

    snapshot_thread = threading.Thread(
        target=_snapshot_loop, name="live-snapshot", daemon=True
    )
    snapshot_thread.start()

    try:
        agent = AIAgent(
            base_url=base_url,
            api_key=os.environ["HERMES_API_KEY"],
            model=model,
            provider=provider,
            max_iterations=max_iters,
            max_tokens=65536,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            quiet_mode=True,
            prefill_messages=prefill_messages,
        )
        # Strip "default" from tool schemas — Claude API rejects it as invalid
        # under JSON Schema draft 2020-12 for tool definitions.
        # Also ensure every "type": "object" has a "required" field — Claude API
        # rejects object schemas without it.
        def _sanitize_schema(obj):
            if isinstance(obj, dict):
                cleaned = {k: _sanitize_schema(v) for k, v in obj.items() if k != "default"}
                if cleaned.get("type") == "object" and "required" not in cleaned:
                    cleaned["required"] = []
                return cleaned
            if isinstance(obj, list):
                return [_sanitize_schema(i) for i in obj]
            return obj

        for tool in agent.tools:
            if "function" in tool and "parameters" in tool["function"]:
                tool["function"]["parameters"] = _sanitize_schema(tool["function"]["parameters"])

        result = agent.run_conversation(prompt, task_id=task_id)
    except Exception as exc:
        err = exc
        traceback.print_exc()
        result = {"error": str(exc), "completed": False}
    finally:
        # Stop the live-snapshot thread and flush one last time so the file
        # reflects the final state. Done before chown so the file gets the
        # right host UID along with the rest of /eval_output.
        stop_snapshot.set()
        snapshot_thread.join(timeout=2)
        _snapshot_once()

        elapsed = time.perf_counter() - start
        try:
            (OUTPUT_DIR / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            with (OUTPUT_DIR / "messages.jsonl").open("w", encoding="utf-8") as f:
                for msg in result.get("messages", []):
                    f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
            # One line per captured API call request id (empty file if the
            # provider returns none — readers should treat absence as "unknown").
            with (OUTPUT_DIR / "request_ids.jsonl").open("w", encoding="utf-8") as f:
                for rec in _REQUEST_IDS:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (OUTPUT_DIR / "meta.json").write_text(
                json.dumps({
                    "task_id": task_id,
                    "model": model,
                    "elapsed_seconds": round(elapsed, 2),
                    "api_calls": result.get("api_calls"),
                    "completed": result.get("completed"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "total_tokens": result.get("total_tokens"),
                    "estimated_cost_usd": result.get("estimated_cost_usd"),
                    "request_ids": [r["request_id"] for r in _REQUEST_IDS],
                    "last_request_id": _REQUEST_IDS[-1]["request_id"] if _REQUEST_IDS else None,
                    "error": str(err) if err else None,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            traceback.print_exc()
        # Container runs as root (entrypoint privilege drop is disabled in
        # Dockerfile.eval). Chown bind mounts back to the host user so
        # outputs stay manageable from the host side.
        host_uid = os.environ.get("HERMES_UID")
        host_gid = os.environ.get("HERMES_GID")
        if host_uid and host_gid:
            import subprocess
            subprocess.run(
                ["chown", "-R", f"{host_uid}:{host_gid}", str(OUTPUT_DIR), str(WORKSPACE)],
                check=False, capture_output=True,
            )
        print(f"[oneshot] done in {elapsed:.1f}s", flush=True)

    return 3 if err else 0


if __name__ == "__main__":
    sys.exit(main())
