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
    prompt = os.environ["HERMES_PROMPT"]
    model = os.environ["HERMES_MODEL"]
    max_iters = int(os.environ.get("HERMES_MAX_ITERATIONS") or "60")
    provider = os.environ.get("HERMES_PROVIDER") or None

    from run_agent import AIAgent

    print(f"[oneshot] task_id={task_id} model={model} max_iter={max_iters} provider={provider or '(default)'}", flush=True)
    print(f"[oneshot] cwd={WORKSPACE} output={OUTPUT_DIR}", flush=True)

    start = time.perf_counter()
    result: dict = {}
    err = None
    try:
        agent = AIAgent(
            base_url=os.environ["HERMES_BASE_URL"],
            api_key=os.environ["HERMES_API_KEY"],
            model=model,
            provider=provider,
            max_iterations=max_iters,
            max_tokens=65536,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
            quiet_mode=True,
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
        elapsed = time.perf_counter() - start
        try:
            (OUTPUT_DIR / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            with (OUTPUT_DIR / "messages.jsonl").open("w", encoding="utf-8") as f:
                for msg in result.get("messages", []):
                    f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
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
