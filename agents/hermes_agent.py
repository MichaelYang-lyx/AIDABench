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
        **kwargs,
    ):
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.save_name = save_name
        self.max_rounds = max_rounds
        self.timeout = max_rounds * 120

        self._setup_profile(api_key, base_url, model_name, save_name)

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

    def _setup_profile(self, api_key: str, base_url: str, model_name: str, profile: str):
        # Create profile (ignore error if already exists)
        self._run_hermes_cmd(["profile", "create", profile], check=False)

        configs = [
            ("model.default", model_name),
            ("model.provider", "custom"),
            ("model.base_url", base_url),
            ("model.api_key", api_key),
        ]
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

    def _load_session(self, session_id: str) -> Any:
        # hermes stores sessions under ~/.hermes/profiles/<profile>/sessions/ when using a profile
        profile_path = HERMES_DIR / "profiles" / self.save_name / "sessions" / f"session_{session_id}.json"
        global_path = HERMES_DIR / "sessions" / f"session_{session_id}.json"
        src = profile_path if profile_path.exists() else global_path
        if not src.exists():
            return None
        try:
            return json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            return None

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

        # task_id: prefer explicit field, then derive from real_output_dir last segment
        task_id = path_info.get("task_id")
        if not task_id and real_output_dir:
            task_id = Path(real_output_dir).name
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

        cmd = [
            "hermes",
            "--profile", self.save_name,
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

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            m = re.search(r"session_id:\s*(\S+)", stderr)
            if m:
                session_id = m.group(1)

            model_response = stdout.strip()

            if session_id:
                session_data = self._load_session(session_id)
                if session_data:
                    history = session_data if isinstance(session_data, list) else [session_data]

            if proc.returncode != 0 and not model_response:
                model_response = f"Error: hermes exited with code {proc.returncode}. stderr: {stderr[:500]}"

        except subprocess.TimeoutExpired:
            model_response = f"Error: hermes timed out after {self.timeout}s"
        except Exception as e:
            model_response = f"Error: {e}"

        return {
            "model_response": model_response,
            "history": history,
            "total_tokens": 0,
            "rounds": 0,
            "session_id": session_id,
        }
