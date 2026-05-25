"""
HermesDockerAgent: 容器化版本的 HermesAgent。

与 HermesAgent（直接调用本地 hermes CLI）不同，此版本通过 Docker 容器运行每个任务：
- 每个任务启动一个独立的 hermes-eval 容器（与 hermes-auto 的 WCB 风格相同）
- workspace 通过卷挂载传入容器
- 输出文件（result.json / messages.jsonl / meta.json）通过卷挂载从容器取回
- 环境隔离、无本地 hermes 安装依赖

前提条件:
    - 本机安装 Docker
    - hermes-eval 镜像已构建（参考 hermes-auto/docker/build.sh）
    - 容器内有 /opt/hermes-eval/oneshot.py 入口

环境变量 / 构造参数:
    EVAL_IMAGE  : 使用的 Docker 镜像名（默认 hermes-eval:latest）
    HERMES_BASE_URL / HERMES_API_KEY : LLM API 参数
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "workspaces"

# --------------------------------------------------------------------------- #
# 默认镜像名，可通过环境变量或构造参数覆盖
# --------------------------------------------------------------------------- #
DEFAULT_EVAL_IMAGE = os.environ.get("EVAL_IMAGE", "hermes-eval:latest")


class HermesDockerAgent:
    """通过 hermes-eval Docker 容器运行 hermes 的 agent。

    接口与 HermesAgent 保持一致（interact / continue_session），
    但底层不依赖本地 hermes CLI，改为 docker run + oneshot.py。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        data_root_path: str,
        save_name: str,
        max_rounds: int = 90,
        provider: str = None,
        eval_image: str = None,
        timeout_seconds: int = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.save_name = save_name
        self.max_rounds = max_rounds
        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider or "custom"
        self.eval_image = eval_image or DEFAULT_EVAL_IMAGE
        # 默认超时：max_rounds * 120s（每轮 2 分钟上限），最少 300s
        self.timeout_seconds = timeout_seconds or max(300, max_rounds * 120)

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _prepare_workspace(self, workspace_dir: Path, real_input_files: list[str]) -> Path:
        """创建 workspace 目录并将输入文件复制进 inputs/ 子目录。"""
        workspace_dir.mkdir(parents=True, exist_ok=True)
        inputs_dir = workspace_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)

        for src in real_input_files:
            src_path = Path(src)
            if src_path.exists():
                dst = inputs_dir / src_path.name
                if dst.is_symlink():
                    dst.unlink()
                if not dst.exists() or dst.stat().st_size == 0:
                    shutil.copy2(str(src_path), str(dst))

        return workspace_dir

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docker_image_present(image: str) -> bool:
        r = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True,
        )
        return r.returncode == 0

    def _run_container(
        self,
        prompt: str,
        workspace_host: Path,
        output_host: Path,
        task_id: str,
        prefill_messages: Optional[list] = None,
    ) -> dict:
        """启动 hermes-eval 容器执行单次任务，返回 result / meta 解析结果。"""
        container_name = f"hermes-eval-{re.sub(r'[^a-z0-9-]', '-', task_id.lower())[:48]}-{uuid.uuid4().hex[:6]}"

        # 如果提供了 prefill 消息历史，写到 workspace 内供容器读取
        prefill_container_path = None
        if prefill_messages:
            prefill_host = workspace_host / ".hermes_prefill.jsonl"
            with prefill_host.open("w", encoding="utf-8") as f:
                for m in prefill_messages:
                    f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
            prefill_container_path = "/workspace/.hermes_prefill.jsonl"

        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            # 让容器内文件归属与宿主相同（避免权限问题）
            "-e", f"HERMES_UID={os.getuid()}",
            "-e", f"HERMES_GID={os.getgid()}",
            # 卷挂载
            "-v", f"{workspace_host.resolve()}:/workspace",
            "-v", f"{output_host.resolve()}:/eval_output",
            # 任务参数
            "-e", f"HERMES_PROMPT={prompt}",
            "-e", f"HERMES_MODEL={self.model_name}",
            "-e", f"HERMES_BASE_URL={self._base_url}",
            "-e", f"HERMES_API_KEY={self._api_key}",
            "-e", f"HERMES_MAX_ITERATIONS={self.max_rounds}",
            "-e", f"HERMES_TASK_ID={task_id}",
            "-e", f"HERMES_PROVIDER={self._provider}",
        ]
        if prefill_container_path:
            docker_cmd += ["-e", f"HERMES_PREFILL_PATH={prefill_container_path}"]
        docker_cmd += [
            self.eval_image,
            "python3", "/opt/hermes-eval/oneshot.py",
        ]

        output_host.mkdir(parents=True, exist_ok=True)
        log_path = output_host / "container.log"

        logger.info("[%s] docker run image=%s workspace=%s", task_id, self.eval_image, workspace_host)
        start = time.perf_counter()
        container_error: Optional[str] = None

        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(docker_cmd, stdout=logf, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=self.timeout_seconds)
                if rc != 0:
                    container_error = f"container exited with code {rc}"
            except subprocess.TimeoutExpired:
                container_error = f"timeout after {self.timeout_seconds}s"
                logger.warning("[%s] %s, killing container", task_id, container_error)
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                proc.wait()

        elapsed = time.perf_counter() - start
        logger.info("[%s] container done in %.1fs (error=%s)", task_id, elapsed, container_error)

        # 读取容器产出
        result: dict = {}
        meta: dict = {}
        messages: list = []

        result_path = output_host / "result.json"
        meta_path = output_host / "meta.json"
        messages_path = output_host / "messages.jsonl"

        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[%s] failed to parse result.json: %s", task_id, e)

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[%s] failed to parse meta.json: %s", task_id, e)

        if messages_path.exists():
            try:
                messages = [
                    json.loads(line)
                    for line in messages_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as e:
                logger.warning("[%s] failed to parse messages.jsonl: %s", task_id, e)

        return {
            "result": result,
            "meta": meta,
            "messages": messages,
            "container_error": container_error,
            "elapsed_seconds": round(elapsed, 2),
            "log_path": str(log_path),
            "output_dir": str(output_host),
        }

    # ------------------------------------------------------------------
    # Public interface — mirrors HermesAgent.interact
    # ------------------------------------------------------------------

    def interact(
        self,
        query: str,
        system_prompt: str,
        run_code_func: Any,
        path_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """运行一次完整的 hermes 任务（容器内），返回与 HermesAgent.interact 相同结构的字典。"""
        if not shutil.which("docker"):
            raise RuntimeError("docker CLI not found in PATH")
        if not self._docker_image_present(self.eval_image):
            raise RuntimeError(
                f"Docker image '{self.eval_image}' not found locally. "
                "Build it first: cd hermes-auto && bash docker/build.sh"
            )

        real_input_dir = path_info.get("real_input_dir", self.data_root_path)

        # 确定 task_id
        task_id = path_info.get("task_id")
        if not task_id and path_info.get("workspace_dir"):
            task_id = Path(path_info["workspace_dir"]).name
        if not task_id:
            task_id = str(abs(hash(query)) % 10 ** 9)

        # 确定 workspace 目录（宿主侧）
        if path_info.get("workspace_dir"):
            workspace_dir = Path(path_info["workspace_dir"])
        else:
            workspace_dir = DEFAULT_WORKSPACE_DIR / self.save_name / str(task_id)

        # 收集输入文件
        real_input_files = []
        if real_input_dir and os.path.isdir(real_input_dir):
            real_input_files = [
                str(p) for p in Path(real_input_dir).iterdir() if p.is_file()
            ]

        work_dir = self._prepare_workspace(workspace_dir, real_input_files)

        # 替换虚拟路径为容器内路径（容器 cwd=/workspace）
        docker_query = query
        docker_query = docker_query.replace(
            path_info.get("mnt_input_dir", "/mnt/data"), "/workspace/inputs"
        )
        docker_query = docker_query.replace(
            path_info.get("mnt_output_dir", "/mnt/result"), "/workspace/outputs"
        )
        docker_query = docker_query.replace("/mnt/output", "/workspace/outputs")

        # eval_output 目录（容器产出落到 workspace 旁边的 _eval_output 子目录）
        run_id = uuid.uuid4().hex[:6]
        output_host = workspace_dir.parent / f"{workspace_dir.name}_eval_{run_id}"

        container_result = self._run_container(
            prompt=docker_query,
            workspace_host=work_dir,
            output_host=output_host,
            task_id=task_id,
        )

        result = container_result["result"]
        meta = container_result["meta"]
        messages = container_result["messages"]

        # 从消息列表提取最后一条 assistant 回复
        model_response = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    model_response = content.strip()
                    break

        if not model_response:
            model_response = result.get("final_response", "")
        if not model_response and container_result.get("container_error"):
            model_response = f"Error: {container_result['container_error']}"

        # 统计 tool_calls
        tool_calls: Dict[str, int] = {}
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    name = tc.get("function", {}).get("name", "unknown")
                    tool_calls[name] = tool_calls.get(name, 0) + 1

        return {
            "model_response": model_response,
            "history": messages,
            "total_tokens": meta.get("total_tokens") or result.get("total_tokens", 0),
            "rounds": meta.get("api_calls") or len(messages),
            "duration_seconds": meta.get("elapsed_seconds") or container_result["elapsed_seconds"],
            "tool_calls": tool_calls,
            "session_id": f"docker-{task_id}-{uuid.uuid4().hex[:8]}",
            "profile": None,
            # 额外字段
            "output_dir": container_result["output_dir"],
            "container_log": container_result["log_path"],
            "container_error": container_result["container_error"],
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # continue_session — 容器化的"伪续接"：通过 prefill_messages 注入消息历史
    # ------------------------------------------------------------------

    def continue_session(
        self,
        session_id: str,
        query: str,
        work_dir: str,
        profile: str = None,
        messages_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """通过 prefill_messages 让新容器复用之前的会话上下文。

        与本地 HermesAgent 真正的 session 续接不同，这里每次启动新容器，
        但把上一轮的消息历史作为 prefill_messages 注入到 AIAgent，从而获得
        相同效果。session_id 仅用于日志记录。
        """
        if not messages_history:
            logger.warning(
                "[%s] continue_session called without messages_history; "
                "agent will start fresh with no prior context",
                session_id,
            )

        workspace_dir = Path(work_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # task_id 用于命名容器和 eval_output 目录
        task_id = f"resume-{session_id[:24]}" if session_id else f"resume-{uuid.uuid4().hex[:8]}"
        run_id = uuid.uuid4().hex[:6]
        output_host = workspace_dir.parent / f"{workspace_dir.name}_eval_{run_id}"

        container_result = self._run_container(
            prompt=query,
            workspace_host=workspace_dir,
            output_host=output_host,
            task_id=task_id,
            prefill_messages=messages_history,
        )

        result = container_result["result"]
        meta = container_result["meta"]
        messages = container_result["messages"]

        # 从消息列表提取最后一条 assistant 回复
        model_response = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    model_response = content.strip()
                    break

        if not model_response:
            model_response = result.get("final_response", "")
        if not model_response and container_result.get("container_error"):
            model_response = f"Error: {container_result['container_error']}"

        return {
            "model_response": model_response,
            "history": messages,
            "session_id": session_id,
            "container_error": container_result["container_error"],
            "elapsed_seconds": container_result["elapsed_seconds"],
            "output_dir": container_result["output_dir"],
            "meta": meta,
        }
