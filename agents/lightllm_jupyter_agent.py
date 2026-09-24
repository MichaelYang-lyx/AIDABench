"""
LightLLM Jupyter Agent — no-skill variant of SkillJupyterAgent.

Supports:
  - lightllm /generate endpoint  (base_url ends with /generate)
  - vLLM /v1/completions endpoint (base_url ends with /completions)
  - GPT-6 Astra through the OpenAI Responses API
  - OpenAI-compatible chat completions (all other OpenAI-style models)

Same parsing logic as SkillJupyterAgent but without skill registration
in the system prompt.
"""

import os
import re
import sys
import json
import base64
import io
import subprocess
import requests
import time
import queue
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

from openai import OpenAI
from PIL import Image

try:
    from jupyter_client import KernelManager
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# Tool call format injected into lightllm system prompt (XML/ChatML path)
# ============================================================
MULTITOOL_TCALL_CONTENT = r"""# Tools

You have access to the following functions:

<tools>
<function>
<name>execute_code</name>
<description>Execute Python code in a persistent Jupyter environment. Variables defined in one call are available in subsequent calls.</description>
<parameters>
<parameter>
<name>code</name>
<type>string</type>
<description>Python code to execute.</description>
</parameter>
<required>["code"]</required>
</parameters>
</function>
<function>
<name>read_file</name>
<description>Read the contents of a file. Text files only. Output capped at 2000 lines / 50 KB. Use offset/limit for large files. Cannot read binary files (.xlsx, .docx, .pdf, .png …) — use execute_code with pandas/openpyxl instead.</description>
<parameters>
<parameter>
<name>path</name>
<type>string</type>
<description>File path to read.</description>
</parameter>
<required>["path"]</required>
</parameters>
</function>
<function>
<name>write_file</name>
<description>Write content to a file. Creates parent directories if needed.</description>
<parameters>
<parameter>
<name>path</name>
<type>string</type>
<description>File path to write to.</description>
</parameter>
<parameter>
<name>content</name>
<type>string</type>
<description>Content to write.</description>
</parameter>
<required>["path", "content"]</required>
</parameters>
</function>
<function>
<name>bash</name>
<description>Execute a shell command and return its output (truncated to ~8000 chars).</description>
<parameters>
<parameter>
<name>command</name>
<type>string</type>
<description>Shell command to execute.</description>
</parameter>
<parameter>
<name>working_dir</name>
<type>string</type>
<description>Optional working directory.</description>
</parameter>
<parameter>
<name>timeout</name>
<type>string</type>
<description>Timeout in seconds (default: 120).</description>
</parameter>
<required>["command"]</required>
</parameters>
</function>
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- You may provide optional reasoning BEFORE the function call, but NOT after
- If no function call is needed, answer directly
</IMPORTANT>"""

# ============================================================
# Native OpenAI tools schema (used by _call_openai_chat)
# ============================================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Execute Python code in a persistent Jupyter environment. "
                "Variables defined in one call are available in subsequent calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Text files only. Output capped at 2000 lines / 50 KB. "
                "Cannot read binary files (.xlsx, .docx, .pdf, .png) — use execute_code with pandas/openpyxl instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return its output (truncated to ~8000 chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": (
                "Inspect an image from the current task's inputs, scratch, or outputs. "
                "The image pixels are returned to the model. Optionally crop to a "
                "pixel box [left, top, right, bottom]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Image path under /mnt/data, /mnt/work, or the task output directory.",
                    },
                    "crop": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Optional pixel crop [left, top, right, bottom].",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# The Responses API uses a flatter function-tool schema than Chat Completions.
RESPONSES_TOOLS_SCHEMA = [
    {
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"].get("description", ""),
        "parameters": tool["function"]["parameters"],
    }
    for tool in TOOLS_SCHEMA
]


SYSTEM_PROMPT = """# Role
你是办公小浣熊，一个由商汤科技研发的专业、稳健的 AI 分析助手。
- 核心能力：写作与文本生成，数据分析与结构化推理，复杂任务拆解与执行规划
- 工作原则：理解用户的核心诉求，根据实际需要组合调用各种工具，在信息可验证、逻辑可追溯的前提下，高质量完成用户请求。

<path_mapping>
用户消息中的文件路径使用虚拟挂载路径，系统会自动重映射到真实路径：
- 输入文件：`/mnt/data/<filename>` → 实际数据目录
- 临时文件：`/mnt/work/<filename>` → 本题临时工作目录
- 输出文件：`/mnt/result/<filename>` → 实际输出目录
请直接使用用户消息中提供的路径，系统会透明地处理路径映射。
</path_mapping>

<workflow>
## 实际工作流程
1. 理解 (Understand)：仔细分析用户的请求，明确目标、相关文件和约束条件。
2. 规划 (Plan)：在脑海中形成一个高层级的分析计划，将复杂问题拆解成一系列可以由工具执行的、更小的逻辑步骤。
3. 单步执行 (Execute)：每次只调用一个工具。分析结果后决定是否需要下一步。若报错，需分析原因并尝试修复（Self-Healing）。如果发现数据有质量问题（如缺失值、异常值），应主动使用工具进行探查或清洗。
4. 结果综合 (Synthesize)：汇总数据，以 Markdown 表格形式呈现清晰、结构化的最终答案。
</workflow>
"""


class _ImageToolResult:
    """In-memory image payload sent to the model but omitted from saved traces."""

    def __init__(self, text: str, data_url: str):
        self.text = text
        self.data_url = data_url


# ============================================================
# Jupyter kernel executor
# ============================================================

class JupyterKernelExecutor:
    def __init__(self, timeout=60):
        # A benchmark run creates many kernels concurrently, sometimes from
        # multiple processes. TCP port discovery is racy across processes and
        # can make one client connect to another task's kernel. Give every
        # executor its own IPC namespace so endpoints cannot collide.
        self._ipc_dir = tempfile.TemporaryDirectory(prefix="aida_kernel_")
        self.km = KernelManager(
            transport="ipc",
            ip=os.path.join(self._ipc_dir.name, "kernel"),
            cache_ports=False,
        )
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.timeout = timeout
        self._wait_for_ready()
        self.last_active = time.time()

    def _wait_for_ready(self):
        try:
            self.kc.wait_for_ready(timeout=30)
        except Exception:
            try:
                self.kc.execute_interactive("", timeout=30)
            except Exception:
                pass

    def execute_code(self, code):
        msg_id = self.kc.execute(code)
        result = ""
        start_time = time.time()
        while True:
            remaining = self.timeout - (time.time() - start_time)
            if remaining <= 0:
                result += "[Timeout] execution took too long or produced no output"
                break
            try:
                msg = self.kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                result += "[Timeout] execution took too long or produced no output"
                break
            # A kernel may still have IOPub traffic buffered from an earlier
            # cell.  Consuming an unrelated ``idle`` here used to make the
            # current call return stale/empty output, which in turn encouraged
            # the model to submit the exact same tool call indefinitely.
            parent_id = msg.get("parent_header", {}).get("msg_id")
            if parent_id != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                result += content.get("text", "")
            elif msg_type == "execute_result":
                result += json.dumps(content.get("data", {}), ensure_ascii=False)
            elif msg_type in {"display_data", "update_display_data"}:
                data = content.get("data", {})
                text_output = data.get("text/plain")
                if text_output:
                    result += str(text_output)
                if "image/png" in data or "image/jpeg" in data:
                    result += (
                        "\n[Image display produced; pixels are not visible in this "
                        "text-only tool result. Inspect the source data directly, "
                        "use OCR/image analysis, or save the image as a deliverable.]"
                    )
            elif msg_type == "error":
                result += "\n".join(content.get("traceback", []))
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
        self.last_active = time.time()
        return result.strip()

    def shutdown(self):
        for fn in [
            lambda: self.kc.stop_channels(),
            lambda: self.km.shutdown_kernel(now=True),
            lambda: (self.km.has_kernel and self.km.kernel is not None
                     and (self.km.kernel.kill() or self.km.kernel.wait())),
        ]:
            try:
                fn()
            except Exception:
                pass
        try:
            self._ipc_dir.cleanup()
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ============================================================
# Inline tool implementations
# ============================================================

def _read_file(path: str) -> str:
    try:
        fp = Path(path).expanduser().resolve()
        if not fp.exists():
            return f"Error: File not found: {path}"
        if not fp.is_file():
            return f"Error: Not a file: {path}"
        try:
            content = fp.read_text(encoding="utf-8")
            if len(content) > 3000:
                content = content[:3000] + "\n\n[Read output capped at 3000 chars.]"
            return content
        except UnicodeDecodeError:
            return f"Error: File is not a text file: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(path: str, content: str) -> str:
    try:
        fp = Path(path).expanduser().resolve()
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _bash(command: str, working_dir: str = None, timeout: int = 120) -> str:
    try:
        process = subprocess.run(
            command, shell=True, capture_output=True,
            cwd=working_dir or os.getcwd(), timeout=timeout,
            env=os.environ.copy()
        )
        stdout = process.stdout.decode("utf-8", errors="replace").strip()
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        if process.returncode != 0:
            parts.append(f"[exit code: {process.returncode}]")
        result = "\n".join(parts) if parts else "(no output)"
        if len(result) > 10000:
            result = result[:5000] + "\n...(truncated)...\n" + result[-2000:]
        return result
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


# ============================================================
# Agent
# ============================================================

class LightLLMJupyterAgent:
    """
    LightLLM Jupyter Agent — no-skill variant.

    Parameters
    ----------
    api_key      : API key
    base_url     : LLM endpoint.
                   Ends with /generate      → lightllm raw endpoint
                   Ends with /completions   → vLLM text completions
                   GPT-6 Astra             → OpenAI Responses API
                   Otherwise                → OpenAI chat completions
    model_name   : model identifier
    data_root_path : root path for benchmark data files
    max_rounds   : max tool-call rounds per query
    """

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 data_root_path: str, max_rounds: int = 20,
                 enable_thinking: str = None,
                 reasoning_effort: str = None, **kwargs):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        self.baseUrl = base_url
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort

    # ----------------------------------------------------------
    # LLM dispatch
    # ----------------------------------------------------------
    def _get_response(self, messages: List[Dict], *, recovery: bool = False,
                      force_no_thinking: bool = False,
                      allow_tools: bool = True,
                      previous_response_id: Optional[str] = None,
                      response_input_start: int = 0) -> tuple:
        """Returns (message_obj_or_str, tokens, finish_reason).

        ``recovery`` requests are deliberately shorter.  They are used only
        after the model returned no usable final answer (or failed to create a
        required artifact).
        """
        stripped = self.baseUrl.rstrip("/")
        if stripped.endswith("/generate") or stripped.endswith("/completions"):
            return self._call_lightllm(messages)
        if self.model_name.lower().startswith("gpt-6-astra"):
            return self._call_openai_responses(
                messages,
                recovery=recovery,
                force_no_thinking=force_no_thinking,
                allow_tools=allow_tools,
                previous_response_id=previous_response_id,
                response_input_start=response_input_start,
            )
        return self._call_openai_chat(
            messages,
            recovery=recovery,
            force_no_thinking=force_no_thinking,
            allow_tools=allow_tools,
        )

    def _call_lightllm(self, messages: List[Dict]) -> tuple:
        input_text = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                input_text += f"<|im_start|>system\n{content}\n\n{MULTITOOL_TCALL_CONTENT}<|im_end|>\n"
            elif role == "user":
                input_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                input_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                input_text += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            else:
                raise ValueError(f"Unknown role: {role}")
        if self.enable_thinking == "think":
            input_text += "<|im_start|>assistant\n<think>\n"
        elif self.enable_thinking == "nothink":
            input_text += "<|im_start|>assistant\n<think>\n\n</think>\n"
        else:
            input_text += "<|im_start|>assistant\n"

        is_completions = self.baseUrl.rstrip("/").endswith("/completions")

        if is_completions:
            data = dict(
                model=self.model_name,
                prompt=input_text,
                max_tokens=8192,
                temperature=0,
            )
        else:
            data = dict(
                inputs=input_text,
                parameters={
                    "max_new_tokens": 8192,
                    "temperature": 0.001,
                    "top_p": 0.95,
                    "stop": ["<|im_end|>"],
                    "stop_sequences": ["<|im_end|>"],
                    "skip_special_tokens": False,
                },
            )

        try:
            raw = requests.post(self.baseUrl, data=json.dumps(data), headers={"Content-Type": "application/json"})
            resp = raw.json()
        except Exception as e:
            print(f"API Error: {e}")
            raise

        if "generated_text" in resp:
            return (
                resp["generated_text"][0].replace("<|im_end|>", ""),
                resp.get("count_output_tokens", 0),
                resp.get("finish_reason"),
            )
        elif "choices" in resp:
            text = resp["choices"][0]["text"].replace("<|im_end|>", "")
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            return text, tokens, resp["choices"][0].get("finish_reason")
        else:
            print(f"Unexpected API response (status {raw.status_code}): {resp}")
            raise RuntimeError(f"API returned no 'generated_text' or 'choices'. Response: {resp}")

    def _call_openai_chat(self, messages: List[Dict], *, recovery: bool = False,
                          force_no_thinking: bool = False,
                          allow_tools: bool = True) -> tuple:
        """Call OpenAI-compatible endpoint using native tool calling."""
        chat_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
            elif role == "assistant" and msg.get("tool_calls"):
                chat_messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": msg["tool_calls"],
                    "reasoning_content": msg.get("reasoning") or "",
                })
            else:
                chat_messages.append({"role": role, "content": msg.get("content", "")})
        try:
            is_gpt6_astra = self.model_name.lower().startswith("gpt-6-astra")
            token_limit_key = (
                "max_completion_tokens" if is_gpt6_astra else "max_tokens"
            )
            request_kwargs = dict(
                model=self.model_name,
                messages=chat_messages,
                stream=False,
            )
            request_kwargs[token_limit_key] = (
                1024 if recovery and not allow_tools else 8192
            )
            # TokenHub's GPT-6 Astra Chat Completions route only permits
            # function tools with reasoning disabled, and its Azure metadata
            # requires stored responses.
            if is_gpt6_astra:
                request_kwargs["reasoning_effort"] = "none"
                request_kwargs["store"] = True
            if allow_tools:
                request_kwargs["tools"] = TOOLS_SCHEMA
                request_kwargs["tool_choice"] = "auto"
            # TokenHub's current Claude endpoints reject temperature entirely.
            if not self.model_name.lower().startswith("claude-"):
                # TokenHub's GLM 5.3 channel accepts at most two decimal places.
                request_kwargs["temperature"] = (
                    0.01 if self.model_name.lower() == "glm-5.3-flash-b" else 0.001
                )
            is_deepseek_api = self.baseUrl.rstrip("/").lower() in {
                "https://api.deepseek.com",
                "https://api.deepseek.com/v1",
            }
            if is_deepseek_api and force_no_thinking:
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "disabled"},
                }
            elif is_deepseek_api and (
                self.enable_thinking in ("think", "nothink") or self.reasoning_effort
            ):
                request_kwargs["extra_body"] = {
                    "thinking": {
                        "type": "disabled" if self.enable_thinking == "nothink" else "enabled",
                    },
                }
                if self.reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
            elif force_no_thinking and not is_gpt6_astra:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"thinking": False},
                }
            elif not is_gpt6_astra and (
                self.enable_thinking in ("think", "nothink") or self.reasoning_effort
            ):
                chat_template_kwargs = {
                    "thinking": self.enable_thinking != "nothink",
                }
                if self.reasoning_effort:
                    chat_template_kwargs["reasoning_effort"] = self.reasoning_effort
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": chat_template_kwargs,
                }
            resp = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            print(f"API Error: {e}")
            raise
        tokens = resp.usage.completion_tokens if resp.usage else 0
        choice = resp.choices[0]
        return choice.message, tokens, getattr(choice, "finish_reason", None)

    @staticmethod
    def _response_field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _call_openai_responses(self, messages: List[Dict], *,
                               recovery: bool = False,
                               force_no_thinking: bool = False,
                               allow_tools: bool = True,
                               previous_response_id: Optional[str] = None,
                               response_input_start: int = 0) -> tuple:
        """Call GPT-6 Astra through Responses while preserving its tool chain."""
        response_input = []
        for msg in messages[response_input_start:]:
            role = msg["role"]
            if role == "tool":
                response_input.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": str(msg.get("content", "")),
                })
            elif role in {"system", "user"}:
                content = msg.get("content", "")
                if isinstance(content, list):
                    converted_content = []
                    for part in content:
                        if part.get("type") == "text":
                            converted_content.append({
                                "type": "input_text",
                                "text": part.get("text", ""),
                            })
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            if isinstance(image_url, dict):
                                image_url = image_url.get("url", "")
                            converted_content.append({
                                "type": "input_image",
                                "image_url": image_url,
                            })
                    content = converted_content
                response_input.append({
                    "role": role,
                    "content": content,
                })
            # Assistant output already belongs to previous_response_id and must
            # not be submitted a second time.

        request_kwargs = {
            "model": self.model_name,
            "input": response_input,
            "max_output_tokens": 1024 if recovery and not allow_tools else 8192,
            "reasoning": {
                "effort": "none" if force_no_thinking else (self.reasoning_effort or "high"),
            },
            "store": True,
        }
        if previous_response_id:
            request_kwargs["previous_response_id"] = previous_response_id
        if allow_tools:
            request_kwargs["tools"] = RESPONSES_TOOLS_SCHEMA
            request_kwargs["tool_choice"] = "auto"

        try:
            resp = self.client.responses.create(**request_kwargs)
        except Exception as e:
            print(f"API Error: {e}")
            raise

        tool_calls = []
        reasoning_parts = []
        for item in self._response_field(resp, "output", []) or []:
            item_type = self._response_field(item, "type", "")
            if item_type == "function_call":
                call_id = (
                    self._response_field(item, "call_id")
                    or self._response_field(item, "id", "")
                )
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name=self._response_field(item, "name", ""),
                        arguments=self._response_field(item, "arguments", "{}"),
                    ),
                ))
            elif item_type == "reasoning":
                for summary in self._response_field(item, "summary", []) or []:
                    text = self._response_field(summary, "text", "")
                    if text:
                        reasoning_parts.append(text)

        status = self._response_field(resp, "status", "completed")
        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "incomplete":
            details = self._response_field(resp, "incomplete_details")
            reason = self._response_field(details, "reason", "incomplete")
            finish_reason = "length" if reason == "max_output_tokens" else reason
        else:
            finish_reason = "stop"

        usage = self._response_field(resp, "usage")
        tokens = self._response_field(usage, "output_tokens", 0) or 0
        message = SimpleNamespace(
            content=self._response_field(resp, "output_text", "") or "",
            reasoning_content="\n".join(reasoning_parts),
            tool_calls=tool_calls,
            response_id=self._response_field(resp, "id"),
            response_input_start=len(messages),
        )
        return message, tokens, finish_reason

    @staticmethod
    def _missing_expected_artifacts(path_info: Dict[str, Any]) -> List[str]:
        """Return required artifact basenames that do not exist exactly."""
        expected = path_info.get("expected_output_files") or []
        output_dir = path_info.get("real_output_dir")
        if not expected or not output_dir:
            return []

        missing = []
        for name in expected:
            basename = os.path.basename(str(name).strip())
            if basename and not os.path.isfile(os.path.join(output_dir, basename)):
                missing.append(basename)
        return missing

    @staticmethod
    def _recovery_instruction(missing_artifacts: List[str]) -> str:
        if missing_artifacts:
            names = ", ".join(missing_artifacts)
            return (
                "The previous response did not complete the task. "
                f"Create the required output file(s) with these exact names: {names}. "
                "Use tools if needed, then provide a concise final answer in content. "
                "Do not repeat prior reasoning."
            )
        return (
            "The previous response contained no usable final answer. "
            "Do not call tools and do not repeat the reasoning. "
            "Return only a concise final answer in content now."
        )

    # ----------------------------------------------------------
    # Tool call parsing
    # ----------------------------------------------------------
    @staticmethod
    def _parse_tool_calls(text: str) -> list:
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        results = []
        for block in pattern.findall(text):
            func_match = re.search(r"<function=(\w+)>", block)
            if not func_match:
                continue
            func_name = func_match.group(1)
            param_matches = re.findall(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL)
            if not param_matches:
                unclosed = re.search(r"<parameter=(\w+)>\s*(.*?)$", block, re.DOTALL)
                if unclosed:
                    param_matches = [(unclosed.group(1), unclosed.group(2).strip())]
            func_args = {}
            for pname, pval in param_matches:
                pval = pval.strip()
                try:
                    func_args[pname] = json.loads(pval)
                except (json.JSONDecodeError, ValueError):
                    func_args[pname] = pval
            results.append((func_name, func_args))
        return results

    # ----------------------------------------------------------
    # Tool dispatch
    # ----------------------------------------------------------
    @staticmethod
    def _inspect_image(
        path: str,
        crop: Any,
        path_info: Dict[str, Any],
    ) -> Any:
        """Load an allowed task image and return an ephemeral data URL."""
        try:
            image_path = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            return f"Error: Image not found: {path} ({exc})"

        allowed_roots = []
        for key in ("real_input_dir", "real_work_dir", "real_output_dir"):
            raw_root = path_info.get(key)
            if not raw_root:
                continue
            try:
                root = Path(raw_root).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if root.is_dir():
                allowed_roots.append(root)

        if not allowed_roots or not any(
            image_path.is_relative_to(root) for root in allowed_roots
        ):
            return (
                "Error: inspect_image only permits files inside this task's "
                "inputs, scratch, or outputs directories."
            )
        if not image_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            with Image.open(image_path) as source:
                source.load()
                original_size = source.size
                image = source
                crop_text = "none"
                if crop is not None:
                    if (
                        not isinstance(crop, (list, tuple))
                        or len(crop) != 4
                        or any(isinstance(value, bool) or not isinstance(value, int) for value in crop)
                    ):
                        return "Error: crop must be four integer pixels: [left, top, right, bottom]."
                    left, top, right, bottom = crop
                    width, height = original_size
                    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                        return (
                            "Error: crop is outside image bounds "
                            f"{width}x{height}: {list(crop)}"
                        )
                    image = source.crop((left, top, right, bottom))
                    crop_text = str(list(crop))

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                payload = buffer.getvalue()
                if len(payload) > 20 * 1024 * 1024:
                    return (
                        "Error: inspected image exceeds 20 MiB after encoding; "
                        "provide a smaller crop."
                    )
                data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
                text = (
                    f"Image loaded from {path}; original size={original_size[0]}x{original_size[1]}, "
                    f"crop={crop_text}, returned size={image.size[0]}x{image.size[1]}. "
                    "The pixels are attached in the next user message."
                )
                return _ImageToolResult(text, data_url)
        except Exception as exc:
            return f"Error inspecting image: {exc}"

    @staticmethod
    def _trace_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return trace-safe messages without embedding image base64 payloads."""
        sanitized = []
        for message in messages:
            dumped = message.model_dump() if hasattr(message, "model_dump") else dict(message)
            content = dumped.get("content")
            if isinstance(content, list):
                safe_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        detail = image_url.get("detail") if isinstance(image_url, dict) else None
                        safe_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": "[image data omitted from trace]",
                                "detail": detail or "high",
                            },
                        })
                    else:
                        safe_parts.append(part)
                dumped["content"] = safe_parts
            sanitized.append(dumped)
        return sanitized

    def _execute_tool(self, func_name: str, func_args: dict,
                      executor: JupyterKernelExecutor,
                      path_info: Dict[str, str]) -> Any:

        def _remap(s):
            if isinstance(path_info, dict):
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    s = s.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
                if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                    s = s.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
                if 'mnt_work_dir' in path_info and 'real_work_dir' in path_info:
                    s = s.replace(path_info['mnt_work_dir'], path_info['real_work_dir'])
            return s

        if func_name == "execute_code":
            code = func_args.get("code", "")
            if not code:
                return "Error: No code provided."
            code = _remap(code)
            code = f"import matplotlib\nmatplotlib.use('Agg')\n{code}"
            try:
                res = executor.execute_code(code)
                if len(str(res)) > 2000:
                    res = str(res)[:1000] + "..." + str(res)[-1000:]
                return f"Executed Results:\n{res}"
            except Exception as e:
                return f"Execution Error: {e}"

        elif func_name == "read_file":
            path = func_args.get("path", "")
            if not path:
                return "Error: No path provided."
            return _read_file(_remap(path))

        elif func_name == "write_file":
            path = func_args.get("path", "")
            content = func_args.get("content", "")
            if not path:
                return "Error: No path provided."
            return _write_file(_remap(path), content)

        elif func_name == "bash":
            command = func_args.get("command", "")
            if not command:
                return "Error: No command provided."
            command = _remap(command)
            working_dir = func_args.get("working_dir", None)
            if isinstance(working_dir, str):
                working_dir = _remap(working_dir)
            timeout = func_args.get("timeout", 120)
            if isinstance(timeout, str):
                try:
                    timeout = int(timeout)
                except ValueError:
                    timeout = 120
            return _bash(command, working_dir=working_dir, timeout=timeout)

        elif func_name == "inspect_image":
            path = func_args.get("path", "")
            if not path:
                return "Error: No path provided."
            return self._inspect_image(
                _remap(path),
                func_args.get("crop"),
                path_info,
            )

        else:
            return f"Error: Unknown function '{func_name}'"

    # ----------------------------------------------------------
    # Main interaction loop
    # ----------------------------------------------------------
    def interact(self, query: str, system_prompt: str,
                 run_code_func: Any, path_info: Dict[str, str]) -> Dict[str, Any]:

        # Honor an explicit prompt supplied by the runner (for example via
        # --prompt_file), while retaining the built-in prompt as a fallback.
        system_prompt = system_prompt or SYSTEM_PROMPT
        stripped = self.baseUrl.rstrip("/")
        is_lightllm = stripped.endswith("/generate") or stripped.endswith("/completions")
        is_gpt6_astra = getattr(self, "model_name", "").lower().startswith("gpt-6-astra")

        def _split_thinking(text: str):
            """Split <think>...</think> from the rest of the message.
            Returns (reasoning, content) where reasoning may be empty.
            Also handles the case where the model continues after a <think> prefix
            injected in the prompt (so generated_text starts mid-think without the tag).
            """
            # Full <think>...</think> present
            m = re.match(r"<think>(.*?)</think>\s*", text, re.DOTALL)
            if m:
                return m.group(1).strip(), text[m.end():].strip()
            # Model output starts mid-think (prompt injected <think>\n already)
            m2 = re.search(r"</think>\s*", text, re.DOTALL)
            if m2:
                return text[:m2.start()].strip(), text[m2.end():].strip()
            return "", text

        input_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        round_count = 0
        all_tokens = 0
        final_response = ""
        fail_times = 0
        recovery_attempts = 0
        max_recovery_attempts = 2
        request_options = {}
        retryable_failure = False

        def execute_tool(func_name: str, func_args: dict) -> Any:
            if is_lightllm and func_name == "inspect_image":
                return "Error: inspect_image requires the native OpenAI tool-calling endpoint."
            try:
                return self._execute_tool(func_name, func_args, executor, path_info)
            except Exception as e:
                return f"Error executing {func_name}: {e}"

        executor = JupyterKernelExecutor()
        try:
            while True:
                if round_count >= self.max_rounds:
                    final_response = "Error: Too many rounds reached."
                    break
                if fail_times > 10:
                    final_response = "Error: Too many API failures."
                    retryable_failure = True
                    break
                round_count += 1

                try:
                    response_obj, completion_tokens, finish_reason = self._get_response(
                        input_message, **request_options
                    )
                    request_options = {}
                    all_tokens += completion_tokens
                except Exception as e:
                    fail_times += 1
                    print(f"API call failed (attempt {fail_times}): {e}")
                    continue

                # ---- text completions path: response_obj is a plain string ----
                if is_lightllm:
                    generated_message = response_obj  # str

                    if "<tool_call>" in generated_message:
                        reasoning, content = _split_thinking(generated_message)
                        input_message.append({
                            "role": "assistant",
                            "content": content,
                            "reasoning": reasoning,
                            "finish_reason": finish_reason,
                        })
                        tool_calls = self._parse_tool_calls(content)
                        for func_name, func_args in tool_calls:
                            result = execute_tool(func_name, func_args)
                            input_message.append({"role": "tool", "name": func_name, "content": result})
                        if not tool_calls:
                            missing_artifacts = self._missing_expected_artifacts(path_info)
                            if content.strip() and not missing_artifacts:
                                final_response = content
                                break
                            if recovery_attempts >= max_recovery_attempts or round_count >= self.max_rounds:
                                final_response = "Error: Model returned no usable final response after recovery attempts."
                                break
                            recovery_attempts += 1
                            input_message.append({
                                "role": "user",
                                "content": self._recovery_instruction(missing_artifacts),
                            })
                    else:
                        reasoning, content = _split_thinking(generated_message)
                        input_message.append({
                            "role": "assistant",
                            "content": content,
                            "reasoning": reasoning,
                            "finish_reason": finish_reason,
                        })
                        missing_artifacts = self._missing_expected_artifacts(path_info)
                        if content.strip() and not missing_artifacts:
                            final_response = content
                            break
                        if recovery_attempts >= max_recovery_attempts or round_count >= self.max_rounds:
                            final_response = "Error: Model returned no usable final response after recovery attempts."
                            break
                        recovery_attempts += 1
                        input_message.append({
                            "role": "user",
                            "content": self._recovery_instruction(missing_artifacts),
                        })

                # ---- OpenAI native tool calling path ----
                else:
                    msg = response_obj  # ChatCompletionMessage
                    text_content = msg.content or ""
                    reasoning_content = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or ""

                    if msg.tool_calls:
                        assistant_entry = {
                            "role": "assistant",
                            "content": text_content,
                            "reasoning": reasoning_content,
                            "finish_reason": finish_reason,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in msg.tool_calls
                            ],
                        }
                        input_message.append(assistant_entry)

                        image_results = []
                        for tc in msg.tool_calls:
                            func_name = tc.function.name
                            try:
                                func_args = json.loads(tc.function.arguments)
                            except (json.JSONDecodeError, ValueError):
                                func_args = {}
                            result = execute_tool(func_name, func_args)
                            if isinstance(result, _ImageToolResult):
                                tool_content = result.text
                                image_results.append(result)
                            else:
                                tool_content = result
                            input_message.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_content,
                            })
                        if image_results:
                            image_content = [{
                                "type": "text",
                                "text": "Image inspection result(s) requested by the assistant:",
                            }]
                            for image_result in image_results:
                                image_content.extend([
                                    {"type": "text", "text": image_result.text},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_result.data_url,
                                            "detail": "high",
                                        },
                                    },
                                ])
                            input_message.append({
                                "role": "user",
                                "content": image_content,
                            })
                        if is_gpt6_astra:
                            request_options = {
                                "previous_response_id": msg.response_id,
                                "response_input_start": msg.response_input_start,
                            }
                    else:
                        input_message.append({
                            "role": "assistant",
                            "content": text_content,
                            "reasoning": reasoning_content,
                            "finish_reason": finish_reason,
                        })
                        missing_artifacts = self._missing_expected_artifacts(path_info)
                        if text_content.strip() and not missing_artifacts:
                            final_response = text_content
                            break

                        if recovery_attempts >= max_recovery_attempts or round_count >= self.max_rounds:
                            final_response = "Error: Model returned no usable final response after recovery attempts."
                            break

                        recovery_attempts += 1
                        input_message.append({
                            "role": "user",
                            "content": self._recovery_instruction(missing_artifacts),
                        })
                        request_options = {
                            "recovery": True,
                            "force_no_thinking": finish_reason == "length",
                            "allow_tools": bool(missing_artifacts),
                        }
                        if is_gpt6_astra:
                            request_options.update({
                                "previous_response_id": msg.response_id,
                                "response_input_start": msg.response_input_start,
                            })

        finally:
            try:
                executor.shutdown()
            except Exception:
                pass

        result = {
            "model_response": final_response,
            "history": self._trace_history(input_message),
            "total_tokens": all_tokens,
            "rounds": round_count,
        }
        if retryable_failure:
            result["_retryable_failure"] = True
        return result
