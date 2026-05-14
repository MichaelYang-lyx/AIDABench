"""
Skill Jupyter Agent with configurable skills directory.

Supports:
  - lightllm /generate endpoint  (base_url ends with /generate)
  - OpenAI-compatible chat completions  (base_url ends with /v1 or similar)

Skills are loaded from --skills_dir (default: skills/).
Each subdirectory that contains a SKILL.md is auto-registered as a skill.

Compatible with AIDABench InferenceRunner interface.
"""

import os
import re
import sys
import json
import subprocess
import requests
import time
import queue
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skills import SkillsLoader

from openai import OpenAI

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
]


# ============================================================
# Skill discovery
# ============================================================


def _build_skills_block(skills_dir: str) -> str:
    """Build the <skills> XML block using SkillsLoader."""
    loader = SkillsLoader(skills_dir)
    skills = loader.list_skills(filter_unavailable=False)
    if not skills:
        return ""
    entries = []
    for s in skills:
        avail = str(s["available"]).lower()
        name = s["name"]
        desc = s["description"]
        loc = s["path"]
        entries.append(
            f"  <skill available='{avail}'>\n"
            f"    <name>{name}</name>\n"
            f"    <description>{desc}</description>\n"
            f"    <location>{loc}</location>\n"
            f"  </skill>"
        )
    return "## 🛠 Available Skills\n<skills>\n" + "\n".join(entries) + "\n</skills>"


def _build_system_prompt(skills_block: str) -> str:
    return f"""# Role
你是办公小浣熊，一个由商汤科技研发的专业、稳健的 AI 分析助手。
- 核心能力：写作与文本生成，数据分析与结构化推理，复杂任务拆解与执行规划
- 工作原则：理解用户的核心诉求，根据实际需要组合调用各种工具，在信息可验证、逻辑可追溯的前提下，高质量完成用户请求。

<path_mapping>
用户消息中的文件路径使用虚拟挂载路径，系统会自动重映射到真实路径：
- 输入文件：`/mnt/data/<filename>` → 实际数据目录
- 输出文件：`/mnt/result/<filename>` → 实际输出目录
请直接使用用户消息中提供的路径，系统会透明地处理路径映射。
</path_mapping>

<workflow>
### 技能优先原则
每次接收任务时，**首先**检查可用技能列表。如果存在匹配的技能，必须用 `read_file` 读取其 SKILL.md 文档，仔细阅读其工作流和最佳实践，然后严格按照技能指导执行任务。
## 实际工作流程
1. 解析与技能匹配 (Identify)：根据文件格式，选择是否激活skills且如果要使用skills，请用read_file工具来打印内容。
2. 分步规划 (Plan)：如果有技能加载，请根据技能的指导来一步步思考，如果没有请根据用户的提问进行深度思考，在脑海中形成一个高层级的分析计划。将用户的复杂问题拆解成一系列可以由工具执行的、更小的逻辑步骤。例如，处理表格时，应先检查所有sheet，找出相关的表再进行进一步分析。
3. 单步执行 (Execute)：分析结果后决定是否需要下一步。若报错，需分析原因并尝试修复（Self-Healing）。当一个工具持续出错时，应考虑其他方案而不是执着于修复。如果发现数据有质量问题（如缺失值、异常值），应主动使用工具进行探查或清洗。
4. 结果综合 (Synthesize)：汇总数据，以 Markdown 表格形式呈现。
</workflow>

{skills_block}
"""


# ============================================================
# Jupyter kernel executor
# ============================================================

class JupyterKernelExecutor:
    def __init__(self, timeout=60):
        self.km = KernelManager()
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
        self.kc.execute(code)
        result = ""
        start_time = time.time()
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
            except queue.Empty:
                result += "[Timeout] execution took too long or produced no output"
                break
            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                result += content.get("text", "")
            elif msg_type == "execute_result":
                result += json.dumps(content.get("data", {}), ensure_ascii=False)
            elif msg_type == "error":
                result += "\n".join(content.get("traceback", []))
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
            if time.time() - start_time > self.timeout:
                result += "\n[Stopped] code exceeded time limit"
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
            if "SKILL.md" not in path and len(content) > 3000:
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

class SkillJupyterAgent:
    """
    Skill Jupyter Agent with configurable skills directory.

    Parameters
    ----------
    api_key      : API key
    base_url     : LLM endpoint.
                   Ends with /generate  → lightllm raw endpoint
                   Otherwise           → OpenAI chat completions
    model_name   : model identifier
    data_root_path : root path for benchmark data files
    max_rounds   : max tool-call rounds per query
    skills_dir   : path to skills directory (default: <project_root>/skills)
    """

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 data_root_path: str, max_rounds: int = 20,
                 skills_dir: str = None, enable_thinking: str = None, **kwargs):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        self.baseUrl = base_url
        self.enable_thinking = enable_thinking

        if skills_dir is None:
            skills_dir = str(_PROJECT_ROOT / "skills")
        else:
            skills_dir = str(Path(skills_dir).resolve())
        self.skills_dir = skills_dir
        self._skills_block = _build_skills_block(skills_dir)
        self._system_prompt_base = _build_system_prompt(self._skills_block)

    # ----------------------------------------------------------
    # LLM dispatch
    # ----------------------------------------------------------
    def _get_response(self, messages: List[Dict]) -> tuple:
        """Returns (message_obj_or_str, tokens).
        lightllm path: returns (str, int)
        openai path:   returns (openai.types.chat.ChatCompletionMessage, int)
        """
        stripped = self.baseUrl.rstrip("/")
        if stripped.endswith("/generate") or stripped.endswith("/completions"):
            return self._call_lightllm(messages)
        return self._call_openai_chat(messages)

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
                temperature=0.0,
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
            return resp["generated_text"][0].replace("<|im_end|>", ""), resp.get("count_output_tokens", 0)
        elif "choices" in resp:
            text = resp["choices"][0]["text"].replace("<|im_end|>", "")
            tokens = resp.get("usage", {}).get("completion_tokens", 0)
            return text, tokens
        else:
            print(f"Unexpected API response (status {raw.status_code}): {resp}")
            raise RuntimeError(f"API returned no 'generated_text' or 'choices'. Response: {resp}")

    def _call_openai_chat(self, messages: List[Dict]) -> tuple:
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
                # Preserve tool_calls so the API can match tool results to calls.
                # DeepSeek requires reasoning_content to always be present when tool calls exist,
                # even if empty — omitting it causes a 400 error.
                entry = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": msg["tool_calls"],
                    "reasoning_content": msg.get("reasoning") or "",
                }
                chat_messages.append(entry)
            else:
                chat_messages.append({"role": role, "content": msg.get("content", "")})
        try:
            
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=8192,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as e:
            print(f"API Error: {e}")
            raise
        tokens = resp.usage.completion_tokens if resp.usage else 0
        return resp.choices[0].message, tokens

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
    def _execute_tool(self, func_name: str, func_args: dict,
                      executor: JupyterKernelExecutor,
                      path_info: Dict[str, str]) -> str:

        def _remap(s):
            if isinstance(path_info, dict):
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    s = s.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
                if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                    s = s.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
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
            timeout = func_args.get("timeout", 120)
            if isinstance(timeout, str):
                try:
                    timeout = int(timeout)
                except ValueError:
                    timeout = 120
            return _bash(command, working_dir=working_dir, timeout=timeout)

        else:
            return f"Error: Unknown function '{func_name}'"

    # ----------------------------------------------------------
    # Main interaction loop
    # ----------------------------------------------------------
    def interact(self, query: str, system_prompt: str,
                 run_code_func: Any, path_info: Dict[str, str]) -> Dict[str, Any]:

        system_prompt = self._system_prompt_base
        stripped = self.baseUrl.rstrip("/")
        is_lightllm = stripped.endswith("/generate") or stripped.endswith("/completions")

        def _split_thinking(text: str):
            """Split <think>...</think> from the rest of the message.
            Returns (reasoning, content) where reasoning may be empty.
            Also handles the case where the model continues after a <think> prefix
            injected in the prompt (so generated_text starts mid-think without the tag).
            """
            import re as _re
            m = _re.match(r"<think>(.*?)</think>\s*", text, _re.DOTALL)
            if m:
                return m.group(1).strip(), text[m.end():].strip()
            m2 = _re.search(r"</think>\s*", text, _re.DOTALL)
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

        executor = JupyterKernelExecutor()
        try:
            while True:
                round_count += 1
                if round_count > self.max_rounds:
                    final_response = "Error: Too many rounds reached."
                    break
                if fail_times > 10:
                    final_response = "Error: Too many API failures."
                    break

                try:
                    response_obj, completion_tokens = self._get_response(input_message)
                    all_tokens += completion_tokens
                except Exception as e:
                    fail_times += 1
                    print(f"API call failed (attempt {fail_times}): {e}")
                    continue

                # ---- lightllm path: response_obj is a plain string ----
                if is_lightllm:
                    generated_message = response_obj  # str

                    if "<tool_call>" in generated_message:
                        reasoning, content = _split_thinking(generated_message)
                        input_message.append({"role": "assistant", "content": content, "reasoning": reasoning})
                        tool_calls = self._parse_tool_calls(content)
                        for func_name, func_args in tool_calls:
                            try:
                                result = self._execute_tool(func_name, func_args, executor, path_info)
                            except Exception as e:
                                result = f"Error executing {func_name}: {e}"
                            input_message.append({"role": "tool", "name": func_name, "content": result})
                        if not tool_calls:
                            final_response = content
                            break
                    else:
                        reasoning, content = _split_thinking(generated_message)
                        final_response = content
                        input_message.append({"role": "assistant", "content": content, "reasoning": reasoning})
                        break

                # ---- OpenAI native tool calling path ----
                else:
                    msg = response_obj  # ChatCompletionMessage
                    text_content = msg.content or ""
                    reasoning_content = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or ""

                    if msg.tool_calls:
                        # Append assistant message with tool_calls
                        assistant_entry = {
                            "role": "assistant",
                            "content": text_content,
                            "reasoning": reasoning_content,
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

                        for tc in msg.tool_calls:
                            func_name = tc.function.name
                            try:
                                func_args = json.loads(tc.function.arguments)
                            except (json.JSONDecodeError, ValueError):
                                func_args = {}
                            try:
                                result = self._execute_tool(func_name, func_args, executor, path_info)
                            except Exception as e:
                                result = f"Error executing {func_name}: {e}"
                            input_message.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            })
                    else:
                        final_response = text_content
                        input_message.append({"role": "assistant", "content": text_content, "reasoning": reasoning_content})
                        break

        finally:
            try:
                executor.shutdown()
            except Exception:
                pass

        return {
            "model_response": final_response,
            "history": [
                msg.model_dump() if hasattr(msg, 'model_dump') else msg
                for msg in input_message
            ],
            "total_tokens": all_tokens,
            "rounds": round_count,
        }
