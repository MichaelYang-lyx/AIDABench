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

# Skill metadata: name, description, file type triggers
_SKILL_META = {
    "minimax-xlsx": {
        "name": "minimax-xlsx",
        "type": "xlsx",
        "description": (
            "Use for ALL Excel/spreadsheet tasks: create new .xlsx from scratch, "
            "read and analyze existing files, edit existing .xlsx with zero format loss, "
            "fix broken formulas, validate formulas. Triggers on: .xlsx, .xls, .csv, "
            "'spreadsheet', 'Excel', 'pivot table', 'financial model', 'formula', "
            "or any request to produce tabular data in Excel format."
        ),
    },
    "pptx-generator": {
        "name": "pptx-generator",
        "type": "pptx",
        "description": (
            "Use for ALL PowerPoint tasks: generate presentations from scratch with "
            "PptxGenJS, edit existing PPTX via XML workflows, or extract text with "
            "markitdown. Triggers on: PPT, PPTX, PowerPoint, presentation, slide, deck."
        ),
    },
    "minimax-pdf": {
        "name": "minimax-pdf",
        "type": "pdf",
        "description": (
            "Use when visual quality matters for a PDF. Three routes: "
            "CREATE (generate from scratch), FILL (complete form fields in existing PDF), "
            "REFORMAT (apply design to existing doc). Triggers on: 'make a PDF', "
            "'generate a report', 'fill in the form', 'reformat this document', "
            "'beautiful PDF', 'professional document', 'cover page'."
        ),
    },
    "minimax-docx": {
        "name": "minimax-docx",
        "type": "docx",
        "description": (
            "Use for ALL Word document tasks: create new .docx from scratch, "
            "fill/edit content in existing documents, apply template formatting. "
            "Triggers on: Word, docx, document, 'write a report', 'draft a proposal', "
            "'make a contract', 'fill in this form', 'reformat to match this template', "
            "or any task whose final output is a .docx file."
        ),
    },
}


def _build_skills_block(skills_dir: str) -> str:
    """Scan skills_dir and build the <skills> XML block for the system prompt."""
    skills_dir = Path(skills_dir)
    entries = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        skill_name = skill_dir.name
        meta = _SKILL_META.get(skill_name, {})
        stype = meta.get("type", "general")
        description = meta.get("description", f"Skill for {skill_name}.")
        entries.append(
            f'  <skill available="true" type="{stype}">\n'
            f'    <name>{skill_name}</name>\n'
            f'    <description>{description}</description>\n'
            f'    <location>{skill_md}</location>\n'
            f'  </skill>'
        )
    if not entries:
        return ""
    return "## 🛠️ Available Skills\n<skills>\n" + "\n".join(entries) + "\n</skills>"


def _build_system_prompt(skills_block: str) -> str:
    return f"""You are a professional AI assistant with advanced data analysis, file management, and code execution capabilities.

<path_mapping>
File paths in user messages use virtual mount paths that are automatically remapped to real paths:
- Input files: `/mnt/data/<filename>` → actual data directory
- Output files: `/mnt/result/<filename>` → actual output directory
Always use the paths exactly as provided in the user message; the system handles the remapping transparently.
</path_mapping>

<workflow>
### Skill-First Principle
At the start of every task, scan the available skills list below. If a matching skill exists, you MUST use `read_file` to load its SKILL.md, read the workflow and best practices carefully, then follow the skill's guidance strictly.

## Execution workflow
1. Identify & match (Identify): Based on file type or task intent, decide whether to activate a skill. If yes, use read_file to load the SKILL.md.
2. Plan (Plan): If a skill is loaded, think step-by-step following the skill's workflow. Otherwise, deeply analyze the user's request and form a high-level plan. Break complex problems into smaller executable steps.
3. Execute step-by-step (Execute): After each step, evaluate the result before proceeding. On errors, analyze and self-heal. If a tool keeps failing, switch to an alternative approach. Proactively handle data quality issues (missing values, type errors).
4. Synthesize (Synthesize): Aggregate results and present a clear, structured final answer.
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
                 skills_dir: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        self.baseUrl = base_url

        if skills_dir is None:
            skills_dir = str(_PROJECT_ROOT / "skills")
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
        if self.baseUrl.rstrip("/").endswith("/generate"):
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
        input_text += "<|im_start|>assistant\n"
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
            raw = requests.post(self.baseUrl, data=json.dumps(data))
            resp = raw.json()
        except Exception as e:
            print(f"API Error: {e}")
            raise
        return resp["generated_text"][0].replace("<|im_end|>", ""), resp["count_output_tokens"]

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
                # Preserve tool_calls so the API can match tool results to calls
                chat_messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": msg["tool_calls"],
                })
            else:
                chat_messages.append({"role": role, "content": msg.get("content", "")})
        try:
            
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                max_tokens=8192,
                temperature=0.001,
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
        is_lightllm = self.baseUrl.rstrip("/").endswith("/generate")

        input_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        round_count = 0
        all_tokens = 0
        final_response = ""
        fail_times = 0
        _recent_outputs = []
        _LOOP_THRESHOLD = 3

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

                    _recent_outputs.append(generated_message)
                    if len(_recent_outputs) >= _LOOP_THRESHOLD:
                        last_n = _recent_outputs[-_LOOP_THRESHOLD:]
                        if all(x == last_n[0] for x in last_n):
                            print(f"[Loop detected] Model repeated same output {_LOOP_THRESHOLD} times. Breaking.")
                            text_only = re.sub(r'<tool_call>.*?</tool_call>', '', generated_message, flags=re.DOTALL).strip()
                            final_response = text_only or generated_message
                            break

                    if "<tool_call>" in generated_message:
                        input_message.append({"role": "assistant", "content": generated_message})
                        tool_calls = self._parse_tool_calls(generated_message)
                        for func_name, func_args in tool_calls:
                            try:
                                result = self._execute_tool(func_name, func_args, executor, path_info)
                            except Exception as e:
                                result = f"Error executing {func_name}: {e}"
                            input_message.append({"role": "tool", "name": func_name, "content": result})
                        if not tool_calls:
                            final_response = generated_message
                            break
                    else:
                        final_response = generated_message
                        input_message.append({"role": "assistant", "content": generated_message})
                        break

                # ---- OpenAI native tool calling path ----
                else:
                    msg = response_obj  # ChatCompletionMessage
                    text_content = msg.content or ""

                    _recent_outputs.append(text_content)
                    if text_content and len(_recent_outputs) >= _LOOP_THRESHOLD:
                        last_n = _recent_outputs[-_LOOP_THRESHOLD:]
                        if all(x == last_n[0] for x in last_n):
                            print(f"[Loop detected] Model repeated same output {_LOOP_THRESHOLD} times. Breaking.")
                            final_response = text_content or "Error: loop detected with no text."
                            break

                    if msg.tool_calls:
                        # Append assistant message with tool_calls
                        assistant_entry = {
                            "role": "assistant",
                            "content": text_content,
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
                        input_message.append({"role": "assistant", "content": text_content})
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
