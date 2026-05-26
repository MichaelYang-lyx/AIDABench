"""
OpenAI Jupyter Agent with Excel_WorkFlow skill for QA benchmark.

Uses the pipeline 2.0 Excel_WorkFlow skill (workflow + 44 capability sub-skills)
in the system prompt. The workflow skill provides the orchestration logic,
and capability sub-skills provide detailed code patterns for specific operations.

Compatible with AIDABench InferenceRunner interface (same as openai_jupyter_agent_skills_qa.py).
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
from typing import Dict, Any, List, Optional, Union

from openai import OpenAI

# Jupyter kernel support
try:
    from jupyter_client import KernelManager
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False


# ============================================================
# Skill paths (relative to project root)
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILL_BASE = str(_PROJECT_ROOT / "skills" / "Excel_WorkFlow")
WORKFLOW_SKILL_PATH = os.path.join(SKILL_BASE, "SKILL.md")
CAPABILITY_DIR = os.path.join(SKILL_BASE, "capability")


# ============================================================
# System prompt — only references the workflow skill
# The model reads it via read_file, then follows internal
# references to load capability sub-skills as needed.
# ============================================================
MULTITOOL_SYSTEM_PROMPT = r"""你是一个具备高级数据分析、文件管理与代码执行能力的智能AI助手。

## ⚙️ 核心能力集
- **Jupyter 交互环境**：支持持久化 Python 代码执行（变量与上下文在跨单元格调用时保持状态）。
- **文件系统**：具备完整的文件读、写、解析及目录管理能力。
- **Shell 终端**：支持标准 Bash 命令操作。
- **技能库 (Skills)**：内置专业任务解决方案。在规划任何任务前，必须优先评估是否可通过现有技能完成。

## 🔄 标准执行工作流 (SOP)
在处理任何用户请求时，必须严格遵循以下执行流：

### 步骤 0：技能嗅探与匹配（强制前置动作）
开始任务前，先扫描下方的 **<skills>** 列表。如果发现与当前任务高度匹配的技能（例如：处理 Excel 时匹配到 excel-data-analysis-workflow），必须立即使用对应工具读取其 `<location>` 指向的 `SKILL.md`。

---

### 分支 A：匹配到技能时（技能驱动模式）
一旦加载了技能文档，你的核心逻辑将由该技能接管：
1. **严格遵循工作流**：按照技能文档定义的工作流（Workflow）逐步推进，每一步对应一次精准的工具调用。
2. **按需懒加载**：仅在执行到特定步骤时，才读取该步骤所需的子技能文档，**严禁**一次性批量加载所有子技能。
3. **动态参数适配**：将技能提供的代码 Snippet 视作模板。你必须根据当前实际数据（如真实的列名、Sheet 表名、数据阈值）动态调整代码，绝不能盲目复制。
4. **综合收尾 (Synthesize)**：所有技能步骤执行完毕后，整合关键数据，输出结构化、清晰的最终结论。

### 分支 B：无匹配技能时（自主规划模式）
1. **意图解析 (Understand)**：深度分析请求，锁定最终目标。如果问题无需代码或工具即可回答，则直接输出回复。
2. **任务拆解 (Plan)**：将复杂问题降维，拆解为逻辑连贯的可执行步骤。
3. **单步执行与自愈 (Execute & Self-Healing)**：
   - 每次仅执行一个具体步骤。
   - 评估当前执行结果后再决定下一步。
   - 遇到报错时，必须主动分析日志并尝试修复（Self-Healing）；如果某种工具/方法陷入死循环，立即切换替代方案。
4. **综合收尾 (Synthesize)**：提炼各步骤的执行结果，向用户交付最终结论。

## 💻 代码执行纪律
- **意图声明**：每次生成/执行代码前，用一句话简明扼要地说明本次代码的预期目标。
- **信噪比控制**：禁止打印海量无用数据。仅输出对“决定下一步行动”或“生成最终结论”有决定性价值的信息。
- **快速失败与防御性编程**：在同一步骤的代码中应包含基本的数据质量检测（如判空、类型检查）并予以处理，避免在多步交互中反复试错。
- **状态复用**：最大化利用 Jupyter 的持久化状态。变量只需定义和读取一次即可跨单元复用，严禁重复读取大文件或进行冗余的重复计算。

## 🛠️ Available Skills
<skills>
  <skill available="true" type="workflow">
    <name>excel-data-analysis-workflow</name>
    <description>用于 Excel 数据分析的多步工作流：支持多 Sheet 读取、大文件检测（>=10k 行转换为 Parquet）、数据清洗、过滤、聚合以及结果导出。内部包含 capability 子技能索引表，需根据进度通过 read_file 按需加载。</description>
    <location>""" + WORKFLOW_SKILL_PATH + r"""</location>
  </skill>
  <skill available="true" type="Doc_Parse">
    <name>doc-parse-and-edit</name>
    <description>核心输入为 .doc 或 .docx 文件时的专属技能。支持：(1) 通过 scripts/parse_doc.py 提取文本、表格、图片与格式；(2) 通过 scripts/ocr_image.py 调用 PaddleOCR-VL API 对内嵌图片（图表/公式）进行 OCR 识别；(3) 利用 python-docx 修改内容（文本替换、日期修改、格式调整）；(4) .doc 与 .docx 格式互转。仅在用户上传 Word 文档且需要提取、重写或理解时触发。禁止在 Excel、PPT、PDF 或纯图片任务中触发。</description>
    <location>""" + "/mnt/afs_agents/hongjiawei/tools/mySkill/Doc_Parse/SKILL.md" + r"""</location>
  </skill>
</skills>
"""

MULTITOOL_SYSTEM_PROMPT_ALIGN = r"""## 🛠️ Available Skills
<skills>
  <skill available="true" type="workflow">
    <name>excel-data-analysis-workflow</name>
    <description>Multi-step workflow for Excel data analysis. Use when a task involves: (1) reading multi-sheet Excel files and counting rows, (2) large file detection (>=10k rows -> Parquet optimization), (3) data cleaning (missing values, text normalization, invalid characters), (4) conditional filtering and category extraction, (5) statistical aggregation across sheets, (6) exporting results as Excel/CSV with download links. Covers the full pipeline from ingestion to report generation. Orchestrates capability sub-skills for each step.</description>
    <location>""" + WORKFLOW_SKILL_PATH + r"""</location>
  </skill>
  <skill available="true" type="Doc_Parse">
    <name>doc-parse-and-edit</name>
    <description>Use this skill when .doc or .docx files are the primary input and the user needs to read, parse, extract content from, edit, or transform Word documents. Provides scripts for: (1) parsing .doc/.docx to extract text, tables, images, and formatting via scripts/parse_doc.py, (2) OCR/recognition of embedded images (charts, tables, formulas) via PaddleOCR-VL API through scripts/ocr_image.py, (3) editing document content (text replacement, date changes, formatting adjustments) via python-docx, (4) converting between .doc and .docx formats. Trigger when user uploads .doc/.docx files and wants: content extraction, text modification, date replacement, format-preserving edits, document rewriting, table extraction, or image understanding from documents. Do NOT trigger for: Excel files (.xlsx/.xls), PowerPoint files (.pptx/.ppt), PDF files, or image-only tasks without a Word document.</description>
    <location>""" + "/mnt/afs_agents/hongjiawei/tools/mySkill/Doc_Parse/SKILL.md" + r"""</location>
  </skill>
</skills>
"""
WORKFLOW_SKILL_PATH = str(_PROJECT_ROOT / "skills" / "Excel_WorkFlow" / "SKILL.md")
_IMAGE_CAPTION_SKILL_PATH = str(_PROJECT_ROOT / "skills" / "Image_Caption" / "SKILL.md")
MULTITOOL_SYSTEM_PROMPT_ALIGN = r"""## 🛠️ Available Skills
<skills>
  <skill available="true" type="workflow">
    <name>excel-data-analysis-workflow</name>
    <description>Multi-step workflow for Excel data analysis. Use when a task involves: (1) reading multi-sheet Excel files and counting rows, (2) large file detection (>=10k rows -> Parquet optimization), (3) data cleaning (missing values, text normalization, invalid characters), (4) conditional filtering and category extraction, (5) statistical aggregation across sheets, (6) exporting results as Excel/CSV with download links. Covers the full pipeline from ingestion to report generation. Orchestrates capability sub-skills for each step.</description>
    <location>""" + WORKFLOW_SKILL_PATH + r"""</location>
  </skill>
  <skill available="true" type="image_caption">
    <name>image-caption-analysis</name>
    <description>Use this skill when image files (.png, .jpg, .jpeg, .gif, .webp, .bmp) are the primary input and the user needs to understand, extract data from, or analyze image content. Provides a pre-configured caption script (scripts/caption.py) that converts images to text descriptions via a vision model — no API key setup needed. Covers: (1) captioning charts/tables/screenshots/diagrams via scripts/caption.py, (2) parsing caption text into structured DataFrames, (3) re-creating visualizations from extracted data, (4) exporting to Excel/CSV. Trigger when user uploads images and wants: data extraction, table OCR, chart analysis, UI description, or diagram understanding. Do NOT trigger for image editing (resize, crop, filter) or image generation.</description>
    <location>""" + _IMAGE_CAPTION_SKILL_PATH + r"""</location>
  </skill>
</skills>
"""

MULTITOOL_TCALL_CONTENT = r"""# Tools

You have access to the following functions:

<tools>
<function>
<name>execute_code</name>
<description>在同一个持续的Jupyter环境中执行一段Python代码。此环境是有状态的，意味着一次调用中定义的变量、函数或导入的库可以被后续的调用使用。</description>
<parameters>
<parameter>
<name>code</name>
<type>string</type>
<description>要执行的Python代码片段。</description>
</parameter>
<required>["code"]</required>
</parameters>
</function>
<function>
<name>read_file</name>
<description>Read the contents of a file. Supports text files only. Output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete. Cannot read binary files (.xlsx, .xls, .docx, .pdf, .png, .jpg, etc.) — use the jupyter tool with pandas/openpyxl instead.</description>
<parameters>
<parameter>
<name>path</name>
<type>string</type>
<description>The file path to read</description>
</parameter>
<required>["path"]</required>
</parameters>
</function>
<function>
<name>write_file</name>
<description>Write content to a file at the given path. Creates parent directories if needed.</description>
<parameters>
<parameter>
<name>path</name>
<type>string</type>
<description>The file path to write to</description>
</parameter>
<parameter>
<name>content</name>
<type>string</type>
<description>The content to write</description>
</parameter>
<required>["path", "content"]</required>
</parameters>
</function>
<function>
<name>bash</name>
<description>Execute a shell command and return its output (truncated to ~8000 chars). For large outputs, pipe through head/tail/grep. Use with caution.</description>
<parameters>
<parameter>
<name>command</name>
<type>string</type>
<description>The shell command to execute</description>
</parameter>
<parameter>
<name>working_dir</name>
<type>string</type>
<description>Optional working directory for the command</description>
</parameter>
<parameter>
<name>timeout</name>
<type>string</type>
<description>Timeout in seconds (default: 120)</description>
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
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


# ============================================================
# Inline tool implementations
# ============================================================

class JupyterKernelExecutor:
    """Persistent Jupyter kernel for stateful code execution."""

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
                result += "[Timeout] 执行时间过长或无输出"
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
                result += "\n[Stopped] 代码运行超出时间限制"
                break

        self.last_active = time.time()
        return result.strip()

    def shutdown(self):
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass
        try:
            if self.km.has_kernel and self.km.kernel is not None:
                self.km.kernel.kill()
                self.km.kernel.wait()
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


def _read_file(path: str) -> str:
    """Read file contents, capped at 3000 chars (SKILL.md uncapped)."""
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"
        try:
            content = file_path.read_text(encoding="utf-8")
            if "SKILL.md" not in path:
                if len(content) > 3000:
                    content = content[:3000] + f"\n\n[Read output capped at 3000 chars for this call.]"
            return content
        except UnicodeDecodeError:
            return f"Error: File is not a text file: {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent dirs if needed."""
    try:
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def _bash(command: str, working_dir: str = None, timeout: int = 120) -> str:
    """Execute a shell command and return output."""
    try:
        cwd = working_dir or os.getcwd()
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            cwd=cwd,
            timeout=timeout,
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
        return f"Error executing command: {str(e)}"


# ============================================================
# Agent class
# ============================================================

class OpenAIJupyterAgentExcelWorkflowQA:
    """
    Multi-tool agent using lightllm /generate endpoint.
    Uses Excel_WorkFlow skills (workflow + capability) in system prompt.

    Tools: execute_code, read_file, write_file, bash
    Compatible with AIDABench InferenceRunner interface.
    """

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 data_root_path: str, max_rounds: int = 20):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        self.baseUrl = base_url

    # ----------------------------------------------------------
    # LLM call — lightllm /generate or OpenAI chat completions
    # ----------------------------------------------------------
    def _get_response_lightllm(self, messages: List[Dict]) -> tuple:
        # If base_url ends with /generate, use lightllm raw endpoint
        if self.baseUrl.rstrip("/").endswith("/generate"):
            return self._call_lightllm(messages)
        # Otherwise use OpenAI chat completions (supports /v1 endpoints)
        return self._call_openai_chat(messages)

    def _call_lightllm(self, messages: List[Dict]) -> tuple:
        input_text = ""
        for message in messages:
            content = message["content"]
            role = message["role"]
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
            raw_response = requests.post(self.baseUrl, data=json.dumps(data))
            response = raw_response.json()
        except Exception as e:
            print(f"API Error: {e}")
            raise e
        response_message = response["generated_text"][0].replace("<|im_end|>", "")
        return response_message, response["count_output_tokens"]

    def _call_openai_chat(self, messages: List[Dict]) -> tuple:
        """Use OpenAI chat completions API with the custom XML tool-call format."""
        # Inject tool definitions into system message
        chat_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chat_messages.append({"role": "system", "content": content + "\n\n" + MULTITOOL_TCALL_CONTENT})
            elif role == "tool":
                chat_messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
            else:
                chat_messages.append({"role": role, "content": content})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                max_tokens=8192,
                temperature=0.001,
            )
        except Exception as e:
            print(f"API Error: {e}")
            raise e
        response_message = response.choices[0].message.content or ""
        total_tokens = response.usage.completion_tokens if response.usage else 0
        return response_message, total_tokens
    
    # ----------------------------------------------------------
    # Tool call parsing
    # ----------------------------------------------------------
    @staticmethod
    def _parse_tool_calls(text: str) -> list:
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        blocks = pattern.findall(text)
        results = []
        for block in blocks:
            func_match = re.search(r"<function=(\w+)>", block)
            if not func_match:
                continue
            func_name = func_match.group(1)
            param_matches = re.findall(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL
            )
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

        if func_name == "execute_code":
            code = func_args.get("code", "")
            if not code:
                return "Error: No code provided."
            if isinstance(path_info, dict):
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    code = code.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
                if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                    code = code.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
            code = f"import matplotlib\nmatplotlib.use('Agg')\n{code}"
            try:
                res = executor.execute_code(code)
                if len(str(res)) > 2000:
                    res = str(res)[:1000] + '...' + str(res)[-1000:]
                return f"Executed Results:\n{res}"
            except Exception as e:
                return f"Execution Error: {e}"

        elif func_name == "read_file":
            path = func_args.get("path", "")
            if not path:
                return "Error: No path provided."
            if isinstance(path_info, dict):
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    path = path.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
            return _read_file(path)

        elif func_name == "write_file":
            path = func_args.get("path", "")
            content = func_args.get("content", "")
            if not path:
                return "Error: No path provided."
            if isinstance(path_info, dict):
                if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                    path = path.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    path = path.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
            return _write_file(path, content)

        elif func_name == "bash":
            command = func_args.get("command", "")
            if not command:
                return "Error: No command provided."
            if isinstance(path_info, dict):
                if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                    command = command.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
                if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                    command = command.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
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
        """
        Multi-tool interaction loop.
        system_prompt is overridden with MULTITOOL_SYSTEM_PROMPT.
        """
        # system_prompt = MULTITOOL_SYSTEM_PROMPT
        # system_prompt = system_prompt + "\n" + MULTITOOL_SYSTEM_PROMPT_ALIGN
        system_prompt=f"""# Role
你是办公小浣熊，一个由商汤科技研发的专业、稳健的 AI 分析助手。\n- 核心能力：写作与文本生成，数据分析与结构化推理，复杂任务拆解与执行规划\n- 工作原则：理解用户的核心诉求，根据实际需要组合调用各种工具，在信息可验证、逻辑可追溯的前提下，高质量完成用户请求。

<workflow>
### 技能优先原则
每次接收任务时，**首先**检查可用技能列表。如果存在匹配的技能，必须用 `read_file` 读取其 SKILL.md 文档，仔细阅读其工作流和最佳实践，然后严格按照技能指导执行任务。
## 实际工作流程
1. 解析与技能匹配 (Identify)：根据文件格式，选择是否激活skills且如果要使用skills，请用read_file工具来打印内容。
2. 分步规划 (Plan)：如果有技能加载，请根据技能的指导来一步步思考，如果没有请根据用户的提问进行深度思考，在脑海中形成一个高层级的分析计划。将用户的复杂问题拆解成一系列可以由工具执行的、更小的逻辑步骤。例如，处理表格时，应先检查所有sheet，找出相关的表再进行进一步分析。
3. 单步执行 (Execute)：分析结果后决定是否需要下一步。若报错，需分析原因并尝试修复（Self-Healing）。当一个工具持续出错时，应考虑其他方案而不是执着于修复。如果发现数据有质量问题（如缺失值、异常值），应主动使用工具进行探查或清洗。
4. 结果综合 (Synthesize)：汇总数据，以 Markdown 表格形式呈现。
</workflow>

{MULTITOOL_SYSTEM_PROMPT_ALIGN}
"""

        input_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        round_count = 0
        all_tokens = 0
        final_response = ""

        executor = JupyterKernelExecutor()

        _recent_outputs = []
        _LOOP_THRESHOLD = 3
        fail_times = 0

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
                generated_message, completion_tokens = self._get_response_lightllm(input_message)
                all_tokens += completion_tokens
            except Exception as e:
                fail_times += 1
                print(f"API call failed (attempt {fail_times}): {e}")
                continue

            _recent_outputs.append(generated_message)
            if len(_recent_outputs) >= _LOOP_THRESHOLD:
                last_n = _recent_outputs[-_LOOP_THRESHOLD:]
                if all(x == last_n[0] for x in last_n):
                    print(f"[Loop detected] Model repeated same output {_LOOP_THRESHOLD} times. Breaking.")
                    text_only = re.sub(r'<tool_call>.*?</tool_call>', '', generated_message, flags=re.DOTALL).strip()
                    if text_only:
                        final_response = text_only
                    else:
                        for msg in reversed(input_message):
                            if isinstance(msg, dict) and msg.get('role') == 'tool':
                                final_response = msg.get('content', '')
                                break
                        if not final_response:
                            final_response = generated_message
                    break

            if "<tool_call>" in generated_message:
                input_message.append({"role": "assistant", "content": generated_message})

                tool_calls = self._parse_tool_calls(generated_message)

                for func_name, func_args in tool_calls:
                    print(f"\n*** Tool call: {func_name} ***")
                    try:
                        result = self._execute_tool(
                            func_name, func_args, executor, path_info
                        )
                    except Exception as e:
                        result = f"Error executing {func_name}: {e}"

                    print(f"  Result: {result[:200]}...")
                    input_message.append({
                        "role": "tool",
                        "name": func_name,
                        "content": result,
                    })

                if not tool_calls:
                    final_response = generated_message
                    break
            else:
                final_response = generated_message
                input_message.append({"role": "assistant", "content": generated_message})
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
