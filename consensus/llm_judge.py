"""LLM Judge evaluation module.

Evaluates a model's analysis output against a three-layer rubric.
Supports multiple runs for stability assessment.
Supports Hermes-based evaluation (code execution for verification).
"""

import json
import os
import re
import shutil

import statistics
from pathlib import Path
from typing import Dict, Any, List
from openai import OpenAI

try:
    import json_repair  # type: ignore
except ImportError:
    json_repair = None


_NOISE_DIR_NAMES = {
    '__pycache__', '.git', '.mypy_cache', '.pytest_cache',
    '.venv', 'venv', 'env', '.env', 'node_modules', '.ipynb_checkpoints',
}


def _is_python_venv(path: str) -> bool:
    """A directory is a Python virtualenv if it contains pyvenv.cfg at its root."""
    return os.path.isfile(os.path.join(path, 'pyvenv.cfg'))


def _copy_ignore(src: str, names: list) -> list:
    """ignore callable for copytree: skip noise dirs and nested virtualenvs."""
    ignored = []
    for n in names:
        full = os.path.join(src, n)
        if n in _NOISE_DIR_NAMES:
            ignored.append(n)
        elif os.path.isdir(full) and _is_python_venv(full):
            ignored.append(n)
    return ignored


def _copy_into(src_path: str, dst_path: Path) -> None:
    """Copy a file or directory tree into dst_path.

    We copy instead of symlink because the judge runs inside a docker container
    where only the judge workspace is bind-mounted: symlinks pointing to absolute
    host paths outside the workspace would be broken inside the container.

    Skips Python virtualenvs (detected via pyvenv.cfg) and well-known noise
    directories, and tolerates broken symlinks instead of crashing the copy.
    """
    if os.path.isdir(src_path):
        # Skip entire venv if the root path itself is one (e.g. model's ocr_env/).
        if _is_python_venv(src_path):
            return
        shutil.copytree(
            src_path, dst_path,
            dirs_exist_ok=True,
            symlinks=False,
            ignore=_copy_ignore,
            ignore_dangling_symlinks=True,
        )
    elif os.path.isfile(src_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


class LLMJudge:
    """Evaluates analysis outputs using LLM-as-judge with three-layer rubric."""

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 use_hermes: bool = False, max_rounds: int = 30, provider: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.use_hermes = use_hermes
        self.max_rounds = max_rounds
        self.provider = provider
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluate(
        self,
        analysis_output: str,
        rubric: Dict[str, Any],
        task_description: str,
        num_runs: int = 1,
        data_files: List[str] = None,
        model_output_workspace: str = None,
        judge_workspace_dir: str = None,
    ) -> Dict[str, Any]:
        """Evaluate an analysis output against a rubric.

        Args:
            analysis_output: The model's analysis to evaluate
            rubric: Three-layer rubric from RubricGenerator
            task_description: The original task/question
            num_runs: Number of independent evaluation runs (default: 1)
            data_files: List of original data file paths (for Hermes mode)
            model_output_workspace: Path to the tested model's workspace (for Hermes mode)
            judge_workspace_dir: Directory to store judge workspace (preserved after eval)

        Returns:
            Dict with scores, mean_score, std_score, variance, layer_scores, detailed_feedback
        """
        scores = []
        layer_scores_list = []
        feedbacks = []
        judge_traces = []

        for run_idx in range(num_runs):
            if self.use_hermes:
                result = self._hermes_evaluation(
                    analysis_output, rubric, task_description, run_idx,
                    data_files=data_files,
                    model_output_workspace=model_output_workspace,
                    judge_workspace_dir=judge_workspace_dir,
                )
            else:
                result = self._single_evaluation(
                    analysis_output, rubric, task_description, run_idx
                )
            trace = result.pop("_hermes_trace", None)
            if trace:
                trace["run_idx"] = run_idx
                judge_traces.append(trace)
            scores.append(result["total_score"])
            layer_scores_list.append(result["layer_scores"])
            feedbacks.append(result["feedback"])

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0

        avg_layer_scores = {}
        if layer_scores_list:
            for layer in layer_scores_list[0].keys():
                avg_layer_scores[layer] = statistics.mean(
                    [ls[layer] for ls in layer_scores_list]
                )

        return {
            "scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "variance": variance,
            "layer_scores": avg_layer_scores,
            "detailed_feedback": feedbacks[0],
            "all_feedbacks": feedbacks,
            "judge_traces": judge_traces,
        }

    def _hermes_evaluation(
        self,
        analysis_output: str,
        rubric: Dict[str, Any],
        task_description: str,
        run_idx: int,
        data_files: List[str] = None,
        model_output_workspace: str = None,
        judge_workspace_dir: str = None,
    ) -> Dict[str, Any]:
        """Perform evaluation using HermesAgent with code execution."""
        from agents.hermes_docker_agent import HermesDockerAgent as HermesAgent

        rubric_text = self._format_rubric(rubric)

        # Prepare judge workspace
        if judge_workspace_dir:
            judge_workspace = Path(judge_workspace_dir) / f"run_{run_idx}"
        else:
            judge_workspace = Path(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "workspaces", "judge", f"run_{run_idx}_{os.getpid()}"
            ))
        judge_workspace.mkdir(parents=True, exist_ok=True)

        # Strategy: copy the model's entire workspace as the judge workspace base,
        # then drop model_response.md alongside it. This mirrors exactly what the
        # tested model saw and produced — its ./inputs/ already contains the raw
        # data, its ./outputs/ (or wherever it chose) contains its deliverables.
        # We copy rather than symlink because the judge runs inside a docker
        # container that only bind-mounts the judge workspace; symlinks to host
        # absolute paths would be broken inside the container.
        if model_output_workspace and os.path.isdir(model_output_workspace):
            for item in os.listdir(model_output_workspace):
                # Skip hidden runtime artifacts (prefill jsonl, judge metadata)
                # and noise (__pycache__). Also skip a colliding model_response.md
                # if any (ours takes priority).
                if item.startswith('.') or item == '__pycache__' or item == 'model_response.md':
                    continue
                src_path = os.path.join(model_output_workspace, item)
                _copy_into(src_path, judge_workspace / item)

        # Fallback: if the model's workspace didn't include ./inputs/ (or it's
        # empty), populate it from the pristine data_files passed by the caller
        # so the judge can still verify numeric claims against original data.
        inputs_dir = judge_workspace / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        if data_files and not any(inputs_dir.iterdir()):
            for src in data_files:
                if os.path.exists(src):
                    _copy_into(src, inputs_dir / os.path.basename(src))

        # Write the model's final assistant message as a sibling file. Done last
        # so it can't be clobbered by the workspace copy above.
        response_file = judge_workspace / "model_response.md"
        response_file.write_text(analysis_output, encoding="utf-8")

        # Build query for hermes judge
        query = f"""你是一个严格的数据分析评估专家。你需要评估一份数据分析报告的质量。

## 原始任务：
{task_description}

## 工作目录说明
当前工作目录 `.` 就是**被测模型当时跑任务的工作区**的完整快照，结构与模型当时看到的完全一致，再外加一个 `./model_response.md`：

- `./model_response.md` — 被测模型最后一条 assistant 回复（**可能只是简短结论，不一定是完整报告**）。
- `./inputs/` — 原始数据文件（模型当时拿到的输入）。
- 其它一切目录与文件（例如 `./outputs/`、`./charts/`、顶层的 .md/.html/.csv/.png/.docx/.ipynb、`./inputs/` 内除原始数据外的文件等）都是**被测模型自己生成的交付物**。被测模型把图表、CSV、报告写到哪个子目录是它自己的选择，没有统一约定。

## 评分标准（Rubric）：
{rubric_text}

## 评估要求：
1. **先把模型的实际交付物找全**：用 `find . -maxdepth 3 -type f | sort`（或 `ls -R`）列出工作目录下所有文件；打开关键的报告类文件（.md / .html / .txt / .docx / .ipynb）以及图表查看实际内容。
2. **绝对不要只看 `./model_response.md` 的简短摘要打分** —— 模型常常把 model_response 写成一句"分析完成，结果保存在 X.md"之类的话，真正的交付内容在工作目录里的具体文件中。务必读完文件再评分。
3. 对报告中的关键数值和结论，编写 Python 代码加载 `./inputs/` 中的原始数据进行验证（若 `./inputs/` 为空或缺失，可放宽数值验证，主要依据交付物本身的完整性、一致性与质量评分）。
4. 根据验证结果和上述 Rubric 逐项评分（每项给 0 到满分之间的分数）。
5. 最终输出严格的 JSON 格式评分结果，格式如下：

```json
{{
  "layer1_scores": [
    {{"criterion": "标准名称", "points_awarded": X, "max_points": Y, "reason": "评分理由"}},
    ...
  ],
  "layer2_scores": [
    {{"dimension": "维度名称", "points_awarded": X, "max_points": Y, "reason": "评分理由"}},
    ...
  ],
  "layer3_scores": [
    {{"criterion": "标准名称", "points_awarded": X, "max_points": Y, "reason": "评分理由"}},
    ...
  ],
  "total_score": XX,
  "summary": "总体评价"
}}
```

注意：total_score 是所有层分数之和（满分100）。请确保最终输出包含上述 JSON。"""

        # Create hermes agent for judging
        profile_name = f"judge_opus_run{run_idx}"
        agent = HermesAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            data_root_path=str(judge_workspace),
            save_name=profile_name,
            max_rounds=self.max_rounds,
            provider=self.provider,
        )

        path_info = {
            "task_id": f"judge_run_{run_idx}",
            "real_input_dir": str(inputs_dir),
            "workspace_dir": str(judge_workspace),
        }

        hermes_result = agent.interact(
            query=query,
            system_prompt="You are a rigorous evaluator of data analysis quality. You verify claims by running code.",
            run_code_func=None,
            path_info=path_info,
        )

        # Parse the result
        content = hermes_result.get("model_response", "")
        result = self._parse_judge_result(content, run_idx)

        # Attach hermes trace to result for caller to save
        result["_hermes_trace"] = {
            "session_id": hermes_result.get("session_id"),
            "profile": hermes_result.get("profile"),
            "history": hermes_result.get("history", []),
            "total_tokens": hermes_result.get("total_tokens", 0),
            "rounds": hermes_result.get("rounds", 0),
            "duration_seconds": hermes_result.get("duration_seconds", 0),
            "tool_calls": hermes_result.get("tool_calls", {}),
            "model_response": content,
        }

        return result

    def _parse_judge_result(self, content: str, run_idx: int) -> Dict[str, Any]:
        """Parse judge result from hermes output."""
        result = self._try_parse_judge_json(content)
        if result is None:
            print(f"Warning: Failed to parse hermes judge result (run {run_idx})")
            result = {
                "layer1_scores": [],
                "layer2_scores": [],
                "layer3_scores": [],
                "total_score": 0,
                "summary": "Parsing failed",
            }

        layer_scores = {
            "layer1": sum(s.get("points_awarded", 0) for s in result.get("layer1_scores", [])),
            "layer2": sum(s.get("points_awarded", 0) for s in result.get("layer2_scores", [])),
            "layer3": sum(s.get("points_awarded", 0) for s in result.get("layer3_scores", [])),
        }

        return {
            "total_score": result.get("total_score", 0),
            "layer_scores": layer_scores,
            "feedback": result,
        }

    @staticmethod
    def _repair_llm_json(s: str) -> str:
        """Repair common LLM-generated JSON quirks that break json.loads.

        - \\' is invalid JSON (single quotes don't need escaping); strip the backslash.
        - Smart quotes are folded to plain ASCII " and '.
        - Trailing commas before } or ] are stripped.
        """
        s = s.replace("\\'", "'")
        s = s.replace("“", '"').replace("”", '"')
        s = s.replace("‘", "'").replace("’", "'")
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        return s

    @staticmethod
    def _iter_balanced_braces(text: str):
        """Yield balanced { ... } substrings of `text`, longest first.

        Skips braces inside double-quoted string literals (handling \\" escapes),
        so JSON-like blocks containing braces inside string values are matched correctly.
        """
        candidates = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] != '{':
                i += 1
                continue
            depth = 0
            j = i
            in_str = False
            escape = False
            while j < n:
                c = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[i:j + 1])
                            break
                j += 1
            i = (j + 1) if depth == 0 else (i + 1)
        candidates.sort(key=len, reverse=True)
        for c in candidates:
            yield c

    @classmethod
    def _try_parse_judge_json(cls, content: str):
        """Try multiple strategies + LLM-JSON repairs. Returns dict or None."""
        def _attempt(s: str):
            # Stage 1: stdlib json on raw + simple-repair forms
            for raw in (s, cls._repair_llm_json(s)):
                try:
                    d = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(d, dict) and "total_score" in d:
                    return d
            # Stage 2: json_repair — tolerates unescaped quotes inside string values,
            # truncated JSON, trailing junk, smart quotes, etc. Required for Chinese
            # judge responses that quote terms like "先天不足" inside reason strings.
            if json_repair is not None:
                try:
                    d = json_repair.loads(s)
                except Exception:
                    d = None
                if isinstance(d, dict) and "total_score" in d:
                    return d
            return None

        # 1) Whole content
        r = _attempt(content)
        if r is not None:
            return r

        # 2) ```json ... ``` fenced blocks (judge may emit multiple — try all)
        for match in re.finditer(r'```(?:json)?\s*(.*?)```', content, re.DOTALL):
            r = _attempt(match.group(1))
            if r is not None:
                return r

        # 3) Brace-balanced scan: longest balanced { ... } block containing total_score
        for block in cls._iter_balanced_braces(content):
            if '"total_score"' not in block:
                continue
            r = _attempt(block)
            if r is not None:
                return r

        return None

    def _single_evaluation(
        self,
        analysis_output: str,
        rubric: Dict[str, Any],
        task_description: str,
        run_idx: int,
    ) -> Dict[str, Any]:
        """Perform a single evaluation run."""
        rubric_text = self._format_rubric(rubric)

        prompt = f"""You are evaluating a data analysis output against a detailed rubric.

# Original Task:
{task_description}

# Analysis Output to Evaluate:
{analysis_output}

# Evaluation Rubric:
{rubric_text}

# Instructions:
1. Evaluate the analysis against each criterion in all three layers
2. Assign points for each criterion (0 to max points)
3. Provide brief justification for each score
4. Calculate total score (out of 100)

Output JSON:
{{
  "layer1_scores": [
    {{"criterion": "...", "points_awarded": X, "max_points": Y, "reason": "..."}},
    ...
  ],
  "layer2_scores": [
    {{"dimension": "...", "points_awarded": X, "max_points": Y, "reason": "..."}},
    ...
  ],
  "layer3_scores": [
    {{"criterion": "...", "points_awarded": X, "max_points": Y, "reason": "..."}},
    ...
  ],
  "total_score": XX,
  "summary": "Overall assessment..."
}}
"""

        # Call judge model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a rigorous evaluator of data analysis quality.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Deterministic for stability
        )

        # Parse response
        try:
            content = response.choices[0].message.content
            # Try direct JSON parse first
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try extracting from markdown code block
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    raise ValueError("No valid JSON found in response")
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: Failed to parse judge result (run {run_idx})")
            result = {
                "layer1_scores": [],
                "layer2_scores": [],
                "layer3_scores": [],
                "total_score": 0,
                "summary": "Parsing failed",
            }

        # Calculate layer totals
        layer_scores = {
            "layer1": sum(s["points_awarded"] for s in result.get("layer1_scores", [])),
            "layer2": sum(s["points_awarded"] for s in result.get("layer2_scores", [])),
            "layer3": sum(s["points_awarded"] for s in result.get("layer3_scores", [])),
        }

        return {
            "total_score": result.get("total_score", 0),
            "layer_scores": layer_scores,
            "feedback": result,
        }

    def _format_rubric(self, rubric: Dict[str, Any]) -> str:
        """Format rubric for the prompt."""
        sections = []

        # Layer 1
        sections.append("## Layer 1 - Must-Find Criteria (50 points):")
        for item in rubric.get("layer1_must_find", []):
            sections.append(
                f"- {item['criterion']}: {item['description']} ({item['points']} points)"
            )

        # Layer 2
        sections.append("\n## Layer 2 - Process Quality (30 points):")
        for item in rubric.get("layer2_process_quality", []):
            sections.append(
                f"- {item['dimension']}: {item['description']} ({item['points']} points)"
            )

        # Layer 3
        sections.append("\n## Layer 3 - Bonus Discovery (20 points):")
        for item in rubric.get("layer3_bonus_discovery", []):
            sections.append(
                f"- {item['criterion']}: {item['description']} ({item['points']} points)"
            )

        return "\n".join(sections)
