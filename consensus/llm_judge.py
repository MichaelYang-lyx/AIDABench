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


class LLMJudge:
    """Evaluates analysis outputs using LLM-as-judge with three-layer rubric."""

    def __init__(self, api_key: str, base_url: str, model_name: str,
                 use_hermes: bool = False, max_rounds: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.use_hermes = use_hermes
        self.max_rounds = max_rounds
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluate(
        self,
        analysis_output: str,
        rubric: Dict[str, Any],
        task_description: str,
        num_runs: int = 1,
        data_files: List[str] = None,
        model_output_workspace: str = None,
    ) -> Dict[str, Any]:
        """Evaluate an analysis output against a rubric.

        Args:
            analysis_output: The model's analysis to evaluate
            rubric: Three-layer rubric from RubricGenerator
            task_description: The original task/question
            num_runs: Number of independent evaluation runs (default: 1)
            data_files: List of original data file paths (for Hermes mode)
            model_output_workspace: Path to the tested model's workspace (for Hermes mode)

        Returns:
            Dict with scores, mean_score, std_score, variance, layer_scores, detailed_feedback
        """
        scores = []
        layer_scores_list = []
        feedbacks = []

        for run_idx in range(num_runs):
            if self.use_hermes:
                result = self._hermes_evaluation(
                    analysis_output, rubric, task_description, run_idx,
                    data_files=data_files,
                    model_output_workspace=model_output_workspace,
                )
            else:
                result = self._single_evaluation(
                    analysis_output, rubric, task_description, run_idx
                )
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
        }

    def _hermes_evaluation(
        self,
        analysis_output: str,
        rubric: Dict[str, Any],
        task_description: str,
        run_idx: int,
        data_files: List[str] = None,
        model_output_workspace: str = None,
    ) -> Dict[str, Any]:
        """Perform evaluation using HermesAgent with code execution."""
        from agents.hermes_agent import HermesAgent

        rubric_text = self._format_rubric(rubric)

        # Prepare judge workspace
        judge_workspace = Path(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workspaces", "judge", f"run_{run_idx}_{os.getpid()}"
        ))
        judge_workspace.mkdir(parents=True, exist_ok=True)

        # Write model response to workspace
        response_file = judge_workspace / "model_response.md"
        response_file.write_text(analysis_output, encoding="utf-8")

        # Symlink original data files into workspace/inputs/
        inputs_dir = judge_workspace / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        if data_files:
            for src in data_files:
                if os.path.exists(src):
                    dst = inputs_dir / os.path.basename(src)
                    if not dst.exists() and not dst.is_symlink():
                        dst.symlink_to(os.path.abspath(src))

        # Copy/symlink model output workspace files
        model_output_dir = judge_workspace / "model_output"
        model_output_dir.mkdir(exist_ok=True)
        if model_output_workspace and os.path.isdir(model_output_workspace):
            for item in os.listdir(model_output_workspace):
                src_path = os.path.join(model_output_workspace, item)
                dst_path = model_output_dir / item
                if os.path.isfile(src_path) and not dst_path.exists():
                    dst_path.symlink_to(os.path.abspath(src_path))

        # Build query for hermes judge
        query = f"""你是一个严格的数据分析评估专家。你需要评估一份数据分析报告的质量。

## 原始任务：
{task_description}

## 被测模型的分析报告：
见 ./model_response.md

## 原始数据文件：
位于 ./inputs/ 目录

## 评分标准（Rubric）：
{rubric_text}

## 评估要求：
1. 先阅读 ./model_response.md 中被测模型的分析报告
2. 对报告中的关键数值和结论，编写 Python 代码加载 ./inputs/ 中的原始数据进行验证
3. 根据验证结果和上述 Rubric 逐项评分（每项给 0 到满分之间的分数）
4. 最终输出严格的 JSON 格式评分结果，格式如下：

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

        # Cleanup workspace
        try:
            shutil.rmtree(str(judge_workspace))
        except Exception:
            pass

        return result

    def _parse_judge_result(self, content: str, run_idx: int) -> Dict[str, Any]:
        """Parse judge result from hermes output."""
        try:
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    # Try to find JSON object in the text
                    match = re.search(r'\{[^{}]*"total_score"[^{}]*\}', content, re.DOTALL)
                    if match:
                        result = json.loads(match.group(0))
                    else:
                        # Last resort: find the largest JSON-like block
                        matches = re.findall(r'\{.*?\}', content, re.DOTALL)
                        result = None
                        for m in reversed(matches):
                            try:
                                candidate = json.loads(m)
                                if "total_score" in candidate:
                                    result = candidate
                                    break
                            except json.JSONDecodeError:
                                continue
                        if result is None:
                            raise ValueError("No valid JSON found")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse hermes judge result (run {run_idx}): {e}")
            result = {
                "layer1_scores": [],
                "layer2_scores": [],
                "layer3_scores": [],
                "total_score": 0,
                "summary": f"Parsing failed: {e}",
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
