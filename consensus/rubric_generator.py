"""Three-layer rubric generation module.

Generates adaptive evaluation rubrics with three layers:
- Layer 1 (Must-Find): Based on consensus findings
- Layer 2 (Process Quality): Universal quality dimensions
- Layer 3 (Bonus Discovery): Rewards novel insights
"""

import json
import re
from typing import List, Dict, Any
from openai import OpenAI


class RubricGenerator:
    """Generates three-layer evaluation rubrics."""

    def __init__(self, api_key: str, base_url: str, model_name: str, language: str = "zh"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.language = language

    def generate(
        self,
        task_description: str,
        consensus_findings: List[Dict],
        data_characteristics: Dict[str, Any] = None,
        non_consensus_findings: List[Dict] = None,
        validated_l3_findings: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate three-layer rubric.

        Args:
            task_description: The original analysis task/question
            consensus_findings: List of consensus findings from ConsensusExtractor
            data_characteristics: Optional dict describing the data (e.g., data types, size)
            non_consensus_findings: List of non-consensus findings
            validated_l3_findings: Cross-validated findings for L3 (if provided, overrides non_consensus for L3)

        Returns:
            Dict with three layers:
            {
              "layer1_must_find": [...],
              "layer2_process_quality": [...],
              "layer3_bonus_discovery": [...],
              "scoring_weights": {"layer1": 0.5, "layer2": 0.3, "layer3": 0.2}
            }
        """
        # Format consensus findings
        consensus_text = self._format_consensus(consensus_findings)
        if validated_l3_findings:
            non_consensus_text = self._format_validated_l3(validated_l3_findings)
        else:
            non_consensus_text = self._format_non_consensus(non_consensus_findings)
        data_text = (
            json.dumps(data_characteristics, indent=2)
            if data_characteristics
            else "Not provided"
        )

        prompt_en = f"""You are designing an evaluation rubric for a data analysis task.

# Task:
{task_description}

# Data Characteristics:
{data_text}

# Consensus Findings (from multiple strong models):
{consensus_text}

# Non-Consensus Findings (cross-validated by other models):
{non_consensus_text}

# Instructions:
Generate a three-layer evaluation rubric:

**Layer 1 - Must-Find (50% weight):**
Based on the consensus findings, create specific evaluation criteria.
Each criterion should check if the analysis identified a key pattern.
Format: List of dicts with 'criterion', 'description', 'points' (total should be 50)

**Layer 2 - Process Quality (30% weight):**
Universal quality dimensions independent of specific findings:
- Numerical accuracy (calculations correct)
- Method appropriateness (right statistical/analytical methods)
- Reasoning coherence (logical flow, no contradictions)
- Conclusion support (claims backed by evidence)
Format: List of dicts with 'dimension', 'description', 'points' (total should be 30)

**Layer 3 - Bonus Discovery (20% weight):**
Based on the cross-validated non-consensus findings listed above (already confirmed by other models), create evaluation criteria for each one.
These findings have been independently verified, so do NOT filter them out. Assign points to all of them (total should be 20).
Format: List of dicts with 'criterion', 'description', 'points' (total should be 20)

Output JSON:
{{
  "layer1_must_find": [
    {{"criterion": "...", "description": "...", "points": 10}},
    ...
  ],
  "layer2_process_quality": [
    {{"dimension": "...", "description": "...", "points": 8}},
    ...
  ],
  "layer3_bonus_discovery": [
    {{"criterion": "...", "description": "...", "points": 10}},
    ...
  ],
  "scoring_weights": {{"layer1": 0.5, "layer2": 0.3, "layer3": 0.2}}
}}
"""

        prompt_zh = f"""你正在为一个数据分析任务设计评估标准（rubric）。

# 任务描述：
{task_description}

# 数据特征：
{data_text}

# 共识发现（来自多个强模型的分析结果）：
{consensus_text}

# 非共识发现（仅被单个模型提到的独有洞察）：
{non_consensus_text}

# 要求：
生成一个三层评估标准：

**第一层 - 必须发现的关键点（权重 50%）：**
基于上述共识发现，创建具体的评估标准。
每条标准应检查分析是否识别出了某个关键模式或规律。
格式：字典列表，包含 'criterion'（标准名称）、'description'（详细描述）、'points'（分值），总分应为 50 分。

**第二层 - 过程质量（权重 30%）：**
与具体发现无关的通用质量维度：
- 数值准确性（计算是否正确）
- 方法适当性（是否使用了正确的统计/分析方法）
- 推理连贯性（逻辑是否通顺，有无自相矛盾）
- 结论支撑度（结论是否有数据/证据支持）
格式：字典列表，包含 'dimension'（维度名称）、'description'（详细描述）、'points'（分值），总分应为 30 分。

**第三层 - 额外发现加分（权重 20%）：**
以下是经过交叉验证的非共识发现（已由其他模型独立确认其正确性和价值）。
请为每条发现编写评分标准，不要过滤掉任何一条，直接为所有发现分配分值（总分 20 分）。
格式：字典列表，包含 'criterion'（标准名称）、'description'（详细描述）、'points'（分值），总分应为 20 分。

请用中文输出所有 criterion/dimension/description 的内容。输出 JSON 格式如下：
{{
  "layer1_must_find": [
    {{"criterion": "...", "description": "...", "points": 10}},
    ...
  ],
  "layer2_process_quality": [
    {{"dimension": "...", "description": "...", "points": 8}},
    ...
  ],
  "layer3_bonus_discovery": [
    {{"criterion": "...", "description": "...", "points": 10}},
    ...
  ],
  "scoring_weights": {{"layer1": 0.5, "layer2": 0.3, "layer3": 0.2}}
}}
"""

        prompt = prompt_zh if self.language == "zh" else prompt_en
        system_msg = (
            "你是一位数据分析任务评估标准设计专家。"
            if self.language == "zh"
            else "You are an expert at designing evaluation rubrics for data analysis tasks."
        )

        # Call LLM for rubric generation
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_msg,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        # Parse response
        try:
            content = response.choices[0].message.content
            try:
                rubric = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    rubric = json.loads(match.group(1))
                else:
                    raise
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse rubric generation result as JSON: {e}")
            rubric = {
                "layer1_must_find": [],
                "layer2_process_quality": [],
                "layer3_bonus_discovery": [],
                "scoring_weights": {"layer1": 0.5, "layer2": 0.3, "layer3": 0.2},
            }

        return rubric

    def _format_consensus(self, consensus_findings: List[Dict]) -> str:
        """Format consensus findings for the prompt."""
        if not consensus_findings:
            return "No consensus findings identified."

        formatted = []
        for i, finding in enumerate(consensus_findings, 1):
            pattern = finding.get("pattern", "")
            freq = finding.get("frequency", 0)
            formatted.append(f"{i}. {pattern} (mentioned by {freq*100:.0f}% of models)")
        return "\n".join(formatted)

    def _format_non_consensus(self, non_consensus_findings: List[Dict]) -> str:
        """Format non-consensus findings for the prompt."""
        if not non_consensus_findings:
            return "No non-consensus findings available."

        formatted = []
        for i, finding in enumerate(non_consensus_findings, 1):
            pattern = finding.get("pattern", "")
            models = finding.get("models", [])
            evidence = finding.get("evidence", [])
            models_str = ", ".join(models) if models else "unknown"
            evidence_str = evidence[0] if evidence else ""
            formatted.append(f"{i}. {pattern} (source: {models_str})\n   Evidence: {evidence_str}")
        return "\n".join(formatted)

    def _format_validated_l3(self, validated_findings: List[Dict]) -> str:
        """Format cross-validated L3 findings for the prompt."""
        if not validated_findings:
            return "No cross-validated findings available."

        formatted = []
        for i, finding in enumerate(validated_findings, 1):
            pattern = finding.get("pattern", "")
            avg_score = finding.get("avg_score", 0)
            source_models = finding.get("source_models", [])
            source_str = ", ".join(source_models) if source_models else "unknown"
            formatted.append(
                f"{i}. {pattern} (validation score: {avg_score:.1f}/10, proposed by: {source_str})"
            )
        return "\n".join(formatted)
