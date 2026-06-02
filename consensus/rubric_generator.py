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

CRITICAL rules for Layer 1:
1. PRESERVE NULL / NEGATIVE FINDINGS. If a consensus finding is a null result (e.g. "no significant
   trend", "no correlation between X and Y", "distribution is uniform across categories", "the goal's
   premise is not supported by the data"), it MUST appear in Layer 1 as-is. Do NOT silently rewrite
   it as a positive claim. The criterion should reward analyses that explicitly state the absence
   with evidence, and should NOT reward analyses that fabricate a trend or correlation to satisfy
   the goal's wording.
2. SURFACE DATA / TASK MISMATCH AS THE FIRST ITEM. If the consensus findings indicate the dataset
   lacks the columns or fields the task requires (e.g. task asks about expense amounts but no
   amount column exists), the FIRST Layer 1 criterion MUST be "correctly identifies that the data
   does not match the task (specify the missing fields)". Do NOT replace Layer 1 with criteria for
   the alternative analysis the models did on whatever columns happened to be present — that
   alternative belongs (at most) in Layer 3.
3. KEEP DESCRIPTIONS GENERIC AND PATTERN-LEVEL. Each `description` must describe the pattern or
   conclusion the analysis should reach, NOT the exact numeric values, dates, or named entities
   from the consensus findings. For example, write "identifies a sustained increase in workload
   for at least one specific agent" instead of "identifies that Charlie's tickets rose from 12
   in May to 38 in October". The criterion should match any analysis that arrives at the same
   pattern, even if it cites different numbers or different entity names. Numbers/entities may
   appear at most as parenthetical hints (e.g. "(such as Charlie or Howard)"), never as required
   matches.

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

第一层的关键规则（必须遵守）：
1. **保留 null / 否定型共识发现**。如果某条共识发现本身就是"否定结论"（例如"未观察到显著趋势"、
   "X 与 Y 之间无显著相关"、"分布在各类别上均衡"、"任务预设的假设在数据中不成立"等），
   该结论必须原样作为第一层标准列出，禁止被改写成正向陈述。该标准应当奖励"用证据明确指出
   X 不存在"的分析，并不应当奖励"为了配合任务措辞而臆造趋势/相关性"的分析。
2. **将"数据与任务不匹配"作为第一条**。如果共识发现指出数据集缺少任务所需的关键列/字段
  （例如任务要求分析金额但无 amount 列），第一层的第一条必须是"正确识别数据与任务不匹配
   （需具体列出缺失字段）"。**禁止**用模型在其他可用列上做出的替代分析来填充第一层——
   那类替代分析最多放到第三层加分项。
3. **description 必须保持模式级、通用化**。每条 description 应描述分析应当得出的"模式或结论"，
   **不要**把共识发现里的具体数字、日期、人名、产品名硬编码进 description。
   例如写"识别出至少一名特定坐席的工作量呈持续上升趋势"，而不是"识别出 Charlie 的工单从 5 月
   12 件升到 10 月 38 件"。一条 criterion 应该能匹配任何得出相同模式的分析——哪怕该分析引用了
   不同的数字或不同的人名。具体数字/人名最多作为括注提示（如"（例如 Charlie 或 Howard）"），
   绝不作为必须命中的硬性匹配条件。

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

        # Call LLM for rubric generation, with retry on parse failure
        max_attempts = 3
        rubric = None
        last_err = None
        for attempt in range(max_attempts):
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
            content = response.choices[0].message.content
            try:
                rubric = json.loads(content)
                break
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    try:
                        rubric = json.loads(match.group(1))
                        break
                    except json.JSONDecodeError as e:
                        last_err = e
                        print(f"  [rubric] attempt {attempt+1}/{max_attempts}: fenced JSON parse failed: {e}")
                        continue
                last_err = "no JSON object in response"
                print(f"  [rubric] attempt {attempt+1}/{max_attempts}: no JSON object found in response")

        if rubric is None or not rubric.get("layer1_must_find"):
            # Surface the failure loudly instead of writing an empty rubric silently
            raise RuntimeError(
                f"Rubric generation failed after {max_attempts} attempts (last error: {last_err}). "
                f"Caller should not write this empty rubric to cache."
            )

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
