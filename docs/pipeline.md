# AIDABench Open-Ended Evaluation Pipeline

## Infer 阶段

被测模型通过 HermesAgent 对数据集进行多轮代码执行分析，生成分析结果（model_response）。

## Eval 阶段

### Step 1: Reference Models 独立分析

三个强模型各自独立跑一遍完整的数据分析（同样通过 HermesAgent 多轮代码执行）：
- GPT-5.5
- Claude 4.6
- Gemini 3.1

每个模型独立产出一份 insights（关键发现列表）。

### Step 2: Consensus 提取

将三个模型的 insights 进行对比，筛选出 ≥60% 模型认同的发现，形成 consensus_findings。

### Step 3: Rubric 生成

基于 consensus_findings 和 non_consensus_findings 自动生成三层评分标准：
- Layer 1 (Must Find)：基于共识发现（≥60% 模型认同）生成的关键评分点
- Layer 2 (Process Quality)：分析过程质量（数值准确性、方法适当性、推理连贯性、结论支撑度）
- Layer 3 (Bonus Discovery)：从低频发现（仅被单个模型提到的独有洞察）中，由 Judge 模型验证其合理性与价值后，筛选出至多 5 个作为额外加分项

### Step 4: Judge 评分

Judge model 按照 rubric 对被测模型的 model_response 进行评分，重复 5 次取平均，输出最终分数。
