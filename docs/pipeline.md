# AIDABench Open-Ended Evaluation Pipeline

## Infer 阶段

被测模型通过 HermesAgent 对数据集进行多轮代码执行分析，生成分析结果（model_response）。

## Eval 阶段

### Step 1: Reference Models 独立分析

三个强模型各自独立跑一遍完整的数据分析（同样通过 HermesAgent 多轮代码执行）：
- GPT-5.5
- Claude Sonnet 4.5
- Gemini 3.1 Pro

每个模型独立产出一份 insights（关键发现列表），并保留 session_id 和 workspace。

### Step 2: Consensus 提取

将三个模型的 insights 进行语义对比，筛选出 ≥60% 模型认同的发现，形成 consensus_findings；剩余为 non_consensus_findings（仅被单个模型提到）。

### Step 3: L3 交叉验证

对每条 non_consensus finding（由模型 A 提出）：
1. 通过 `hermes --resume {session_id}` 继续模型 B、C 的 session（保留原始数据访问和代码执行能力）
2. 让 B、C 对该发现打分（10分制）：事实是否正确？是否有价值？
3. 汇总所有非共识发现的交叉验证分数，选取得分最高的 top 10 作为 L3 候选

### Step 4: Rubric 生成

基于前序步骤的结果自动生成三层评分标准：
- Layer 1 (Must Find, 50分)：基于 consensus_findings 生成的关键评分点
- Layer 2 (Process Quality, 30分)：分析过程质量（数值准确性、方法适当性、推理连贯性、结论支撑度）
- Layer 3 (Bonus Discovery, 20分)：基于交叉验证通过的 top 10 非共识发现生成加分项

### Step 5: Judge 评分

Judge model（claude-opus-4-6）通过 HermesAgent 按照 rubric 对被测模型的 model_response 进行评分：
- Judge 可执行代码验证被测模型报告中的数值和结论
- Workspace 保留在 output/evals/{模型名称}/workspace/{task_id}/ 中
- 输出逐项评分和总分

