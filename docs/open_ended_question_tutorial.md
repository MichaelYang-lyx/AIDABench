# Open-Ended Question 评估教程

## 框架概述

AIDABench 的 Open-Ended Question 评估采用 **ConsensusEval** 框架，核心思路是：用多个参考模型独立分析同一份数据，提取共识发现作为评分标准，再用 LLM Judge 对被评估模型的输出打分。

整个 pipeline 分为两大阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Stage 1: Infer（推理）                         │
│  被评估模型通过 Hermes Agent 分析数据，生成分析报告                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Stage 2: Eval（评估）                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Reference    │    │ Consensus    │    │ Cross-Validation     │  │
│  │ Model Infer  │───▶│ Extraction   │───▶│ (非共识发现验证)       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                     │               │
│                                                     ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Final Score  │◀───│ LLM Judge    │◀───│ Rubric Generation    │  │
│  │              │    │ (Hermes)     │    │ (三层评分标准)         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 评估流程详解

1. **Reference Model Infer** — 多个参考模型（如 gemini、claude、deepseek）各自通过 Hermes Agent 分析数据，产出独立的 findings
2. **Consensus Extraction** — 语义对齐多个模型的输出，按设定阈值（如 60%）划分共识发现（L1）和非共识发现
3. **Cross-Validation** — 恢复参考模型的 Hermes session，让其他模型验证非共识发现的正确性（L3）
4. **Rubric Generation** — 基于共识和验证结果生成三层评分标准：
   - L1 (Must-Find): 共识发现，必须覆盖
   - L2 (Process Quality): 分析过程质量
   - L3 (Bonus Discovery): 额外有价值的发现
5. **LLM Judge** — Judge 模型通过 Hermes Agent 对被评估模型的输出逐项打分

---

## 配置说明

### 1. 环境准备

```bash
cd /data/projects/AIDABench
source .venv/bin/activate
```

确保已安装 `hermes` CLI 并可在 PATH 中访问。

### 2. 配置文件：`configs/reference_models.json`

这是评估的核心配置文件，定义了参考模型、共识提取模型和 Judge 模型。

```json
{
  "models": [
    {
      "name": "gemini-3.1-pro-preview",
      "api_url": "https://your-api-gateway/v1",
      "api_key": "sk-xxx",
      "model_id": "gemini-3.1-pro-preview",
      "temperature": 0.7,
      "max_tokens": 32768
    },
    {
      "name": "claude-opus-4-6",
      "api_url": "https://your-api-gateway/v1",
      "api_key": "sk-xxx",
      "model_id": "claude-opus-4-6",
      "provider": "anthropic",
      "temperature": 0.7,
      "max_tokens": 32768
    },
    {
      "name": "deepseek-v4-pro",
      "api_url": "https://your-api-gateway/v1",
      "api_key": "sk-xxx",
      "model_id": "deepseek-v4-pro",
      "temperature": 0.7,
      "max_tokens": 32768
    }
  ],
  "consensus_threshold": 0.6,
  "consensus_model": {
    "api_url": "https://your-api-gateway/v1",
    "api_key": "sk-xxx",
    "model_id": "gemini-3.1-pro-preview",
    "temperature": 0.0
  },
  "judge_model": {
    "api_url": "https://your-api-gateway/v1",
    "api_key": "sk-xxx",
    "model_id": "claude-opus-4-6",
    "provider": "anthropic",
    "temperature": 0.7
  },
  "num_judge_runs": 1
}
```

#### 字段说明

| 字段 | 说明 |
|------|------|
| `models` | 参考模型列表，建议 3 个以上以确保共识质量 |
| `models[].name` | 模型标识名，用于缓存目录命名 |
| `models[].api_url` | API 网关地址（OpenAI 兼容格式，以 `/v1` 结尾） |
| `models[].api_key` | API 密钥 |
| `models[].model_id` | 实际模型 ID |
| `models[].provider` | 模型提供商。Anthropic 模型必须设为 `"anthropic"`，其他模型可省略（默认 `"custom"`） |
| `consensus_threshold` | 共识阈值，0.6 表示 ≥60% 模型提到的发现算共识 |
| `consensus_model` | 用于语义对齐/共识提取的模型（建议用 temperature=0） |
| `judge_model` | 最终评分的 Judge 模型 |
| `num_judge_runs` | Judge 评分次数（多次取平均可提高稳定性） |

#### provider 字段注意事项

- Anthropic 模型（如 claude-opus-4-6）**必须**设置 `"provider": "anthropic"`，否则 Hermes 无法正确路由 API 调用
- 其他通过 OpenAI 兼容接口调用的模型不需要设置 provider

### 3. 数据集配置：`data/open_ended_test/test_tasks.json`

定义评估任务列表：

```json
[
  {
    "task_id": "open_ended_001",
    "query": "Analyze this incident management dataset...",
    "dataset_csv_path": "data/open_ended_test/input/open_ended_001/flag-1.csv",
    "metadata": {
      "goal": "Find the discrepancy...",
      "role": "L2 Support Agent",
      "category": "Incidents Management",
      "dataset_description": "The dataset comprises 500 entries..."
    }
  }
]
```

每个任务需要：
- `task_id`: 唯一标识
- `query`: 分析任务描述
- `dataset_csv_path`: 数据文件路径（相对于项目根目录）
- `metadata`: 任务元信息（用于 rubric 生成）

---

## 使用方法

### Stage 1: Infer（被评估模型推理）

让被评估模型分析数据集中的每个任务：

```bash
cd /data/projects/AIDABench
python infer/run.py \
    --dataset open_ended_test \
    --api_key sk-xxx \
    --base_url https://your-api-gateway/v1 \
    --model_name claude-opus-4-6 \
    --save_name claude-opus-4-6 \
    --agent_type hermes \
    --max_rounds 60 \
    --num_workers 2 \
    --provider anthropic
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集名称，对应 `data/` 下的目录 |
| `--api_key` | 被评估模型的 API key |
| `--base_url` | API 网关地址 |
| `--model_name` | 模型 ID |
| `--save_name` | 输出目录名（默认同 model_name） |
| `--agent_type` | Agent 类型，open-ended 任务用 `hermes` |
| `--max_rounds` | Hermes 最大交互轮数（建议 60） |
| `--num_workers` | 并行 worker 数 |
| `--provider` | Anthropic 模型需设为 `anthropic` |

输出目录结构：
```
output/preds/{save_name}/open_ended_test/
├── conv/                    # 推理结果
│   ├── open_ended_001.json
│   └── open_ended_002.json
└── workspace/               # Hermes 工作空间（含生成的代码和文件）
    ├── open_ended_001/
    └── open_ended_002/
```

### Stage 2: Eval（评估）

```bash
cd /data/projects/AIDABench
python evaluation/runner/eval_open_ended.py \
  --input_path output/preds/claude-opus-4-6/open_ended_test/conv \
  --output_path output/eval/claude-opus-4-6/open_ended_test \
  --config_path configs/reference_models.json \
  --dataset open_ended_test \
  --language zh \
  --max_workers 2 \
  --use_cache
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--input_path` | 被评估模型的推理结果目录（conv 目录） |
| `--output_path` | 评估结果输出目录 |
| `--config_path` | 参考模型配置文件路径 |
| `--dataset` | 数据集名称 |
| `--language` | 输出语言（`zh` 或 `en`） |
| `--max_workers` | 任务级并行 worker 数（同时处理多个 task） |
| `--ref_max_workers` | 参考模型级并行 worker 数（同时运行多个参考模型） |
| `--use_cache` | 启用缓存（强烈建议），避免重复调用参考模型 |

### Build Cache（单独生成参考缓存）

在评估多个被测模型前，可以先单独生成 reference cache，避免重复调用参考模型：

```bash
# 串行（默认）
python evaluation/runner/eval_open_ended.py build-cache \
  --task_config data/open_ended_test/test_tasks.json

# 参考模型并行（4 个参考模型并行）
python evaluation/runner/eval_open_ended.py build-cache \
  --task_config data/open_ended_test/test_tasks.json \
  --ref_max_workers 4

# task 级 + 参考模型级双重并行
python evaluation/runner/eval_open_ended.py build-cache \
  --task_config data/open_ended_test/test_tasks.json \
  --max_workers 2 \
  --ref_max_workers 4
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `--task_config` | 任务配置文件路径（默认 `data/open_ended_test/test_tasks.json`） |
| `--config_path` | 参考模型配置文件路径（默认 `configs/reference_models.json`） |
| `--max_workers` | 任务级并行 worker 数 |
| `--ref_max_workers` | 参考模型级并行 worker 数 |
| `--use_cache` | 启用缓存，跳过已完成的任务 |

> **注**：`build-cache` 生成的缓存与 `eval` 命令共享，先跑 `build-cache` 再跑 `eval --use_cache` 可显著节省时间。

#### 缓存机制

启用 `--use_cache` 后，中间结果会缓存到 `output/reference_cache/{dataset}/` 目录：

```
output/reference_cache/open_ended_test/
└── open_ended_001/
    ├── gemini-3.1-pro-preview/
    │   ├── response.json              # 模型分析结果
    │   ├── trace_infer.json           # Infer 阶段完整交互记录
    │   ├── trace_cross_validation.json # Cross-validation 交互记录
    │   └── workspace/                 # Hermes 工作空间
    ├── claude-opus-4-6/
    │   ├── response.json
    │   ├── trace_infer.json
    │   ├── trace_cross_validation.json
    │   └── workspace/
    ├── deepseek-v4-pro/
    │   └── ...
    ├── consensus.json                 # 共识提取结果
    ├── rubric.json                    # 生成的三层评分标准
    └── cross_validation.json          # Cross-validation 汇总结果
```

缓存是增量的：
- 如果某个参考模型已有缓存，不会重新跑
- 如果 consensus + rubric 已缓存，直接跳到 Judge 阶段
- 如果 cross_validation 已缓存，跳过验证阶段

要重新生成某个阶段的结果，删除对应的缓存文件即可。

---

## 评估多个模型

Eval 命令的 `--input_path` 指向不同模型的推理结果即可。Reference cache 是共享的，只需生成一次：

```bash
# 评估模型 A
python evaluation/runner/eval_open_ended.py \
  --input_path output/preds/model-a/open_ended_test/conv \
  --output_path output/eval/model-a/open_ended_test \
  --config_path configs/reference_models.json \
  --dataset open_ended_test \
  --language zh \
  --use_cache

# 评估模型 B（复用同一份 reference cache）
python evaluation/runner/eval_open_ended.py \
  --input_path output/preds/model-b/open_ended_test/conv \
  --output_path output/eval/model-b/open_ended_test \
  --config_path configs/reference_models.json \
  --dataset open_ended_test \
  --language zh \
  --use_cache
```

---

## 常见问题

### Anthropic 模型报错 "No Anthropic credentials found"

确保配置中设置了 `"provider": "anthropic"`。Hermes 对 anthropic provider 会通过环境变量 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_BASE_URL` 传递凭证。

### 某个参考模型在 cross-validation 中缺失

可能原因：
- 网络超时（检查日志中的 timeout 信息）
- API 网关对该模型不稳定
- 该模型的 infer trace 未正确保存 session_id

解决：删除 `cross_validation.json` 缓存后重跑。

### 如何更换参考模型

修改 `configs/reference_models.json` 中的 `models` 列表，然后删除对应 task 的整个缓存目录重新生成：

```bash
rm -rf output/reference_cache/open_ended_test/open_ended_001/
```

### 如何只重跑 Judge 阶段

保留 reference cache 中的 `consensus.json`、`rubric.json`、`cross_validation.json`，删除 `output/eval/` 下的评估结果文件即可。

---

## 输出结构

### reference_cache 目录

参考模型运行结果缓存在 `output/reference_cache/{dataset}/{task_id}/`：

```
output/reference_cache/open_ended_test/
└── open_ended_001/
    ├── gemini-3.1-pro-preview/
    │   ├── response.json               # 模型最终分析结果（用于共识提取）
    │   ├── trace_infer.json            # 推理阶段完整记录（两轮对话历史）
    │   ├── trace_cross_validation.json # Cross-validation 阶段对话历史
    │   └── workspace/                  # Hermes 工作空间（代码、图表等产出）
    ├── claude-opus-4-6/
    │   └── ...（同上）
    ├── deepseek-v4-pro/
    │   └── ...（同上）
    ├── consensus.json                  # 共识提取结果（含 consensus/non-consensus findings）
    ├── rubric.json                     # 三层评分标准
    └── cross_validation.json           # Cross-validation 汇总结果
```

### eval 输出目录

评估结果保存在 `output/eval/{model_name}/{dataset}/`：

```
output/eval/claude-opus-4-6/open_ended_test/
├── open_ended_001.json    # 单任务评估结果（含三层评分明细）
├── open_ended_002.json
└── summary.json           # 所有任务汇总得分
```

`summary.json` 格式：

```json
{
  "model": "claude-opus-4-6",
  "dataset": "open_ended_test",
  "scores": {
    "open_ended_001": 90,
    "open_ended_002": 72
  },
  "average_score": 81.0
}
```

> **注**：得分范围 0–100，由三层评分标准加权计算：L1（Must-Find）×0.5 + L2（Process Quality）×0.3 + L3（Bonus）×0.2
