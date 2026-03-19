<div align="center">

<img src="resources/logo.png" alt="AIDABench Logo" width="420"/>

# AIDABench: AI Data Analytics Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2603.15636-b31b1b.svg)](https://arxiv.org/abs/2603.15636)
[![HuggingFace](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMichaelYang-lyx%2FAIDA&query=%24.downloads&label=%F0%9F%A4%97%20HuggingFace&suffix=%20downloads&color=yellow)](https://huggingface.co/datasets/MichaelYang-lyx/AIDA)
[![OpenCompass](https://img.shields.io/badge/OpenCompass-Hub-4B0082)](https://hub.opencompass.org.cn/dataset-detail/AIDABench)
[![GitHub](https://img.shields.io/badge/GitHub-AIDABench-blue?logo=github)](https://github.com/MichaelYang-lyx/AIDABench)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

**A comprehensive benchmark for evaluating AI systems on end-to-end data analytics over real-world documents.**

[**Paper**](https://arxiv.org/abs/2603.15636) | [**Dataset**](https://huggingface.co/datasets/MichaelYang-lyx/AIDA) | [**OpenCompass**](https://hub.opencompass.org.cn/dataset-detail/AIDABench) | [**Code**](https://github.com/MichaelYang-lyx/AIDABench) | [**中文版**](README_zh.md)

---

</div>

## Quick Start

### 1. Environment Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv
source .venv/bin/activate

# Install all dependencies
uv sync --all-extras
```

<details>
<summary><b>Available dependency groups</b></summary>

| Group          | Description                              |
| :------------- | :--------------------------------------- |
| `analysis`     | numpy, pandas, matplotlib, scipy, etc.   |
| `excel`        | xlsxwriter, pyxlsb, calamine             |
| `docx`         | python-docx, docxtpl, etc.               |
| `pptx`         | python-pptx, pptxtopdf, etc.             |
| `pdf`          | pypdf, pdfminer, camelot, etc.           |
| `image`        | pillow, opencv, heif/avif support        |
| `ocr`          | tesseract, easyocr                       |
| `convert`      | Document format conversion (LibreOffice) |
| `aspose_cloud` | Aspose Cloud SDK                         |
| `all`          | All of the above                         |

</details>

### 2. Download Dataset

```bash
uv run python download_data.py
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in the following:

```env
# Chart Evaluation (Gemini)
CHART_EVAL_API_URL=
CHART_EVAL_API_KEY=
CHART_EVAL_MODEL_NAME=gemini-3-pro-preview

# Numerical Evaluation (QwQ)
NUMERICAL_EVAL_API_URL=
NUMERICAL_EVAL_API_KEY=
NUMERICAL_EVAL_MODEL_NAME=QwQ-32B

# File Generation Evaluation (Claude)
FILE_GENERATION_EVAL_API_URL=
FILE_GENERATION_EVAL_API_KEY=
FILE_GENERATION_EVAL_MODEL_NAME=claude-sonnet-4-5-20250929
```

### 4. Run Inference & Evaluation

```bash
# ====== Config ======
MODEL_NAME="YOUR_MODEL_NAME"
SAVE_NAME="YOUR_SAVE_NAME"
BASE_URL="http://YOUR_API_BASE_URL/v1"
API_KEY="YOUR_API_KEY"
# ====================

# Run inference (dataset=all runs QA, data_visualization, file_generation)
uv run infer/run.py \
  --dataset all \
  --base_url "${BASE_URL}" \
  --api_key "${API_KEY}" \
  --model_name "${MODEL_NAME}" \
  --save_name "${SAVE_NAME}" \
  --num_workers 10 \
  --prompt_file openai_tool_general_20round.txt \
  --agent_type "openai_subprocess_agent" \
  --max_rounds 20

# Run evaluation
uv run python evaluation/run.py --dataset file_generation --model_name "${SAVE_NAME}" --max_workers 10
uv run python evaluation/run.py --dataset QA --model_name "${SAVE_NAME}" --max_workers 5
uv run python evaluation/run.py --dataset data_visualization --model_name "${SAVE_NAME}" --max_workers 5
```

## Overview

Existing benchmarks often focus on isolated capabilities or simplified scenarios. **AIDABench** bridges this gap by providing end-to-end data analytics tasks spanning heterogeneous data sources — spreadsheets, databases, financial reports, and operational records.

<div align="center">
<img src="resources/figure1_overview.png" alt="AIDABench Overview" width="90%"/>
<br/>
<em>Figure 1: Overview of the AIDABench evaluation framework.</em>
</div>

## Task Categories

AIDABench is organized around three primary capability dimensions:

| Category               | Proportion | Description                                                                          |
| :--------------------- | :--------: | :----------------------------------------------------------------------------------- |
| **File Generation**    |   43.3%    | Data wrangling — filtering, normalization, deduplication, joins, cross-sheet linkage |
| **Question Answering** |   37.5%    | Analytical queries — aggregation, ranking, comparisons, trend analysis               |
| **Data Visualization** |   19.2%    | Chart creation — bar/line/pie charts with style requirements and constraints         |

### Task Complexity

| Level  | Proportion | Reasoning Steps |
| :----- | :--------: | :-------------- |
| Easy   |   29.5%    | ≤ 6 steps       |
| Medium |   49.4%    | 7–12 steps      |
| Hard   |   21.1%    | ≥ 13 steps      |

> **27.4%** of tasks require cross-file reasoning over multiple input files (up to 14 files).

<div align="center">
<img src="resources/figure2_scenarios.png" alt="Evaluation Scenarios" width="90%"/>
<br/>
<em>Figure 2: Example evaluation scenarios for QA, Data Visualization, and File Generation.</em>
</div>

## Evaluation Framework

All models are evaluated under a unified **tool-augmented protocol**: the model receives task instructions and associated files, then executes **arbitrary Python code** within a **sandboxed environment** to complete the task.

Three dedicated **LLM-based evaluators** are used:

| Evaluator                   | Target           | Approach                                       |
| :-------------------------- | :--------------- | :--------------------------------------------- |
| **QA Evaluator**            | Textual answers  | Binary judge for answer correctness            |
| **Visualization Evaluator** | Charts & figures | Scores correctness + readability               |
| **File Evaluator**          | Spreadsheets     | Coarse-to-fine structural & content validation |

<div align="center">
<img src="resources/figure3_evaluators.png" alt="Evaluator Design" width="90%"/>
<br/>
<em>Figure 3: The design of the three types of evaluators in AIDABench.</em>
</div>

## Citation

If you find AIDABench useful for your research, please cite our paper:

```bibtex
@article{yang2026aidabench,
  title={AIDABench: AI Data Analytics Benchmark},
  author={Yang, Yibo and Lei, Fei and Sun, Yixuan and Zeng, Yantao and Lv, Chengguang and Hong, Jiancao and Tian, Jiaojiao and Qiu, Tianyu and Wang, Xin and Chen, Yanbing and Li, Yanjie and Pan, Zheng and Zhou, Xiaochen and Chen, Guanzhou and Lv, Haoran and Xu, Yuning and Ou, Yue and Liu, Haodong and He, Shiqi and Jia, Anya and Xin, Yulei and Wu, Huan and Liu, Liang and Ge, Jiaye and Dong, Jianxin and Lin, Dahua and Sun, Wenxiu},
  journal={arXiv preprint arXiv:2603.15636},
  year={2026}
}
```
