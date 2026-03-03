## 环境配置（uv）

- 安装 `uv`：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 创建并激活环境：

  - 创建环境：`uv venv` (默认创建 `.venv` 目录)
  - 激活环境：`source .venv/bin/activate`

- 安装依赖：

  - **安装所有功能**（全家桶）：
    ```bash
    uv sync --all-extras
    ```

  **可用功能组说明**：

  - `analysis`: 纯数据分析（numpy, pandas, matplotlib, scipy 等）
  - `excel`: Excel 读写增强（xlsxwriter, pyxlsb, calamine）
  - `docx`: Word 文档处理（python-docx, docxtpl 等）
  - `pptx`: PPT 处理（python-pptx, pptxtopdf 等）
  - `pdf`: PDF 处理（pypdf, pdfminer, camelot 等）
  - `image`: 图像处理（pillow, opencv, heif/avif 支持等）
  - `ocr`: OCR 识别（tesseract, easyocr）
  - `convert`: 文档格式转换服务
  - `aspose_cloud`: Aspose Cloud SDK
  - `all`: 包含以上所有

**注意**：

- 如果你希望在当前终端会话中直接使用 `python` 等命令，需要执行 `source .venv/bin/activate`。
- 如果你使用 `uv run`（例如 `uv run python script.py`），则无需手动激活，uv 会自动使用环境。

## 下载数据集

安装环境后，请运行以下命令下载数据集（需确保网络畅通）：

```bash
uv run python download_data.py
```

## 配置 .env

- 复制 `.env.example` 为 `.env`：`cp .env.example .env`
- 填写以下变量：
  - `CHART_EVAL_API_URL`
  - `CHART_EVAL_API_KEY`
  - `CHART_EVAL_MODEL_NAME = gemini-3-pro-preview `
  - `NUMERICAL_EVAL_API_URL`
  - `NUMERICAL_EVAL_API_KEY`
  - `NUMERICAL_EVAL_MODEL_NAME = QwQ-32B`
  - `FILE_GENERATION_EVAL_API_URL`
  - `FILE_GENERATION_EVAL_API_KEY`
  - `FILE_GENERATION_EVAL_MODEL_NAME = claude-sonnet-4-5-20250929`

## 运行评测

### 1. 运行推理与评估

```bash
# ====== Config ======
MODEL_NAME="YOUR_MODEL_NAME"
SAVE_NAME="YOUR_SAVE_NAME"
BASE_URL="http://YOUR_API_BASE_URL/v1"
API_KEY="YOUR_API_KEY"
# ====================

# 运行推理 (dataset=all 表示同时运行 QA, data_visualization, file_generation)
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

# 运行评估
uv run python evaluation/run.py --dataset file_generation --model_name "${SAVE_NAME}" --max_workers 10
uv run python evaluation/run.py --dataset QA --model_name "${SAVE_NAME}" --max_workers 5
uv run python evaluation/run.py --dataset data_visualization --model_name "${SAVE_NAME}" --max_workers 5
```

## 数据目录结构

- `data/`：存放题目与标准答案（GT）
  - `chart/`：图表可视化数据与图像
  - `numerical/`：数值统计数据
  - `editing/`：数据编辑数据与表格
- `preds/`：存放模型预测结果
  - `<model_name>/`：按模型名称区分的子目录
    - `chart/`
    - `numerical/`
    - `editing/`
