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

## 配置 .env

- 复制 `.env.example` 为 `.env`：`cp .env.example .env`
- 填写以下变量：
  - `CHART_EVAL_API_URL`
  - `CHART_EVAL_API_KEY`
  - `CHART_EVAL_MODEL_NAME`
  - `NUMERICAL_EVAL_API_URL`
  - `NUMERICAL_EVAL_API_KEY`
  - `NUMERICAL_EVAL_MODEL_NAME`

## 运行评测

- 使用 uv 运行：

  ```bash
  uv run python auto_eval.py --dataset numerical_statistics_mini --model_name qwq-32b
  ```

  ```bash
  uv run python auto_eval.py --dataset all_mini --model_name qwq-32b

  uv run python auto_eval.py --dataset file_generation_mini --model_name qwq-32b
  ```

- 或使用脚本（默认使用 `qwq-32b`）：
  ```bash
  ./run_eval.sh numerical_statistics_mini
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

## 新增工具脚本

### 1. 数据转换工具 (Excel -> JSONL)

位于 `transfer_data/xlsx_to_jsonl.py`。
用于将 XLSX 表格转换为 JSONL 格式。
**使用方法**：
修改脚本内的 `input_path` 和 `output_path` 参数，然后运行：

```bash
uv run python transfer_data/xlsx_to_jsonl.py
```

### 2. 文件筛选与复制工具

位于 `some_functions/select_data/copy_selected_files.py`。
用于根据总表（Excel 或 JSONL）中的文件路径信息，将源文件复制到目标目录。支持多种分隔符（换行、分号、空格等）。
**使用方法**：
修改脚本内的参数（如 `master_table_path`, `output_base_dir` 等），然后运行：

```bash
uv run python some_functions/select_data/copy_selected_files.py
```

### 3. 并行推理框架

位于 `infer/` 目录。

- `framework.py`: 通用并行推理框架，支持多线程并发与断点续跑（自动跳过已处理 ID）。
- `run_chart.py`: 针对 `chart_mini.jsonl` 的推理脚本示例，集成了代码执行工具（CodeExecutionToolkit）。

**使用方法**：
支持通过命令行参数传入 OpenAI 兼容的 API 配置。

```bash
uv run python infer/run_chart.py \
  --api_key YOUR_API_KEY \
  --base_url YOUR_BASE_URL \
  --model_name YOUR_MODEL_NAME \
  --num_workers 8
```

例如：

```bash
uv run python infer/run_chart.py \
  --api_key sk-xxxx \
  --base_url https://api.deepseek.com \
  --model_name deepseek-chat \
  --num_workers 8
```

结果将保存在 `preds/{model_name}/chart_mini/conv/{id}.json`。
