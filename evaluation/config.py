import os
from dotenv import load_dotenv

load_dotenv()

def _get(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env: {name}")
    return v

CHART_EVAL_API_URL = _get("CHART_EVAL_API_URL")
CHART_EVAL_API_KEY = _get("CHART_EVAL_API_KEY")
CHART_EVAL_MODEL_NAME = _get("CHART_EVAL_MODEL_NAME")

NUMERICAL_EVAL_API_URL = _get("NUMERICAL_EVAL_API_URL")
NUMERICAL_EVAL_API_KEY = _get("NUMERICAL_EVAL_API_KEY")
NUMERICAL_EVAL_MODEL_NAME = _get("NUMERICAL_EVAL_MODEL_NAME")

FILE_GENERATION_EVAL_API_URL = _get("FILE_GENERATION_EVAL_API_URL")
FILE_GENERATION_EVAL_API_KEY = _get("FILE_GENERATION_EVAL_API_KEY")
FILE_GENERATION_EVAL_MODEL_NAME = _get("FILE_GENERATION_EVAL_MODEL_NAME")

DATASET_MAPPING = {
    "chart": "data/chart/chart.json",
    "chart_mini": "data/chart/chart_mini.json",
    "numerical": "data/numerical/numerical.json",
    "numerical_mini": "data/numerical/numerical_mini.json",
    "file_generation": "data/file_generation/file_generation.json",
    "file_generation_mini": "data/file_generation/file_generation_mini.json",
}
