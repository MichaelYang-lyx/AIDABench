#!/usr/bin/env python3
"""Convert wps_2_0_open_ended.jsonl entries to test_tasks.json (open_ended format).

Unlike convert_qa_to_open_ended.py, this does NOT copy input files: WPS 2.0 already
keeps each task's files in data/wps_2_0_open_ended/all_inputs/{n}/, so we just point
`dataset_csv_path` directly at them.
"""
import json
import os

PROJECT_ROOT = "/data/projects/AIDABench"
DATASET_DIR = os.path.join(PROJECT_ROOT, "data/wps_2_0_open_ended")
SRC_JSONL = os.path.join(DATASET_DIR, "wps_2_0_open_ended.jsonl")
INPUT_ROOT = os.path.join(DATASET_DIR, "all_inputs")
TASK_CONFIG = os.path.join(DATASET_DIR, "test_tasks.json")

DATASET_REL = "data/wps_2_0_open_ended"


def main():
    src_tasks = []
    with open(SRC_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                src_tasks.append(json.loads(line))
    print(f"Loaded {len(src_tasks)} tasks from {SRC_JSONL}")

    out_tasks = []
    missing = 0
    for item in src_tasks:
        task_id = item["id"]
        question = item.get("question", "")
        input_file = (item.get("input_file") or "").strip()

        dataset_csv_path = ""
        input_basename = ""
        if input_file:
            full_path = os.path.join(INPUT_ROOT, input_file)
            if os.path.exists(full_path):
                dataset_csv_path = f"{DATASET_REL}/all_inputs/{input_file}"
                input_basename = os.path.basename(input_file)
            else:
                missing += 1
                print(f"  WARNING: input file missing for {task_id}: {full_path}")

        metadata = {
            "goal": question[:200] if question else "",
            "role": "Data Analyst",
            "category": item.get("output_type", "Data Analysis"),
            "original_id": task_id,
            "input_files": [input_basename] if input_basename else [],
        }
        if item.get("reference") is not None:
            metadata["reference"] = item["reference"]
        if item.get("rubrics") is not None:
            metadata["rubrics"] = item["rubrics"]

        out_tasks.append({
            "task_id": task_id,
            "query": question,
            "dataset_csv_path": dataset_csv_path,
            "metadata": metadata,
        })

    with open(TASK_CONFIG, "w", encoding="utf-8") as f:
        json.dump(out_tasks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out_tasks)} tasks to {TASK_CONFIG}")
    if missing:
        print(f"Tasks with missing input files: {missing}")


if __name__ == "__main__":
    main()
