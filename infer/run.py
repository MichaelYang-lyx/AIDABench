import argparse
import subprocess
import sys
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path so we can import 'infer' as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from infer.failure_policy import is_retryable_api_failure

def get_sys_msg(sys_msg_path, task):
    p = Path(sys_msg_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"sys_msg_path 文件不存在: {p}")
    sys_msg = p.read_text(encoding="utf-8")
    sys_msg = sys_msg.replace('{task_prompt}', task)
    return sys_msg

def check_and_clean_failed_preds(output_dir):
    """
    Archive API/service failures so they can run again. Keep terminal model
    outcomes as checkpoints and restore legacy terminal traces for evaluation.
    """
    if not os.path.exists(output_dir):
        return

    # Derive Eval Directory
    # Convention: .../output/preds/{model}/{dataset}/conv -> .../output/evals/{model}/{dataset}
    eval_dir = None
    if "/preds/" in output_dir and output_dir.endswith("/conv"):
        eval_dir = output_dir.replace("/preds/", "/evals/").replace("/conv", "")
    
    print(f"Checking for failed predictions in {output_dir}...")
    if eval_dir and os.path.exists(eval_dir):
        print(f"Also checking corresponding eval files in {eval_dir}...")

    # Only transport/service failures should lose their checkpoint. Model-side
    # failures (including round limits and empty final answers) are evaluated.
    files_to_delete_conv = set()
    files_to_delete_eval = set()

    # 1. Scan Conv Files (Inference)
    try:
        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(output_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    model_response = str(data.get("model_response", ""))
                    has_error = is_retryable_api_failure(model_response)
                    if has_error:
                        print(f"Found retryable inference (API/service error): {filename}")
                        files_to_delete_conv.add(filename)
                        files_to_delete_eval.add(filename)
                except Exception:
                    pass
    except OSError:
        pass

    # 2. Scan Eval Files (Evaluation)
    if eval_dir and os.path.exists(eval_dir):
        try:
            for filename in os.listdir(eval_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(eval_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Gather fields to check
                        fields_to_check = []
                        fields_to_check.append(str(data.get("reason", "")))
                        fields_to_check.append(str(data.get("eval_reason", "")))
                        
                        # Handle chart correctness (dict or str)
                        corr = data.get("correctness")
                        if isinstance(corr, dict):
                            fields_to_check.append(str(corr.get("reason", "")))
                        else:
                            fields_to_check.append(str(corr))

                        # Handle chart visual (dict)
                        vis = data.get("visual")
                        if isinstance(vis, dict):
                            fields_to_check.append(str(vis.get("reason", "")))


                        if any(is_retryable_api_failure(field) for field in fields_to_check):
                            print(f"Found failed eval (Error in reason/correctness): {filename}")
                            files_to_delete_eval.add(filename)
                            # If eval failed, we only delete eval file to let it re-run.
                            # Inference file might be correct, so we keep it.
                            # files_to_delete_conv.add(filename) 
                    except Exception:
                        pass
        except OSError:
            pass

    # 3. Archive failed inference traces, then remove their checkpoints so
    # they can resume on the next inference run. A failed conv JSON must not
    # remain in place because its presence marks the task as processed.
    failed_trace_dir = os.path.join(os.path.dirname(output_dir), "failed_traces")

    count_conv = 0
    for fname in files_to_delete_conv:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                record['_failure_archived_at'] = datetime.now(timezone.utc).isoformat()
                os.makedirs(failed_trace_dir, exist_ok=True)
                trace_path = os.path.join(failed_trace_dir, f"{fname[:-5]}.jsonl")
                with open(trace_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Archived failed inference trace: {trace_path}")

                os.remove(p)
                count_conv += 1
                print(f"Deleted inference file: {p}")
                # Also delete corresponding workspace directory
                task_id = fname[:-5]  # strip .json
                workspace_dir = os.path.join(os.path.dirname(output_dir), 'workspace', task_id)
                if os.path.isdir(workspace_dir):
                    import shutil
                    shutil.rmtree(workspace_dir)
                    print(f"Deleted workspace: {workspace_dir}")
            except OSError as e:
                print(f"Error deleting {p}: {e}")
    
    count_eval = 0
    if eval_dir and os.path.exists(eval_dir):
        for fname in files_to_delete_eval:
            p = os.path.join(eval_dir, fname)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    count_eval += 1
                    print(f"Deleted eval file: {p}")
                except OSError as e:
                    print(f"Error deleting {p}: {e}")

    if count_conv > 0 or count_eval > 0:
        print(f"Cleanup finished. Removed {count_conv} inference files and {count_eval} eval files.")

    _restore_terminal_results(output_dir, eval_dir, failed_trace_dir)


def _restore_terminal_results(output_dir, eval_dir, failed_trace_dir):
    """Restore only the latest terminal attempt from legacy retry archives."""
    candidates = {}

    def consider(item_id, record, happened_at, artifact_root=None):
        if not isinstance(record, dict) or str(record.get("id")) != item_id:
            return
        if item_id not in candidates or happened_at > candidates[item_id][0]:
            candidates[item_id] = (happened_at, record, artifact_root)

    if os.path.isdir(failed_trace_dir):
        for filename in os.listdir(failed_trace_dir):
            if not filename.endswith(".jsonl"):
                continue
            trace_path = os.path.join(failed_trace_dir, filename)
            try:
                last_record = None
                with open(trace_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            last_record = json.loads(line)
                timestamp = (last_record or {}).get("_failure_recorded_at") or (last_record or {}).get("_failure_archived_at")
                happened_at = datetime.fromisoformat(timestamp) if timestamp else datetime.fromtimestamp(os.path.getmtime(trace_path), timezone.utc)
                consider(filename[:-6], last_record, happened_at)
            except (OSError, ValueError, TypeError, AttributeError) as e:
                print(f"Could not read failed trace {trace_path}: {e}")

    conv_path = Path(output_dir).resolve()
    if conv_path.name == "conv" and conv_path.parents[2].name == "preds":
        quarantine_root = conv_path.parents[2].parent / "retry_quarantine"
        model_name = conv_path.parents[1].name
        dataset_name = conv_path.parent.name
        if quarantine_root.is_dir():
            for stamp_dir in quarantine_root.iterdir():
                archived_conv = stamp_dir / model_name / dataset_name / "conv"
                if not archived_conv.is_dir():
                    continue
                try:
                    happened_at = datetime.strptime(stamp_dir.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                for archived_path in archived_conv.glob("*.json"):
                    try:
                        with open(archived_path, encoding="utf-8") as f:
                            record = json.load(f)
                        consider(archived_path.stem, record, happened_at, archived_conv.parent / "artifacts")
                    except (OSError, ValueError, TypeError) as e:
                        print(f"Could not read quarantined result {archived_path}: {e}")

    for item_id, (_, record, artifact_root) in candidates.items():
        checkpoint_path = conv_path / f"{item_id}.json"
        if checkpoint_path.exists():
            continue
        response = str(record.get("model_response", "")).strip()
        if response not in (
            "Error: Too many rounds reached.",
            "Error: Model returned no usable final response after recovery attempts.",
        ):
            continue
        restored = dict(record)
        restored.pop("_retryable_failure", None)
        restored.pop("_failure_recorded_at", None)
        restored.pop("_failure_archived_at", None)
        try:
            if artifact_root:
                for kind in ("pictures", "generated_files"):
                    source = artifact_root / kind / item_id
                    target = conv_path.parent / kind / item_id
                    if source.is_dir() and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(target))
            if eval_dir:
                eval_path = os.path.join(eval_dir, f"{item_id}.json")
                if os.path.exists(eval_path):
                    os.remove(eval_path)
                    print(f"Removed stale evaluation cache: {eval_path}")
            with open(checkpoint_path, "x", encoding="utf-8") as f:
                json.dump(restored, f, ensure_ascii=False, indent=2)
            print(f"Restored terminal model result for evaluation: {checkpoint_path}")
        except (OSError, ValueError, TypeError) as e:
            print(f"Could not restore terminal result for {item_id}: {e}")

def _load_params_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Pre-parse to detect --params before the main parser runs,
    # so JSON values can fill in required arguments like --dataset/--api_key/--model_name.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--params", default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    json_defaults = {}
    if pre_args.params:
        json_defaults = _load_params_json(pre_args.params)

    parser = argparse.ArgumentParser(description="Unified Entry Point for OfficeBench Inference")

    # Common arguments for all tasks
    parser.add_argument("--params", default=None, help="Path to a JSON config file. Values are used as defaults and can be overridden by explicit CLI flags.")
    parser.add_argument("--dataset", default=json_defaults.get("dataset"), required="dataset" not in json_defaults, help="Dataset name (e.g., chart, chart_mini)")
    parser.add_argument("--api_key", default=json_defaults.get("api_key", ""), required="api_key" not in json_defaults, help="API Key")
    parser.add_argument("--base_url", default=json_defaults.get("base_url", ""), help="OpenAI Base URL (not needed for proxy_jupyter_agent)")
    parser.add_argument("--model_name", default=json_defaults.get("model_name"), required="model_name" not in json_defaults, help="Model Name to use")
    parser.add_argument("--save_name", default=json_defaults.get("save_name"), help="Name to use for saving results (default: model_name)")
    parser.add_argument("--num_workers", type=int, default=json_defaults.get("num_workers", 4), help="Number of parallel workers")
    parser.add_argument("--data_root", default=json_defaults.get("data_root", os.path.join(os.getcwd(), "data")), help="Root directory for data files")
    parser.add_argument("--output_path", default=json_defaults.get("output_path"), help="Optional output path")
    parser.add_argument("--data_path", default=json_defaults.get("data_path"), help="Optional specific data path")
    parser.add_argument("--prompt_file", default=json_defaults.get("prompt_file"), help="Name of the prompt file in infer/prompts/ or absolute path")
    parser.add_argument("--need_info", action="store_true", default=json_defaults.get("need_info", False), help="Enable file info enhancement (default: False)")
    parser.add_argument("--agent_type", default=json_defaults.get("agent_type", "openai_jupyter_agent"), help="Agent type to use.")
    parser.add_argument("--max_rounds", type=int, default=json_defaults.get("max_rounds", 20), help="Maximum number of rounds for the agent (default: 20)")
    parser.add_argument("--channel_code", default=json_defaults.get("channel_code", "ali"), help="Channel code for proxy agent (default: ali)")
    parser.add_argument("--transaction_id", default=json_defaults.get("transaction_id", "proxy_task"), help="Transaction ID for proxy agent")
    parser.add_argument("--enable_thinking", default=json_defaults.get("enable_thinking", None), help="Thinking mode: 'think' to enable thinking, 'nothink' to suppress thinking, omit for default behavior")
    parser.add_argument("--reasoning_effort", default=json_defaults.get("reasoning_effort", None), help="Reasoning effort passed through chat_template_kwargs (for example: high or max)")
    parser.add_argument("--skills_dir", default=json_defaults.get("skills_dir"), help="Skills directory for skill_jupyter_agent (default: skills/)")
    parser.add_argument("--temperature", type=float, default=json_defaults.get("temperature", 0.0), help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=json_defaults.get("top_p", 1.0), help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--raccoon_project_uuid", default=json_defaults.get("raccoon_project_uuid", ""), help="Raccoon project UUID (xhx_pipeline only)")
    parser.add_argument("--enable_web_search", action="store_true", default=json_defaults.get("enable_web_search", False), help="Enable web search in Raccoon (xhx_pipeline only)")
    parser.add_argument("--deep_think", action="store_true", default=json_defaults.get("deep_think", False), help="Enable deep think mode in Raccoon (xhx_pipeline only)")
    parser.add_argument("--provider", default=json_defaults.get("provider"), help="Model provider for hermes (e.g., anthropic, openai, google, custom)")

    # Capture all arguments
    args = parser.parse_args()

    # Attach helper function to args so downstream runners can use it
    args.get_sys_msg_func = get_sys_msg
    # Get the directory where this script (infer/run.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine which datasets to run (preserve original casing for paths)
    datasets_to_run = []
    if args.dataset.lower() == 'all':
        datasets_to_run = ['data_visualization', 'QA', 'file_generation']
    else:
        datasets_to_run = [args.dataset]

    for ds_name in datasets_to_run:
        ds_lower = ds_name.lower()
        # Create a new args object for this dataset to avoid modifying the original
        current_args = argparse.Namespace(**vars(args))
        current_args.dataset = ds_name
        
        # Print Banner
        print("\n" + "="*40)
        print(f" mode: infer    dataset: {ds_name}")
        print("="*40 + "\n")

        print(f"\n>>> Starting task for dataset: {ds_name}")

        # Pre-processing: Check and clean failed predictions
        # Construct the expected output path to check for existing failed results
        if current_args.output_path:
            output_path_to_check = os.path.abspath(current_args.output_path)
        else:
            save_name = getattr(current_args, 'save_name', None) or current_args.model_name
            output_path_to_check = os.path.abspath(os.path.join("output", "preds", save_name, ds_name, "conv"))
        
        check_and_clean_failed_preds(output_path_to_check)

        # Dispatch Logic
        if "data_visualization" in ds_lower or "chart" in ds_lower:
            # Matches data_visualization, chart, chart_mini, chart_test, etc.
            try:
                from infer.runner.run_data_visualization import run as run_chart_task
                print(f"Dispatching to infer.runner.run_data_visualization for dataset '{ds_name}'...")
                run_chart_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_data_visualization: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing task: {e}")
                sys.exit(1)
                
        elif "qa" in ds_lower or "numeric" in ds_lower or "numerical" in ds_lower or "wps" in ds_lower:
            try:
                from infer.runner.run_QA import run as run_numeric_task
                print(f"Dispatching to infer.runner.run_QA for dataset '{ds_name}'...")
                run_numeric_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_QA: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing task: {e}")
                sys.exit(1)

        elif any(kw in ds_lower for kw in ["generation", "ppt", "doc", "excel"]):
            try:
                from infer.runner.run_file_generation import run as run_file_generation_task
                print(f"Dispatching to infer.runner.run_file_generation for dataset '{ds_name}'...")
                run_file_generation_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_file_generation: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing task: {e}")
                sys.exit(1)

        elif "open_ended" in ds_lower or "open" in ds_lower:
            try:
                from infer.runner.run_open_ended import run as run_open_ended_task
                print(f"Dispatching to infer.runner.run_open_ended for dataset '{ds_name}'...")
                run_open_ended_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_open_ended: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing task: {e}")
                sys.exit(1)

        else:
            print(f"Dataset '{ds_name}' is not currently supported by this runner.")
            print("Supported datasets: data_visualization, QA, generation, ppt, doc, excel, numerical")
            sys.exit(1)

if __name__ == "__main__":
    main()
