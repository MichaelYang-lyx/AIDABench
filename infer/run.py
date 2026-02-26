import argparse
import subprocess
import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path so we can import 'infer' as a package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_sys_msg(sys_msg_path, task):
    p = Path(sys_msg_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"sys_msg_path 文件不存在: {p}")
    sys_msg = p.read_text(encoding="utf-8")
    sys_msg = sys_msg.replace('{task_prompt}', task)
    return sys_msg

def check_and_clean_failed_preds(output_dir):
    """
    Check all json files in output_dir (inference results).
    If 'model_response' contains '502 Bad Gateway' or 'Error code: 503', delete the file.
    Also check corresponding evaluation files and delete them if inference is bad.
    Additionally, check evaluation files for the same errors in 'reason'/'correctness',
    and if found, delete both eval and inference files.
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

    error_patterns = ["Request timed out","ClaudeSubprocessAgent","claude_subprocess_agent.py","Error code: 429","502 Bad Gateway", "Error code: 503", "engine is currently overloaded"]
    
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
                    if any(err in model_response for err in error_patterns):
                        print(f"Found failed inference (Error in model_response): {filename}")
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


                        if any(any(err in field for err in error_patterns) for field in fields_to_check):
                            print(f"Found failed eval (Error in reason/correctness): {filename}")
                            files_to_delete_eval.add(filename)
                            # If eval failed, we only delete eval file to let it re-run.
                            # Inference file might be correct, so we keep it.
                            # files_to_delete_conv.add(filename) 
                    except Exception:
                        pass
        except OSError:
            pass

    # 3. Perform Deletion
    count_conv = 0
    for fname in files_to_delete_conv:
        p = os.path.join(output_dir, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
                count_conv += 1
                print(f"Deleted inference file: {p}")
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

def main():
    parser = argparse.ArgumentParser(description="Unified Entry Point for OfficeBench Inference")
    
    # Common arguments for all tasks
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., chart, chart_mini)")
    parser.add_argument("--api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", required=True, help="OpenAI Base URL")
    parser.add_argument("--model_name", required=True, help="Model Name to use")
    parser.add_argument("--save_name", help="Name to use for saving results (default: model_name)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--data_root", default=os.path.join(os.getcwd(), "data"), help="Root directory for data files")
    parser.add_argument("--output_path", help="Optional output path")
    parser.add_argument("--data_path", help="Optional specific data path")
    parser.add_argument("--prompt_file", help="Name of the prompt file in infer/prompts/ or absolute path")
    parser.add_argument("--need_info", action="store_true", help="Enable file info enhancement (default: False)")
    parser.add_argument("--agent_type", default="openai_jupyter_agent", help="Agent type to use. Choices: openai_jupyter_agent, openai_subprocess_agent, claude_jupyter_agent, claude_subprocess_agent. (default: openai_jupyter_agent)")
    parser.add_argument("--max_rounds", type=int, default=20, help="Maximum number of rounds for the agent (default: 20)")

    # Capture all arguments
    args = parser.parse_args()
    
    # Attach helper function to args so downstream runners can use it
    args.get_sys_msg_func = get_sys_msg
    dataset = args.dataset.lower()
    
    # Get the directory where this script (infer/run.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine which datasets to run
    datasets_to_run = []
    if dataset == 'all':
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
                
        elif "qa" in ds_lower or "numeric" in ds_lower or "numerical" in ds_lower:
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

        else:
            print(f"Dataset '{ds_name}' is not currently supported by this runner.")
            print("Supported datasets: data_visualization, QA, generation, ppt, doc, excel, numerical")
            sys.exit(1)

if __name__ == "__main__":
    main()
