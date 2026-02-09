import argparse
import subprocess
import sys
import os
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
    parser.add_argument("--agent_type", default="openai_jupyter_agent", help="Agent type to use (default: openai_jupyter_agent)")

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
        datasets_to_run = ['chart', 'numeric', 'file_generation']
    else:
        datasets_to_run = [args.dataset]

    for ds_name in datasets_to_run:
        ds_lower = ds_name.lower()
        # Create a new args object for this dataset to avoid modifying the original
        current_args = argparse.Namespace(**vars(args))
        current_args.dataset = ds_name
        
        print(f"\n>>> Starting task for dataset: {ds_name}")

        # Dispatch Logic
        if "chart" in ds_lower:
            # Matches chart, chart_mini, chart_test, etc.
            try:
                from infer.runner.run_chart import run as run_chart_task
                print(f"Dispatching to infer.runner.run_chart for dataset '{ds_name}'...")
                run_chart_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_chart: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error executing task: {e}")
                sys.exit(1)
                
        elif "numeric" in ds_lower or "numerical" in ds_lower:
            try:
                from infer.runner.run_numeric import run as run_numeric_task
                print(f"Dispatching to infer.runner.run_numeric for dataset '{ds_name}'...")
                run_numeric_task(current_args)
            except ImportError as e:
                print(f"Error importing infer.runner.run_numeric: {e}")
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
            print("Supported datasets: chart, chart_mini, generation, ppt, doc, excel, numerical")
            sys.exit(1)

if __name__ == "__main__":
    main()
