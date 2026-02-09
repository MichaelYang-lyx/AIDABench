import argparse
import sys
import os

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description="Unified Entry Point for OfficeBench Evaluation")
    
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., chart, chart_mini)")
    parser.add_argument("--model_name", required=True, help="Model Name used for inference")
    parser.add_argument("--input_path", help="Path to inference results (default: preds/{model}/{dataset}/conv)")
    parser.add_argument("--picture_dir", help="Directory with generated pictures (default: preds/{model}/{dataset}/pictures)")
    parser.add_argument("--generated_files_dir", help="Directory with generated files (default: preds/{model}/{dataset}/generated_files)")
    parser.add_argument("--output_path", help="Path to save eval results (default: eval_results/{model}/{dataset}/result.jsonl)")
    parser.add_argument("--max_workers", type=int, default=10, help="Max workers for concurrency (default: 10)")
    parser.add_argument("--data_root", help="Root directory for datasets (to find reference files)")
    parser.add_argument("--api_key", help="API Key for Evaluator Agent")
    parser.add_argument("--base_url", help="Base URL for Evaluator Agent")
    parser.add_argument("--evaluator_model", help="Model name for the evaluator agent (default: same as model_name)")
    
    args = parser.parse_args()
    
    dataset = args.dataset.lower()
    
    # Construct Defaults
    # Note: project_root might be where we want to base relative paths
    # but os.path.join("preds", ...) is relative to CWD. 
    # infer/run.py uses CWD for defaults implicitly or explicit joins.
    # Let's use absolute paths based on project_root to be safe, or just CWD if that's the convention.
    # infer/run.py defaults: os.path.join("preds", ...) -> relative to where you run it.
    # But here I will make them absolute based on project_root to avoid CWD ambiguity if run from subdir.
    
    if not args.input_path:
        args.input_path = os.path.join(project_root, "output", "preds", args.model_name, dataset, "conv")
        
    if not args.picture_dir:
        args.picture_dir = os.path.join(project_root, "output", "preds", args.model_name, dataset, "pictures")

    if not args.generated_files_dir:
        args.generated_files_dir = os.path.join(project_root, "output", "preds", args.model_name, dataset, "generated_files")
        
    if not args.output_path:
        args.output_path = os.path.join(project_root, "output", "evals", args.model_name, dataset)

    if not args.data_root:
        # Default to project_root
        args.data_root = project_root

    print(f"Evaluation Configuration:")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {args.model_name}")
    print(f"  Input Path: {args.input_path}")
    print(f"  Picture Dir: {args.picture_dir}")
    print(f"  Generated Files Dir: {args.generated_files_dir}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Data Root: {args.data_root}")
    
    # Check if input exists
    if not os.path.exists(args.input_path):
        print(f"Warning: Input path does not exist: {args.input_path}")
        # We proceed, maybe load_dataset will fail cleanly or user made a mistake
        
    # Dispatch
    if "chart" in dataset:
        try:
            from evaluation.runner.eval_chart import run as run_chart_eval
            run_chart_eval(args)
        except ImportError as e:
            print(f"Error importing evaluation.runner.eval_chart: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error executing evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif "file" in dataset:
        try:
            from evaluation.runner.eval_file_generation import run as run_file_eval
            run_file_eval(args)
        except ImportError as e:
            print(f"Error importing evaluation.runner.eval_file_generation: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error executing evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif "numeric" in dataset:
        try:
            from evaluation.runner.eval_numeric import run as run_numeric_eval
            run_numeric_eval(args)
        except ImportError as e:
            print(f"Error importing evaluation.runner.eval_numeric: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error executing evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Dataset '{dataset}' not supported for evaluation yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
