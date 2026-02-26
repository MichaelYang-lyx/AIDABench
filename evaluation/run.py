import argparse
import sys
import os
import json

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

    # Pre-check and clean failed predictions/evaluations
    # This ensures that if we are re-running, we don't skip failed items from previous runs
    try:
        from infer.run import check_and_clean_failed_preds
        # args.input_path is the inference directory (e.g., .../conv)
        # This function will check inference files AND corresponding eval files
        if os.path.exists(args.input_path):
            print(f"Running pre-evaluation cleanup on {args.input_path}...")
            check_and_clean_failed_preds(args.input_path)
    except ImportError:
        print("Warning: Could not import check_and_clean_failed_preds from infer.run. Skipping cleanup.")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    # Print Banner
    print("\n" + "="*40)
    print(f" mode: eval     dataset: {dataset}")
    print("="*40 + "\n")

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
    if "data_visualization" in dataset or "chart" in dataset:
        try:
            from evaluation.runner.eval_data_visualization import run as run_chart_eval
            run_chart_eval(args)
        except ImportError as e:
            print(f"Error importing evaluation.runner.eval_data_visualization: {e}")
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
    elif "qa" in dataset or "numeric" in dataset:
        try:
            from evaluation.runner.eval_QA import run as run_numeric_eval
            run_numeric_eval(args)
        except ImportError as e:
            print(f"Error importing evaluation.runner.eval_QA: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error executing evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Dataset '{dataset}' not supported for evaluation yet.")
        sys.exit(1)

    # Central Summary Update
    summary_file_path = os.path.join(args.output_path, "summary.json")
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                local_summary = json.load(f)
            
            central_summary_path = os.path.join(project_root, "output", "evals", "summary.json")
            os.makedirs(os.path.dirname(central_summary_path), exist_ok=True)
            
            central_data = {}
            if os.path.exists(central_summary_path):
                try:
                    with open(central_summary_path, 'r', encoding='utf-8') as f:
                        central_data = json.load(f)
                except:
                    pass
            
            if args.model_name not in central_data:
                central_data[args.model_name] = {}
                
            model_entry = central_data[args.model_name]
            
            if "data_visualization" in dataset or "chart" in dataset:
                if 'correctness_score' in local_summary:
                    model_entry['data_visualization_correctness_score'] = round(float(local_summary['correctness_score']), 4)
                if 'visual_score' in local_summary:
                    model_entry['data_visualization_readability_score'] = round(float(local_summary['visual_score']), 4)
                
            elif "file" in dataset: # file_generation
                if 'score' in local_summary:
                    model_entry['file_generation_score'] = round(float(local_summary['score']), 4)
                    
            elif "qa" in dataset or "numeric" in dataset:
                if 'score' in local_summary:
                    model_entry['QA_score'] = round(float(local_summary['score']), 4)
            
            # Reorder keys in model_entry
            ordered_keys = [
                "QA_score",
                "data_visualization_correctness_score",
                "data_visualization_readability_score",
                "file_generation_score"
            ]
            
            new_model_entry = {}
            # Add known keys in order
            for key in ordered_keys:
                if key in model_entry:
                    new_model_entry[key] = model_entry[key]
            
            # Add any other keys that might exist
            for key in model_entry:
                if key not in ordered_keys:
                    new_model_entry[key] = model_entry[key]
            
            central_data[args.model_name] = new_model_entry

            # Save back
            with open(central_summary_path, 'w', encoding='utf-8') as f:
                json.dump(central_data, f, ensure_ascii=False, indent=2)
                
            print(f"Updated central summary at {central_summary_path}")
            
        except Exception as e:
            print(f"Error updating central summary: {e}")

if __name__ == "__main__":
    main()
