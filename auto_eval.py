import argparse
import os
import sys

# Ensure the current directory is in the path so we can import the evaluation package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.config import DATASET_MAPPING
from evaluation.evaluators import NumericalEvaluator, ChartEvaluator, FileGenerationEvaluator

def main():
    parser = argparse.ArgumentParser(description="Auto Evaluation Script")
    parser.add_argument("--dataset", required=True, 
                        choices=["all", "all_mini"] + list(DATASET_MAPPING.keys()),
                        help="Select the evaluation dataset")
    parser.add_argument("--model_name", required=True,
                        help="Name of the model being evaluated (used as subdirectory in preds/)")
    args = parser.parse_args()
    
    datasets_to_run = []
    if args.dataset == "all":
        datasets_to_run = ["chart", "numerical", "file_generation"]
    elif args.dataset == "all_mini":
        datasets_to_run = ["chart_mini", "numerical_mini", "file_generation_mini"]
    else:
        datasets_to_run = [args.dataset]
        
    for ds_name in datasets_to_run:
        data_path = DATASET_MAPPING[ds_name]
        
        # Determine pred_path based on dataset type
        if "chart" in ds_name:
            # For chart, pred_path is the directory containing images
            pred_path = os.path.join("preds", args.model_name, "chart")
        elif "file_generation" in ds_name:
            # For file_generation, pred_path is the specific excel file
            # Assuming the file is named 'file_generation_pred.xlsx'
            pred_path = os.path.join("preds", args.model_name, "file_generation", "file_generation_pred.xlsx")
        else:
            # For numerical, pred_path mimics the data structure (json file)
            rel_path = os.path.relpath(data_path, "data")
            pred_path = os.path.join("preds", args.model_name, rel_path)
        
        print(f"\n=== Running Evaluation for {ds_name} ===")
        print(f"Data Path: {data_path}")
        print(f"Pred Path: {pred_path}")
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}. Skipping.")
            continue
            
        if not os.path.exists(pred_path):
            print(f"Warning: Pred path not found: {pred_path}. Skipping.")
            continue
            
        evaluator = None
        if "chart" in ds_name:
            evaluator = ChartEvaluator()
        elif "numerical" in ds_name:
            evaluator = NumericalEvaluator()
        elif "file_generation" in ds_name:
            evaluator = FileGenerationEvaluator()
        else:
            print(f"Unknown dataset type: {ds_name}")
            continue

        if evaluator:
            evaluator.evaluate_dataset(data_path, pred_path)

if __name__ == "__main__":
    main()
