import os
import sys
import argparse
import json
import concurrent.futures
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluators.numerical_evaluator import NumericalEvaluator
from evaluation.utils import load_dataset

def process_single_row(row, i, args, evaluator):
    # Remove large fields to save space/memory if needed
    row.pop('history', None)
    
    # Ensure answer field is present (map model_response to answer if needed)
    if 'answer' not in row and 'model_response' in row:
        row['answer'] = row['model_response']
    
    # Evaluate
    try:
        eval_res = evaluator.evaluate_single(row)
    except Exception as e:
        eval_res = {'score': 0, 'reason': f"Exception: {str(e)}"}
    
    # Update row with result
    row['score'] = eval_res.get('score', 0)
    row['reason'] = eval_res.get('reason', '')
    
    # Save per-item result
    save_result(row, args, i)
    return row

def get_result_file_path(row, i, args):
    item_id = row.get('id', f"eval_{i}")
    file_name = f"{item_id}.json"
    return os.path.join(args.output_path, file_name)

def save_result(row, args, i):
    file_path = get_result_file_path(row, i, args)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

def run(args):
    """
    Run numeric evaluation.
    args should have: input_path, output_path
    """
    # Load data
    print(f"Loading inference results from {args.input_path}...")
    data = load_dataset(args.input_path)
    
    if not data:
        print("No data found in input file.")
        sys.exit(1)

    evaluator = NumericalEvaluator()
    
    results = []
    total_score = 0
    total_items = 0
    
    print(f"Starting evaluation...")
    print(f"Saving results to {args.output_path}...")
    os.makedirs(os.path.abspath(args.output_path), exist_ok=True)

    # Use ThreadPoolExecutor for concurrency
    max_workers = args.max_workers
    
    # Track futures and processed items
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, row in enumerate(data):
            # Check if result already exists (Resume capability)
            result_path = get_result_file_path(row, i, args)
            if os.path.exists(result_path):
                # Load existing result
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                    
                    results.append(existing_result)
                    try:
                        score = float(existing_result.get('score', 0))
                        total_score += score
                        total_items += 1
                    except:
                        pass
                    continue # Skip re-evaluation
                except Exception as e:
                    print(f"Error loading existing result for item {i}, re-evaluating: {e}")
            
            # If not exists or error loading, submit for evaluation
            futures.append(executor.submit(process_single_row, row, i, args, evaluator))
        
        # Process futures as they complete
        if futures:
            print(f"Resuming/Running {len(futures)} items...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    row = future.result()
                    results.append(row)
                    
                    try:
                        score = float(row.get('score', 0))
                        total_score += score
                        total_items += 1
                    except:
                        pass
                except Exception as e:
                    print(f"Error processing item: {e}")
        else:
            print("All items already evaluated.")

    # Also save a summary file
    summary_path = os.path.join(args.output_path, "summary.json")
    
    dataset_size = len(data)
    avg_score = (total_score / dataset_size) if dataset_size > 0 else 0
    
    summary = {
        "total_items": dataset_size,
        "total_score": total_score,
        "score": round(avg_score, 4)
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluation Complete.")
    print(f"Total Items: {dataset_size}")
    print(f"Total Score: {total_score}")
    print(f"Score: {avg_score:.4f}")

if __name__ == "__main__":
    pass
