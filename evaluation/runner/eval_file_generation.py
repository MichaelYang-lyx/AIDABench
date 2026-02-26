import os
import sys
import argparse
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluators.file_evaluator_agent import FileEvaluatorAgent
from evaluation.utils import load_dataset
from evaluation.config import FILE_GENERATION_EVAL_API_KEY, FILE_GENERATION_EVAL_API_URL, FILE_GENERATION_EVAL_MODEL_NAME

def process_single_row(row, i, args, evaluator):
    
    # Remove large fields
    row.pop('history', None)
    # row.pop('model_response', None) # Might be useful for debugging
    
    output_path_str = row.get('output_file', '')
    if not output_path_str:
        row['eval_score'] = 0
        row['eval_reason'] = "No output_file in data"
        save_result(row, args, i)
        return row
    
    # Assuming first output file is the one to evaluate
    first_output_path = output_path_str.split('\n')[0].strip()
    if not first_output_path:
        row['eval_score'] = 0
        row['eval_reason'] = "Empty output_file"
        save_result(row, args, i)
        return row
    
    generated_file_path = first_output_path
    if not os.path.isabs(generated_file_path):
        # Assume it's in args.generated_files_dir if provided
        if hasattr(args, 'generated_files_dir') and args.generated_files_dir:
             task_id = row.get('id', 'unknown')
             # Note: run_file_generation.py saves to generated_files_dir/task_id/filename
             generated_file_path = os.path.join(args.generated_files_dir, str(task_id), os.path.basename(generated_file_path))
    
    if not os.path.exists(generated_file_path):
         row['eval_score'] = 0
         row['eval_reason'] = f"Generated file not found: {generated_file_path}"
         save_result(row, args, i)
         return row

    # Reference File
    
    reference_path = row.get('reference', row.get('ground_truth_files', row.get('reference_file', '')))
    if isinstance(reference_path, list):
        if reference_path:
            reference_path = reference_path[0]
        else:
            reference_path = ""
    elif isinstance(reference_path, str):
        if '\n' in reference_path:
            reference_path = reference_path.split('\n')[0].strip()
            
    if not reference_path:
        row['eval_score'] = 0
        row['eval_reason'] = "No reference file specified"
        save_result(row, args, i)
        return row
    
    # Check if reference file exists
    if not os.path.isabs(reference_path) and hasattr(args, 'data_root') and args.data_root:
        # Try multiple locations
        candidates = [
            os.path.join(args.data_root, reference_path),
            os.path.join(args.data_root, "data", "file_generation", args.dataset, "reference", str(row.get('id', '')), reference_path),
            os.path.join(args.data_root, "file_generation", args.dataset, "reference", str(row.get('id', '')), reference_path),
            os.path.join(args.data_root, "data", args.dataset, "reference", str(row.get('id', '')), reference_path),
        ]
        
        found = False
        for p in candidates:
            if os.path.exists(p):
                reference_path = p
                found = True
                break
        
        if not found:
            row['eval_score'] = 0
            row['eval_reason'] = f"Reference file not found. Tried: {candidates}"
            save_result(row, args, i)
            return row
    elif not os.path.exists(reference_path):
        row['eval_score'] = 0
        row['eval_reason'] = f"Reference file not found: {reference_path}"
        save_result(row, args, i)
        return row

    question = row.get('question', '')

    # Evaluate
    try:
        eval_res = evaluator.evaluate(
            question=question,
            reference_path=reference_path,
            prediction_path=generated_file_path
        )
    except Exception as e:
        eval_res = {'is_correct': False, 'reason': f"Exception: {str(e)}"}
    
    is_correct = eval_res.get('is_correct', False)
    reason = eval_res.get('reason', '')
    
    if "eval_history" in eval_res:
        row['eval_history'] = eval_res['eval_history']

    row['correctness'] = {
        'score': 1.0 if is_correct else 0.0,
        'reason': reason
    }
    row['eval_score'] = 1.0 if is_correct else 0.0
    row['eval_reason'] = reason
    
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
    # Load data
    data = load_dataset(args.input_path)
    
    # Initialize Evaluator
    # Use default values if not provided, or fallback to the provided values
    api_key = args.api_key or FILE_GENERATION_EVAL_API_KEY
    base_url = args.base_url or FILE_GENERATION_EVAL_API_URL
    evaluator_model = getattr(args, 'evaluator_model', None) or FILE_GENERATION_EVAL_MODEL_NAME
    
    evaluator = FileEvaluatorAgent(
        api_key=api_key,
        base_url=base_url,
        model_name=evaluator_model
    )
    
    results = []
    
    print(f"Starting evaluation...")
    print(f"Saving results to {args.output_path}...")
    os.makedirs(os.path.abspath(args.output_path), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, row in enumerate(data):
            # Check if result already exists (Resume capability)
            result_path = get_result_file_path(row, i, args)
            
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                    results.append(existing_result)
                    continue
                except Exception as e:
                    print(f"Error loading existing result for item {i}, re-evaluating: {e}")
            
            futures[executor.submit(process_single_row, row, i, args, evaluator)] = i
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing row: {e}")
                
    # Calculate summary stats
    total_score = 0
    total_items = 0
    for res in results:
        total_score += res.get('eval_score', 0)
        total_items += 1
        
    dataset_size = len(data) # Should use original dataset size or results size? Using data size is safer for completion rate.
    avg_score = (total_score / dataset_size) if dataset_size > 0 else 0
    
    summary_path = os.path.join(args.output_path, "summary.json")
    summary = {
        "total_items": dataset_size,
        "total_score": total_score,
        "score": round(avg_score, 4)
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluation finished. Results saved to {args.output_path}")
    print(f"Total Items: {dataset_size}")
    print(f"Total Score: {total_score}")
    print(f"Score: {avg_score:.4f}")
