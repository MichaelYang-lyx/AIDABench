import os
import sys
import argparse
import json
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluators.chart_evaluator import ChartEvaluator
from evaluation.utils import load_dataset

def process_single_row(row, i, args, evaluator):
    # Remove large fields to save space/memory
    row.pop('history', None)
    row.pop('model_response', None)
    
    output_path_str = row.get('output_path') or row.get('output_file', '')
    if not output_path_str:
        # print(f"Skipping item {i}: No output_path defined.")
        row['eval_score'] = 0
        row['eval_reason'] = "No output_path in data"
        save_result(row, args, i)
        return row

    first_output_path = output_path_str.split('\n')[0].strip()
    if not first_output_path:
         # print(f"Skipping item {i}: Empty output_path.")
         row['eval_score'] = 0
         row['eval_reason'] = "Empty output_path"
         save_result(row, args, i)
         return row
         
    image_filename = os.path.basename(first_output_path)
    
    # Try to find the image in task subdirectory first (as run_chart.py saves it there)
    task_id = row.get('id', 'unknown')
    image_full_path = os.path.join(args.picture_dir, str(task_id), image_filename)
    
    if not os.path.exists(image_full_path):
        # Fallback to checking directly in picture_dir
        image_full_path_fallback = os.path.join(args.picture_dir, image_filename)
        if os.path.exists(image_full_path_fallback):
            image_full_path = image_full_path_fallback
    
    # Prepare row for evaluator
    eval_row = row.copy()
    eval_row['image_path'] = image_full_path
    
    # Evaluate
    # evaluate_single returns {'correct_rubrics': int, 'reason': str, ...}
    try:
        eval_res = evaluator.evaluate_single(eval_row)
        
    except Exception as e:
        eval_res = {'correct_rubrics': 0, 'reason': f"Exception: {str(e)}", 'total_rubrics': 3}
    
    correct_rubrics = eval_res.get('correct_rubrics', eval_res.get('score', 0))
    total_rubrics = eval_res.get('total_rubrics', 3)
    
    score = (correct_rubrics / total_rubrics) if total_rubrics > 0 else 0
    
    row['correctness'] = {
        'score': round(score, 2),
        'correct_rubrics': correct_rubrics,
        'total_rubrics': total_rubrics,
        'reason': eval_res.get('reason', '')
    }

    # Visual Evaluation
    try:
        visual_res = evaluator.evaluate_visual(eval_row)
    except Exception as e:
        visual_res = {'reason': f"Exception: {str(e)}"}
    
    row['visual'] = visual_res

    # Calculate visual score
    visual_keys = [
        "labels_and_titles",
        "layout_spacing",
        "color_accessibility",
        "axis_scaling",
        "chart_type_suitability",
        "font_and_legends",
        "annotation_readability",
        "visual_hierarchy_and_emphasis"
    ]
    
    total_visual_score = 0
    valid_visual_items = 0
    
    for key in visual_keys:
        if key in visual_res:
            try:
                # Ensure value is numeric (0 or 1)
                val = int(visual_res[key])
                total_visual_score += val
                valid_visual_items += 1
            except:
                pass
                
    if valid_visual_items > 0:
        row['visual']['score'] = round(total_visual_score / valid_visual_items, 2)
    else:
        row['visual']['score'] = 0
    
    # Calculate weighted total score
    correctness_score = row['correctness'].get('score', 0)
    visual_score = row['visual'].get('score', 0)
    score_val = round(visual_score * 0.3 + correctness_score * 0.7, 2)
    
    # Create ordered dict with score first
    final_row = {'score': score_val}
    for k, v in row.items():
        if k != 'score':
            final_row[k] = v
    
    save_result(final_row, args, i)
    return final_row

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
    Run chart evaluation.
    args should have: input_path, picture_dir, output_path
    """
    # Load data
    print(f"Loading inference results from {args.input_path}...")
    data = load_dataset(args.input_path)
    
    if not data:
        print("No data found in input file.")
        sys.exit(1)

    evaluator = ChartEvaluator()
    
    results = []
    total_score = 0
    total_correctness_score = 0
    total_visual_score = 0
    total_items = 0
    
    print(f"Starting evaluation...")
    print(f"Saving results to {args.output_path}...")
    os.makedirs(os.path.abspath(args.output_path), exist_ok=True)

    # Use ThreadPoolExecutor for concurrency
    # Max workers can be adjusted. 5-10 is usually good for API calls.
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
                    
                    # Update stats
                    results.append(existing_result)
                    try:
                        # Support old format (eval_score) and new format (correctness.score)
                        # Note: Now total_score sums up the normalized scores (0-1)
                        # Now total_score is based on the weighted total score 'score' field
                        
                        # Extract component scores
                        c_score = 0
                        v_score = 0
                        
                        if 'correctness' in existing_result:
                            if 'score' in existing_result['correctness']:
                                c_score = float(existing_result['correctness']['score'])
                            else:
                                ck = int(existing_result['correctness'].get('correct_rubrics', 0))
                                tk = int(existing_result['correctness'].get('total_rubrics', 3))
                                c_score = (ck / tk) if tk > 0 else 0
                        
                        if 'visual' in existing_result and 'score' in existing_result['visual']:
                            v_score = float(existing_result['visual']['score'])

                        if 'score' in existing_result:
                            score = float(existing_result['score'])
                        elif 'correctness' in existing_result:
                            # Backward compatibility if weighted score wasn't saved but parts were
                            # Use the pre-calculated normalized score if available
                            score = round(v_score * 0.3 + c_score * 0.7, 2)
                        else:
                            # Old format was just integer score (maybe needs normalization? Assuming old score was 0/1 binary or count)
                            # If old score was just "pass/fail" (0 or 1), it is already normalized.
                            # If it was a count, we might be mixing scales. Assuming old eval was binary or we accept inconsistency for old data.
                            score = float(existing_result.get('eval_score', 0))
                            # For old format, assume correctness is the score and visual is 0? Or just leave them 0.
                            c_score = score
                            
                        total_score += score
                        total_correctness_score += c_score
                        total_visual_score += v_score
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
                        # Extract component scores
                        c_score = 0
                        v_score = 0
                        
                        if 'correctness' in row:
                            if 'score' in row['correctness']:
                                c_score = float(row['correctness']['score'])
                            else:
                                ck = int(row['correctness'].get('correct_rubrics', 0))
                                tk = int(row['correctness'].get('total_rubrics', 3))
                                c_score = (ck / tk) if tk > 0 else 0
                        
                        if 'visual' in row and 'score' in row['visual']:
                            v_score = float(row['visual']['score'])

                        # Use the weighted total score 'score'
                        if 'score' in row:
                            score = float(row['score'])
                        elif 'correctness' in row:
                            score = round(v_score * 0.3 + c_score * 0.7, 2)
                        else:
                            score = float(row.get('eval_score', 0))
                            c_score = score
                            
                        total_score += score
                        total_correctness_score += c_score
                        total_visual_score += v_score
                        total_items += 1 
                    except:
                        pass
                except Exception as e:
                    print(f"Error processing item: {e}")
        else:
            print("All items already evaluated.")

    # Also save a summary file if needed, or just print stats
    summary_path = os.path.join(args.output_path, "summary.json")
    
    # Calculate score based on total items in dataset (treating failures/skips as 0)
    dataset_size = len(data)
    avg_score = (total_score / dataset_size) if dataset_size > 0 else 0
    avg_correctness = (total_correctness_score / dataset_size) if dataset_size > 0 else 0
    avg_visual = (total_visual_score / dataset_size) if dataset_size > 0 else 0
    
    summary = {
        "total_items": dataset_size,
        "total_score": total_score,
        "score": round(avg_score, 4),
        "correctness_score": round(avg_correctness, 4),
        "visual_score": round(avg_visual, 4)
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluation Complete.")
    print(f"Total Items: {dataset_size}")
    print(f"Total Score: {total_score}")
    print(f"Score: {avg_score:.4f}")
    print(f"Correctness Score: {avg_correctness:.4f}")
    print(f"Visual (Readability) Score: {avg_visual:.4f}")

if __name__ == "__main__":
    pass
