import os
import sys
import json
import re
import csv
import time
import shutil
import concurrent.futures
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from openai import OpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from consensus.rubric_generator import RubricGenerator
from consensus.consensus_extractor import ConsensusExtractor
from consensus.llm_judge import LLMJudge
from evaluation.utils import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_json_response(content: str) -> Any:
    """Parse JSON from LLM response, handling markdown code blocks."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def load_reference_models_config(config_path: str = "configs/reference_models.json") -> Dict:
    """Load reference models configuration."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, 'r') as f:
        return json.load(f)


def load_ground_truth_insights(task_config_path: str = "data/open_ended_test/test_tasks.json") -> Dict[str, Dict]:
    """Load task config (queries, metadata, csv paths) for test tasks."""
    if not os.path.isabs(task_config_path):
        task_config_path = os.path.join(PROJECT_ROOT, task_config_path)

    with open(task_config_path, 'r') as f:
        tasks = json.load(f)

    ground_truth_map = {}
    for task in tasks:
        task_id = task.get('task_id')
        ground_truth_map[task_id] = {
            "query": task.get('query', ''),
            "metadata": task.get('metadata', {}),
            "dataset_csv_path": task.get('dataset_csv_path', '')
        }

    return ground_truth_map


def get_data_description(task_id: str, ground_truth_map: Dict) -> str:
    """Get data description from metadata and CSV header."""
    task_info = ground_truth_map.get(task_id, {})
    metadata = task_info.get('metadata', {})
    csv_path = task_info.get('dataset_csv_path', '')

    parts = []
    if metadata.get('dataset_description'):
        parts.append(f"Description: {metadata['dataset_description']}")
    if metadata.get('category'):
        parts.append(f"Category: {metadata['category']}")
    if metadata.get('goal'):
        parts.append(f"Goal: {metadata['goal']}")

    if csv_path:
        full_csv_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
        if os.path.exists(full_csv_path):
            try:
                with open(full_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= 6:
                            break
                        rows.append(row)
                if rows:
                    parts.append(f"Columns: {', '.join(rows[0])}")
                    if len(rows) > 1:
                        parts.append(f"Sample rows:\n" + "\n".join([", ".join(r) for r in rows[1:4]]))
            except Exception:
                pass

    return "\n".join(parts) if parts else "No data description available."


# =============================================================================
# Cache helpers
# =============================================================================

def get_task_cache_dir(cache_path: str, task_id: str) -> str:
    return os.path.join(cache_path, task_id)


def get_model_cache_dir(cache_path: str, task_id: str, model_name: str) -> str:
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    return os.path.join(get_task_cache_dir(cache_path, task_id), safe_name)


def cache_model_response(cache_path: str, task_id: str, model_name: str,
                         response_data: Dict, trace_data: Dict, data_files: List[str]):
    """Cache a single reference model's response, trace, and data files."""
    model_dir = get_model_cache_dir(cache_path, task_id, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "response.json"), 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "trace.json"), 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)

    data_dir = os.path.join(model_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for src_path in data_files:
        if os.path.exists(src_path):
            dst_path = os.path.join(data_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)


def load_cached_model_response(cache_path: str, task_id: str, model_name: str) -> Dict:
    """Load cached response for a model. Returns None if not cached."""
    model_dir = get_model_cache_dir(cache_path, task_id, model_name)
    resp_path = os.path.join(model_dir, "response.json")
    if os.path.exists(resp_path):
        with open(resp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def cache_consensus_and_rubric(cache_path: str, task_id: str,
                               consensus_findings: List, non_consensus_findings: List, rubric: Dict):
    """Cache consensus findings and rubric at task level."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    os.makedirs(task_dir, exist_ok=True)

    with open(os.path.join(task_dir, "consensus_findings.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "consensus_findings": consensus_findings,
            "non_consensus_findings": non_consensus_findings
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(task_dir, "rubric.json"), 'w', encoding='utf-8') as f:
        json.dump(rubric, f, ensure_ascii=False, indent=2)


def load_cached_consensus_and_rubric(cache_path: str, task_id: str) -> Tuple:
    """Load cached consensus and rubric. Returns (consensus, non_consensus, rubric) or (None, None, None)."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    consensus_path = os.path.join(task_dir, "consensus_findings.json")
    rubric_path = os.path.join(task_dir, "rubric.json")

    if os.path.exists(consensus_path) and os.path.exists(rubric_path):
        with open(consensus_path, 'r', encoding='utf-8') as f:
            c_data = json.load(f)
        with open(rubric_path, 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        return c_data.get("consensus_findings", []), c_data.get("non_consensus_findings", []), rubric

    return None, None, None


# =============================================================================
# Reference model execution
# =============================================================================

def run_reference_models(query: str, data_description: str, config: Dict,
                         cache_path: str, task_id: str, data_files: List[str]) -> List[Dict]:
    """Run reference models with hermes agent, then extract insights in second round."""
    results = []

    for model in config['models']:
        model_name = model['name']

        # Check cache first
        cached = load_cached_model_response(cache_path, task_id, model_name)
        if cached is not None:
            print(f"    [Cache HIT] {model_name}")
            results.append(cached)
            continue

        print(f"    [Cache MISS] Running {model_name} with hermes...")
        t_start = time.time()
        try:
            # Step 1: Run hermes agent for this model
            from agents.hermes_agent import HermesAgent
            import shutil

            # Create a unique profile name for this reference model
            # Hermes profile names must match [a-z0-9][a-z0-9_-]{0,63}
            safe_model_name = model['model_id'].replace('/', '_').replace(' ', '_').replace('.', '-').lower()
            profile_name = f"ref_{task_id}_{safe_model_name}"[:64]  # Limit to 64 chars

            # Create a temporary input directory for this model's run
            model_dir = os.path.join(cache_path, task_id, model_name.replace('/', '_').replace(' ', '_'))
            model_input_dir = os.path.join(model_dir, "inputs")
            os.makedirs(model_input_dir, exist_ok=True)

            # Copy data files to the input directory
            for src_file in data_files:
                if os.path.exists(src_file):
                    dst_file = os.path.join(model_input_dir, os.path.basename(src_file))
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

            agent = HermesAgent(
                api_key=model['api_key'],
                base_url=model['api_url'],
                model_name=model['model_id'],
                data_root_path=model_dir,  # Parent directory, not inputs itself
                save_name=profile_name,
                max_rounds=model.get('max_rounds', 20)
            )

            # Prepare path_info for hermes
            path_info = {
                "task_id": task_id,
                "real_input_dir": model_input_dir,
                "real_output_dir": os.path.join(model_dir, "outputs"),
                "mnt_input_dir": "./inputs",
                "mnt_output_dir": "./outputs",
                "workspace_dir": os.path.join(model_dir, "workspace")
            }

            # Run hermes with the original query, mentioning the data files
            file_names = [os.path.basename(f) for f in data_files]
            files_str = ", ".join(file_names) if file_names else "the provided data"
            hermes_query = f"{query}\n\nThe data file(s) are located at: ./inputs/{files_str}"

            hermes_result = agent.interact(
                query=hermes_query,
                system_prompt="You are an expert data analyst. Analyze the dataset thoroughly.",
                run_code_func=None,
                path_info=path_info
            )

            first_round_response = hermes_result.get('model_response', '')
            session_id = hermes_result.get('session_id', '')

            # Step 2: Extract insights in second round
            # Load the session to get full conversation history
            session_data = agent._load_session(session_id) if session_id else None

            # Prepare second round prompt
            insight_extraction_prompt = (
                "Based on your analysis above, please summarize all key findings, patterns, trends, "
                "and insights you discovered. List them clearly and concisely as bullet points. "
                "Focus on actionable insights and important patterns in the data."
            )

            # Call the model again for insight extraction
            client = OpenAI(api_key=model['api_key'], base_url=model['api_url'])

            # Build conversation history for second round
            messages = [
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": first_round_response},
                {"role": "user", "content": insight_extraction_prompt}
            ]

            response = client.chat.completions.create(
                model=model['model_id'],
                messages=messages,
                temperature=0.3,
                max_tokens=2048
            )

            t_end = time.time()
            insights_response = response.choices[0].message.content

            resp_data = {
                "model_name": model_name,
                "model_response": insights_response  # This is what consensus extraction will use
            }

            trace_data = {
                "model_name": model_name,
                "model_id": model['model_id'],
                "api_url": model['api_url'],
                "first_round_query": hermes_query,  # Save the actual query sent to hermes
                "first_round_response": first_round_response,
                "hermes_session_id": session_id,
                "hermes_history": session_data if session_data else [],
                "second_round_prompt": insight_extraction_prompt,
                "insights_response": insights_response,
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', None),
                    "total_tokens": getattr(response.usage, 'total_tokens', None),
                },
                "elapsed_seconds": round(t_end - t_start, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            cache_model_response(cache_path, task_id, model_name, resp_data, trace_data, data_files)
            results.append(resp_data)

        except Exception as e:
            t_end = time.time()
            print(f"    Warning: Reference model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

            resp_data = {"model_name": model_name, "model_response": "", "error": str(e)}
            trace_data = {
                "model_name": model_name,
                "error": str(e),
                "elapsed_seconds": round(t_end - t_start, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            cache_model_response(cache_path, task_id, model_name, resp_data, trace_data, data_files)
            results.append(resp_data)

    return results


# =============================================================================
# Main task processing
# =============================================================================

def process_single_task(row: Dict, i: int, args, config: Dict, ground_truth_map: Dict) -> Dict:
    """Process a single open-ended task using full ConsensusEval pipeline with caching."""
    row.pop('history', None)

    model_response = row.get('answer') or row.get('model_response') or row.get('analysis', '')
    task_id = row.get('id') or row.get('task_id', f"task_{i}")
    query = row.get('query', '')

    task_info = ground_truth_map.get(task_id, {})
    if not query:
        query = task_info.get('query', '')

    dataset_name = getattr(args, 'dataset', 'open_ended_test')
    use_cache = getattr(args, 'use_cache', False)
    cache_path = getattr(args, 'cache_path', None)
    if not cache_path:
        cache_path = os.path.join(PROJECT_ROOT, "output", "reference_cache", dataset_name)

    try:
        # Check if we can use cached consensus + rubric
        consensus_findings, non_consensus_findings, rubric = None, None, None
        if use_cache:
            consensus_findings, non_consensus_findings, rubric = load_cached_consensus_and_rubric(cache_path, task_id)
            if rubric is not None:
                print(f"  [Task {i+1}] Using cached consensus + rubric")

        if rubric is None:
            # Step 1: Get data description
            print(f"  [Task {i+1}] Getting data description...")
            data_description = get_data_description(task_id, ground_truth_map)

            # Resolve data files for caching
            # First try dataset_csv_path from task config, then scan input directory
            csv_path = task_info.get('dataset_csv_path', '')
            data_files = []
            if csv_path:
                full_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
                if os.path.exists(full_path):
                    # If it points to a file, also include all sibling files in the same directory
                    parent_dir = os.path.dirname(full_path)
                    data_files = [
                        os.path.join(parent_dir, f) for f in sorted(os.listdir(parent_dir))
                        if os.path.isfile(os.path.join(parent_dir, f))
                    ]

            # Step 2: Run reference models (with per-model cache)
            print(f"  [Task {i+1}] Running {len(config['models'])} reference models...")
            reference_responses = run_reference_models(
                query, data_description, config, cache_path, task_id, data_files
            )
            valid_responses = [r for r in reference_responses if r.get('model_response')]
            print(f"  [Task {i+1}] Got {len(valid_responses)}/{len(reference_responses)} valid responses")

            # Step 3: Extract consensus
            print(f"  [Task {i+1}] Extracting consensus findings...")
            extractor = ConsensusExtractor(
                api_key=config['judge_model']['api_key'],
                base_url=config['judge_model']['api_url'],
                model_name=config['judge_model']['model_id'],
                consensus_threshold=config.get('consensus_threshold', 0.6),
                language=getattr(args, 'language', 'zh'),
            )
            consensus_findings, non_consensus_findings = extractor.extract(valid_responses)
            print(f"  [Task {i+1}] Found {len(consensus_findings)} consensus, {len(non_consensus_findings)} non-consensus findings")

            # Step 4: Generate rubric
            print(f"  [Task {i+1}] Generating three-layer rubric...")
            rubric_gen = RubricGenerator(
                api_key=config['judge_model']['api_key'],
                base_url=config['judge_model']['api_url'],
                model_name=config['judge_model']['model_id'],
                language=args.language
            )
            rubric = rubric_gen.generate(
                task_description=query,
                consensus_findings=consensus_findings,
                data_characteristics=task_info.get('metadata'),
                non_consensus_findings=non_consensus_findings,
            )

            # Save to cache (overwrite)
            cache_consensus_and_rubric(cache_path, task_id, consensus_findings, non_consensus_findings, rubric)

        print(f"  [Task {i+1}] Rubric: L1={len(rubric.get('layer1_must_find',[]))} items, "
              f"L2={len(rubric.get('layer2_process_quality',[]))} items, "
              f"L3={len(rubric.get('layer3_bonus_discovery',[]))} items")

        # Step 5: Judge the model response
        print(f"  [Task {i+1}] Judging model response...")
        judge = LLMJudge(
            api_key=config['judge_model']['api_key'],
            base_url=config['judge_model']['api_url'],
            model_name=config['judge_model']['model_id'],
            use_hermes=True,
            max_rounds=config.get('judge_max_rounds', 30),
        )

        # Resolve data files and model workspace for Hermes judge
        judge_data_files = []
        model_workspace_path = None

        # Get original data files
        csv_path_for_judge = task_info.get('dataset_csv_path', '')
        if csv_path_for_judge:
            full_path = os.path.join(PROJECT_ROOT, csv_path_for_judge) if not os.path.isabs(csv_path_for_judge) else csv_path_for_judge
            if os.path.exists(full_path):
                parent_dir = os.path.dirname(full_path)
                judge_data_files = [
                    os.path.join(parent_dir, f) for f in sorted(os.listdir(parent_dir))
                    if os.path.isfile(os.path.join(parent_dir, f))
                ]

        # Try to find model's workspace from infer output
        input_path = args.input_path
        if os.path.isdir(input_path):
            # input_path is like output/preds/{model}/{dataset}/conv
            # workspace is at output/preds/{model}/{dataset}/workspace/{task_id}
            parent_of_conv = os.path.dirname(input_path.rstrip('/'))
            candidate = os.path.join(parent_of_conv, 'workspace', task_id)
            if os.path.isdir(candidate):
                model_workspace_path = candidate

        judge_workspace_dir = os.path.join(args.output_path, "workspace", task_id)

        eval_result = judge.evaluate(
            analysis_output=model_response,
            rubric=rubric,
            task_description=query,
            num_runs=config.get('num_judge_runs', 1),
            data_files=judge_data_files,
            model_output_workspace=model_workspace_path,
            judge_workspace_dir=judge_workspace_dir,
        )

        row['score'] = eval_result.get('mean_score', 0)
        row['reason'] = eval_result.get('detailed_feedback', '')
        row['rubric'] = rubric
        row['consensus_findings'] = consensus_findings
        row['non_consensus_findings'] = non_consensus_findings
        row['eval_details'] = eval_result

    except Exception as e:
        import traceback
        print(f"  [Task {i+1}] Error: {str(e)}")
        traceback.print_exc()
        row['score'] = 0
        row['reason'] = f"Evaluation failed: {str(e)}"
        row['error'] = str(e)

    save_result(row, args, i)
    return row


def get_result_file_path(row: Dict, i: int, args) -> str:
    item_id = row.get('id') or row.get('task_id', f"eval_{i}")
    file_name = f"{item_id}.json"
    return os.path.join(args.output_path, file_name)


def save_result(row: Dict, args, i: int):
    file_path = get_result_file_path(row, i, args)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


def run(args):
    """Run open-ended evaluation using ConsensusEval."""
    print("\n" + "="*50)
    print("Starting Open-Ended Evaluation (ConsensusEval)")
    print("="*50)

    config_path = getattr(args, 'config_path', 'configs/reference_models.json')
    print(f"\nLoading reference models config from: {config_path}")
    config = load_reference_models_config(config_path)
    print(f"  - Loaded {len(config['models'])} reference models")
    print(f"  - Judge model: {config['judge_model']['model_id']}")
    print(f"  - Consensus threshold: {config.get('consensus_threshold', 0.6)}")
    print(f"  - Number of judge runs: {config.get('num_judge_runs', 1)}")

    use_cache = getattr(args, 'use_cache', False)
    cache_path = getattr(args, 'cache_path', None)
    if not cache_path:
        dataset_name = getattr(args, 'dataset', 'open_ended_test')
        cache_path = os.path.join(PROJECT_ROOT, "output", "reference_cache", dataset_name)
    print(f"  - Use cache: {use_cache}")
    print(f"  - Cache path: {cache_path}")

    print("\nLoading task config...")
    ground_truth_map = load_ground_truth_insights()
    print(f"  - Loaded config for {len(ground_truth_map)} tasks")

    os.makedirs(args.output_path, exist_ok=True)

    print(f"\nLoading predictions from: {args.input_path}")
    dataset = load_dataset(args.input_path)
    print(f"  - Found {len(dataset)} tasks to evaluate")

    if len(dataset) == 0:
        print("\nNo tasks found. Exiting.")
        return

    print("\nStarting evaluation...\n")
    results = []
    max_workers = getattr(args, 'max_workers', 1)

    if max_workers == 1:
        for i, row in enumerate(tqdm(dataset, desc="Evaluating tasks")):
            result = process_single_task(row, i, args, config, ground_truth_map)
            results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_task, row, i, args, config, ground_truth_map)
                for i, row in enumerate(dataset)
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating tasks"):
                results.append(future.result())

    scores = [r.get('score', 0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)
    print(f"Total tasks evaluated: {len(results)}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Results saved to: {args.output_path}")
    print("="*50 + "\n")

    summary_path = os.path.join(args.output_path, "summary.json")
    summary = {
        "total_tasks": len(results),
        "average_score": avg_score,
        "scores": scores,
        "config": {
            "reference_models": [m['name'] for m in config['models']],
            "judge_model": config['judge_model']['model_id'],
            "consensus_threshold": config.get('consensus_threshold', 0.6),
            "num_judge_runs": config.get('num_judge_runs', 1),
            "use_cache": use_cache
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run open-ended evaluation")
    parser.add_argument("--input_path", required=True, help="Path to model predictions")
    parser.add_argument("--output_path", required=True, help="Path to save evaluation results")
    parser.add_argument("--config_path", default="configs/reference_models.json", help="Path to reference models config")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--use_cache", action="store_true", help="Reuse cached consensus findings and rubrics")
    parser.add_argument("--cache_path", default=None, help="Path for consensus/rubric cache (default: output/reference_cache/{dataset})")
    parser.add_argument("--dataset", default="open_ended_test", help="Dataset name for cache directory")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Language for rubric generation (zh or en)")
    args = parser.parse_args()
    args.use_hermes_judge = True
    run(args)
