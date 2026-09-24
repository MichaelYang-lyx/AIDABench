import json
import os
import concurrent.futures
from datetime import datetime, timezone
from typing import Callable, Dict, List, Any
from tqdm import tqdm
from infer.failure_policy import is_retryable_api_failure

class InferenceRunner:
    def __init__(self, num_workers: int = 4, id_field: str = "id"):
        self.num_workers = num_workers
        self.id_field = id_field

    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def load_processed_ids(self, output_path: str) -> set:
        processed_ids = set()
        
        # Check if output_path is a directory (new behavior)
        if os.path.isdir(output_path):
            # Scan directory for {id}.json files
            for filename in os.listdir(output_path):
                if filename.endswith(".json"):
                    # Assuming filename is {id}.json
                    processed_id = filename[:-5]
                    processed_ids.add(processed_id)
            return processed_ids

        # Legacy behavior: check single jsonl file
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if self.id_field in item:
                                processed_ids.add(item[self.id_field])
                        except json.JSONDecodeError:
                            continue
        return processed_ids

    def _write_retryable_failure(self, result: Dict, output_path: str) -> str:
        """Persist a retryable result without making it an inference checkpoint.

        ``conv/<ID>.json`` controls resume behavior, so failed results must not
        be written there.  Keep every failed attempt in the sibling
        ``failed_traces/<ID>.jsonl`` instead; the next inference run will still
        pick up the task while its full history remains available for diagnosis.
        """
        item_id = str(result.get(self.id_field, "unknown"))
        trace_dir = os.path.join(
            os.path.dirname(os.path.abspath(output_path)), "failed_traces"
        )
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"{item_id}.jsonl")

        record = dict(result)
        record["_failure_recorded_at"] = datetime.now(timezone.utc).isoformat()
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return trace_path

    def run(self, 
            data_path: str, 
            output_path: str, 
            process_func: Callable[[Dict], Dict],
            model_kwargs: Dict[str, Any] = None):
        
        # Check if output_path is intended to be a directory
        # We assume if it doesn't have an extension like .jsonl or .txt, it's a directory
        # OR if the user explicitly created it as a directory
        is_directory_mode = not os.path.splitext(output_path)[1]
        
        if is_directory_mode:
            os.makedirs(output_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Load data
        all_data = self.load_data(data_path)
        print(f"Loaded {len(all_data)} items from {data_path}")

        # Check existing progress (Breakpoint fix)
        processed_ids = self.load_processed_ids(output_path)
        print(f"Found {len(processed_ids)} already processed items.")

        # Filter remaining tasks
        tasks_to_run = [d for d in all_data if d.get(self.id_field) not in processed_ids]
        print(f"Remaining items to process: {len(tasks_to_run)}")

        if not tasks_to_run:
            print("All tasks completed.")
            return

        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            def safe_process(item):
                try:
                    return process_func(item, **(model_kwargs or {}))
                except Exception as e:
                    print(f"Error processing {item.get(self.id_field)}: {e}")
                    result = item.copy()
                    result['error'] = str(e)
                    return result

            # Submit all tasks
            future_to_item = {executor.submit(safe_process, item): item for item in tasks_to_run}
            
            # If not directory mode, open file handle once
            f_out = None
            if not is_directory_mode:
                 f_out = open(output_path, 'a', encoding='utf-8')

            try:
                for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(tasks_to_run)):
                    result = future.result()
                    if result:
                        if result.get("_retryable_failure") and is_retryable_api_failure(result.get("model_response")):
                            item_id = result.get(self.id_field, "unknown")
                            trace_path = self._write_retryable_failure(result, output_path)
                            print(
                                f"Retryable inference failure for {item_id}; "
                                f"checkpoint not written, trace saved to {trace_path}."
                            )
                            continue
                        # An old agent may still mark a terminal model outcome
                        # retryable. The checkpoint should reflect its final status.
                        result.pop("_retryable_failure", None)
                        if is_directory_mode:
                            # Save to individual file: output_path/{id}.json
                            item_id = result.get(self.id_field, "unknown")
                            file_name = f"{item_id}.json"
                            file_path = os.path.join(output_path, file_name)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        else:
                            # Append to single file
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
            finally:
                if f_out:
                    f_out.close()
