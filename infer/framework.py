import json
import os
import concurrent.futures
from typing import Callable, Dict, List, Any
from tqdm import tqdm

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
