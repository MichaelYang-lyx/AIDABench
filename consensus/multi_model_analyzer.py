"""Multi-model reference analysis engine.

Orchestrates N different models (via HermesAgent) to independently analyze the same data.
Supports parallel execution and collects all analysis results.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.hermes_agent import HermesAgent


class MultiModelAnalyzer:
    """Runs multiple models in parallel on the same data analysis task."""

    def __init__(
        self,
        reference_models: List[Dict[str, str]],
        data_root_path: str,
        max_rounds: int = 90,
        max_workers: int = 3,
    ):
        """
        Args:
            reference_models: List of dicts with keys: 'api_key', 'base_url', 'model_name'
            data_root_path: Root path for data files
            max_rounds: Max rounds per agent
            max_workers: Max parallel workers
        """
        self.reference_models = reference_models
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        self.max_workers = max_workers

    def analyze(
        self,
        query: str,
        system_prompt: str,
        path_info: Dict[str, str],
        task_id: str,
    ) -> List[Dict[str, Any]]:
        """Run all reference models on the same task.

        Args:
            query: The analysis task/question
            system_prompt: System prompt for the agent
            path_info: Dict with 'real_input_dir', 'real_output_dir', etc.
            task_id: Unique task identifier

        Returns:
            List of analysis results, one per model. Each result contains:
            - model_name: str
            - model_response: str
            - history: list
            - session_id: str
        """
        results = []

        def run_single_model(model_config: Dict[str, str]) -> Dict[str, Any]:
            """Run a single model analysis."""
            model_name = model_config["model_name"]
            save_name = f"consensus_ref_{model_name.replace('/', '_')}"

            agent = HermesAgent(
                api_key=model_config["api_key"],
                base_url=model_config["base_url"],
                model_name=model_name,
                data_root_path=self.data_root_path,
                save_name=save_name,
                max_rounds=self.max_rounds,
            )

            # Run analysis
            result = agent.interact(
                query=query,
                system_prompt=system_prompt,
                run_code_func=None,  # Not used by HermesAgent
                path_info=path_info,
            )

            # Attach model name to result
            result["model_name"] = model_name
            return result

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_single_model, model_config): model_config
                for model_config in self.reference_models
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Running reference models",
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    model_config = futures[future]
                    print(
                        f"Error running model {model_config['model_name']}: {e}"
                    )
                    # Add error result
                    results.append(
                        {
                            "model_name": model_config["model_name"],
                            "model_response": f"Error: {e}",
                            "history": [],
                            "session_id": None,
                            "error": str(e),
                        }
                    )

        return results
