from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from toolkits import CodeExecutionToolkit

class FileEvaluatorAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, max_rounds: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_rounds = max_rounds

    def evaluate(self, question: str, reference_path: str, prediction_path: str) -> Dict[str, Any]:
        """
        Evaluate the prediction against the reference using the LLM + Code Interpreter.
        Uses ClaudeJupyterAgent for the interaction loop.
        """
        try:
            from agents.claude_jupyter_agent import ClaudeJupyterAgent
            from agents.claude_subprocess_agent import ClaudeSubprocessAgent
        except ImportError:
            # Fallback if not in path, try adding project root
            import sys
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
            if project_root not in sys.path:
                sys.path.append(project_root)
            from agents.claude_jupyter_agent import ClaudeJupyterAgent

        # Initialize the agent
        # We use a dummy data_root_path as we provide absolute paths for files
        agent = ClaudeSubprocessAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            data_root_path="/tmp",
            max_rounds=self.max_rounds
        )
        
        # Initialize toolkit for this evaluation session
        # Use a unique namespace/session to avoid conflicts
        toolkit = CodeExecutionToolkit(sandbox="subprocess", namespace="evaluator", default_session_id=f"eval_{time.time()}")
        run_code_func = toolkit.get_tools()[0]
        
        system_prompt = f"""你是一个智能文件评估员。
你的任务是根据“参考文件”和“问题”来判断“预测文件”是否正确。

问题: {{question}}

文件:
- 参考文件: "{{reference_path}}"
- 预测文件: "{{prediction_path}}"

指令:
1. 编写 Python 代码加载这两个文件（例如，使用 pandas 或 openpyxl）。
2. 比较文件的内容。注意问题中的要求。
   - 如果问题要求特定的值，检查该值是否存在。
   - 如果问题要求一个表格，比较数据框。
   - 要合理：除非特别说明，否则忽略微小的浮点差异或行/列顺序。
3. 分析后，以 JSON 格式输出你的最终判断：
   {{{{"is_correct": true/false, "reason": "解释"}}}}
   
你必须使用 `execute_code` 工具来检查文件。不要猜测。"""

        # Format the system prompt with actual values
        system_prompt = system_prompt.format(
            question=question,
            reference_path=reference_path,
            prediction_path=prediction_path
        )
        
        try:
            # Run interaction loop via OpenAIJupyterAgent
            result = agent.interact(
                query="Please start the evaluation.",
                system_prompt=system_prompt,
                run_code_func=run_code_func,
                path_info={} # No path replacement needed
            )
            
            model_response = result.get("model_response", "")
            history = result.get("history", [])
            
            # Extract JSON from response
            try:
                # Try to find JSON block using regex
                import re
                json_match = re.search(r"\{.*\}", model_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    eval_result = json.loads(json_str)
                    if "is_correct" in eval_result and "reason" in eval_result:
                        eval_result["eval_history"] = history
                        return eval_result
                
                # If the whole response is JSON
                eval_result = json.loads(model_response)
                if "is_correct" in eval_result and "reason" in eval_result:
                    eval_result["eval_history"] = history
                    return eval_result
                    
            except Exception:
                pass
                
            return {
                "is_correct": False, 
                "reason": f"Could not parse evaluation result from model response: {model_response}",
                "eval_history": history
            }
            
        finally:
             # Cleanup
            toolkit.reset_session()
