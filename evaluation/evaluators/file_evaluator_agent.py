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
        
#         system_prompt = f"""你是一个智能文件评估员。
# 你的任务是根据“参考文件”和“问题”来判断“预测文件”是否正确。

# 问题: {{question}}

# 文件:
# - 参考文件: "{{reference_path}}"
# - 预测文件: "{{prediction_path}}"

# 指令:
# 1. 编写 Python 代码加载这两个文件（例如，使用 pandas 或 openpyxl）。
# 2. 比较文件的内容。注意问题中的要求。
#    - 如果问题要求特定的值，检查该值是否存在。
#    - 如果问题要求一个表格，比较数据框。
#    - 要合理：除非特别说明，否则忽略微小的浮点差异或行/列顺序。
# 3. 分析后，以 JSON 格式输出你的最终判断：
#    {{{{"is_correct": true/false, "reason": "解释"}}}}
   
# 你必须使用 `execute_code` 工具来检查文件。不要猜测。"""


        system_prompt =  f"""你是一个智能文件评估员（File Judge）。
目标：根据“问题 + 参考文件”判断“预测文件”是否满足要求，并输出 JSON 结论。

问题: {{question}}

文件:
- 参考文件: "{{reference_path}}"
- 预测文件: "{{prediction_path}}"

【工具调用预算（必须遵守）】
- 你最多只能调用 `execute_code` 20 次（绝对不允许超过 20 次)。
- 每次调用前先在文本里写清楚“这一次要验证什么”，避免无目的重跑。
- 一旦拿到足够证据，就必须立刻停止继续调用工具并输出最终 JSON。

【总体策略：先粗后细，抽测优先，必要时才加深】
你要按以下 3 阶段执行，且必须早停：
阶段1（1-5次调用）：快速探测/建索引
- 读取参考文件和预测文件的基础信息：文件类型、sheet 列表、每个 sheet 的行列数、列名集合、dtype 概览。
- 如果题目有明确的表头/列顺序/输出格式/文件结构等要求，需要严格比对如不匹配直接给出 false 的证据并结束。否则可采用宽松比对：允许行/列顺序不同、允许多余列（不影响问题要求）、允许微小浮点误差（1e-6）、允许空值表示差异（NaN/None/"" 视为等价），并用抽样（头/中/尾/随机）验证内容一致性。
- 同时在代码里生成“抽样行索引”（头/中/尾 + 随机）供后续复用，避免重复扫描。

阶段2（5-10次调用）：抽测内容一致性（文件比较大时禁止全量逐行，少量时可以）
- 对每个要比较的 sheet，做抽样对比：
  * 头 N 行 + 尾 N 行 + 中间 N 行 + 随机 N 行（默认 N=30；若行数小则自动缩小或全量）。
- 规范化后比较（忽略微小浮点误差、忽略行/列顺序）：
  * 列：按列名对齐
  * 行：用“行哈希”或“排序+对齐”做集合式对比（仅对样本）
  * 数值：abs/rel tol=1e-6；日期转 ISO；字符串 strip；空值统一
- 若样本差异显著，直接判 false 并结束；若样本一致，继续阶段3。

阶段3（最多再用 1-5 次调用）：只做“问题相关”的关键校验
- 从问题中只提取高收益约束来验证（不要做全面 NLP）：
  例如：去重/保留第一条、排序、过滤条件、分组汇总、TopK、生成新表头/新列、特定值是否存在、统计值（总和/均值/计数）等。
- 对能快速计算的约束做验证：
  * 例如：行数*

【禁止行为（非常重要）】
- 禁止逐行全量比对大文件；除非文件很小（如 < 200 行）或问题明确要求全量一致。
- 禁止为“验证更多细节”无限加工具调用，一定不能调用20次以上；
- 禁止输出除最终 JSON 之外的内容（中间过程只在代码里临时计算，不要 print）。

【输出格式（最终只输出一次）】
以 JSON 输出最终判断：
{{{{"is_correct": true/false, "reason": "简洁说明：比较了哪些sheet/抽样策略/关键差异证据或通过的关键检查"}}}}。

现在开始。你可以调用 `execute_code`，但必须严格按阶段推进并遵守预算。
"""


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
