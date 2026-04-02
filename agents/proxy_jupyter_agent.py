import os
import sys
import json
import ast
import uuid
import time
from typing import Dict, Any, List, Union
import openai_proxy

# Try imports, assuming the project root is in PYTHONPATH or handled by the runner
try:
    from toolkits import CodeExecutionToolkit, generate_file_info_string, extract_workbook_summary3b
except ImportError:
    # If run directly or path not set, try adding project root
    # assuming this file is in agents/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        from toolkits import CodeExecutionToolkit, generate_file_info_string, extract_workbook_summary3b
    except ImportError:
        print("Warning: Could not import CodeExecutionToolkit or helpers in Proxy Agent.")
        CodeExecutionToolkit = None
        generate_file_info_string = None
        extract_workbook_summary3b = None

class ProxyJupyterAgent:
    def __init__(self, api_key: str, model_name: str, data_root_path: str,
                 channel_code: str = "ali", transaction_id: str = "proxy_task",
                 enable_thinking: bool = True, max_rounds: int = 20):
        self.client = openai_proxy.GptProxy(api_key=api_key)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.channel_code = channel_code
        self.transaction_id = transaction_id
        self.enable_thinking = enable_thinking
        self.max_rounds = max_rounds

        # Define the tools (OpenAI format)
        self.tools = [{
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "在同一个持续的Jupyter环境中执行一段Python代码（有状态）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "要执行的、简短的Python代码片段。"
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False
                }
            }
        }]

    def _get_response(self, messages: List[Dict]):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.generate(
                    messages=messages,
                    model=self.model_name,
                    channel_code=self.channel_code,
                    transaction_id=f"{self.transaction_id}_{uuid.uuid4().hex[:8]}",
                    enable_thinking=self.enable_thinking,
                    tools=self.tools,
                    tool_choice="auto",
                )
                response_json = response.json()

                if response_json.get('code') != 10000:
                    msg = response_json.get('msg', 'Unknown error')
                    print(f"API returned error: {msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(20)
                        continue
                    raise Exception(f"Proxy API error: {msg}")

                message = response_json['data']['response_content']['choices'][0]['message']
                completion_tokens = response_json['data']['response_content']['usage']['completion_tokens']
                return message, completion_tokens
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API Error: {e}, retrying in 20s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(20)
                    continue
                print(f"API Error (final): {e}")
                raise e
        raise Exception("Max retries exceeded")

    def interact(self, query: str, system_prompt: str, run_code_func: Any, path_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Run the interaction loop with the model and tools.
        """

        input_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        round_count = 0
        all_tokens = 0
        final_response = ""

        # Interaction Loop
        while True:
            round_count += 1
            if round_count > self.max_rounds:
                final_response = "Error: Too many rounds reached."
                break

            try:
                generated_message, completion_tokens = self._get_response(input_message)
                all_tokens += completion_tokens
            except Exception as e:
                final_response = f"Error during API call: {e}"
                break

            # Check for tool calls
            tool_calls = generated_message.get('tool_calls')

            if tool_calls:
                # Add assistant message with tool calls to history
                input_message.append(generated_message)

                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = tool_call['function']['arguments']
                    tool_call_id = tool_call['id']

                    if function_name == "execute_code":
                        try:
                            # Parse arguments
                            try:
                                args_dict = json.loads(function_args)
                                code = args_dict.get('code', '')
                            except json.JSONDecodeError:
                                # Fallback for malformed JSON
                                try:
                                     args_dict = ast.literal_eval(function_args)
                                     code = args_dict.get('code', '')
                                except:
                                     code = ""

                            if not code:
                                execution_result = "Error: No code provided in arguments."
                            else:
                                # Replace data path placeholders if any
                                code_to_exec = code
                                if isinstance(path_info, dict):
                                    if 'mnt_input_dir' in path_info and 'real_input_dir' in path_info:
                                        code_to_exec = code_to_exec.replace(path_info['mnt_input_dir'], path_info['real_input_dir'])
                                    if 'mnt_output_dir' in path_info and 'real_output_dir' in path_info:
                                        code_to_exec = code_to_exec.replace(path_info['mnt_output_dir'], path_info['real_output_dir'])
                                else:
                                    code_to_exec = code.replace('/mnt/data', self.data_root_path)

                                # Execute
                                try:
                                    code_to_exec = f"import matplotlib\nmatplotlib.use('Agg')\n{code_to_exec}"

                                    res = run_code_func(code=code_to_exec)

                                    if len(str(res)) > 2000:
                                        res = str(res)[:1000] + '...' + str(res)[-1000:]
                                        execution_result = f"Executed Results(Response too long; showing the first 1000 characters and the last 1000 characters.):\n{res}"
                                    else:
                                        execution_result = f"Executed Results:\n{res}"
                                except Exception as e:
                                    execution_result = f"Execution Error: {e}"

                            # Append Tool Output
                            input_message.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": "execute_code",
                                "content": execution_result
                            })

                        except Exception as e:
                             error_msg = f"Error processing tool call: {e}"
                             input_message.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": "execute_code",
                                "content": error_msg
                            })
                    else:
                         input_message.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": "Error: Unknown function."
                            })
            else:
                # No tool calls -> Final Answer
                final_text = generated_message.get('content')
                if final_text:
                    input_message.append(generated_message)
                    final_response = final_text
                else:
                    final_response = "Empty response from model."
                break

        return {
            "model_response": final_response,
            "history": input_message,
            "total_tokens": all_tokens,
            "rounds": round_count
        }
