import os
import sys
import json
import ast
from typing import Dict, Any, List, Union
from openai import OpenAI

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
        print("Warning: Could not import CodeExecutionToolkit or helpers in OpenAI Agent.")
        CodeExecutionToolkit = None
        generate_file_info_string = None
        extract_workbook_summary3b = None

class ClaudeJupyterAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, data_root_path: str, max_rounds: int = 20):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        
        # Define the tools (OpenAI format)
        self.tools = [
                {
                    "name": "execute_code",
                    "description": "在有状态的 Jupyter 环境中执行 Python 代码。支持变量、函数和导入库的跨调用持久化。适用于数学计算、数据处理和逻辑验证。输出将捕获标准输出和异常信息。",
                    "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": { "type": "string", "description": "要执行的 Python 源代码。" }
                    },
                    "required": ["code"]
                    }
                }
                ]

    def _get_response_openai(self, messages: List[Dict]):
        has_system = bool(messages) and messages[0].get("role") == "system"
        try:
            if has_system:
                system_msg = messages[0].get("content")
                response= self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages[1:],          
                    max_tokens=16000,
                    temperature=1,
                    tools=self.tools,
                    # tool_choice="auto",
                    extra_body={"system": [{"type": "text", "text": system_msg}]},
                )
            
            else:
                response= self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,              
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=1.0,
                )
            return response.choices[0].message, response.usage.completion_tokens
        except Exception as e:
            print(f"API Error: {e}")
            raise e

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
                generated_message, completion_tokens = self._get_response_openai(input_message)
                
                
                all_tokens += completion_tokens
            except Exception as e:
                final_response = f"Error during API call: {e}"
                break
                
            # Check for tool calls
            tool_calls = generated_message.tool_calls
            
            if tool_calls:
                # Add assistant message with tool calls to history
                assistant_content = []
                generated_text = generated_message.content
                if generated_text:
                    assistant_content.append({
                        "type": "text",
                        "text": generated_text
                    })
                else:
                    assistant_content.append({
                        "type": "text",
                        "text": "call_tools"
                    })
                
                for tool_call in tool_calls:
                    try:
                        args_input = json.loads(tool_call.function.arguments)
                    except:
                        args_input = {}
                        
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": args_input
                    })

                input_message.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    tool_call_id = tool_call.id
                    
                    if function_name == "execute_code":
                        try:
                            # Parse arguments
                            try:
                                args_dict = json.loads(function_args)
                                code = args_dict.get('code', '')
                            except json.JSONDecodeError:
                                # Fallback for malformed JSON
                                try:
                                     # User example used ast.literal_eval as fallback
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
                                    # Ensure non-interactive backend for Matplotlib to avoid GUI errors in threads

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
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_call_id,
                                        "content": execution_result
                                    }
                                ]
                            })
                            
                        except Exception as e:
                             error_msg = f"Error processing tool call: {e}"
                             input_message.append({
                                 "role": "user",
                                 "content": [
                                     {
                                         "type": "tool_result",
                                         "tool_use_id": tool_call_id,
                                         "content": error_msg
                                     }
                                 ]
                             })
                    else:
                         input_message.append({
                             "role": "user",
                             "content": [
                                 {
                                     "type": "tool_result",
                                     "tool_use_id": tool_call_id,
                                     "content": "Error: Unknown function."
                                 }
                             ]
                         })
            else:
                # No tool calls -> Final Answer
                final_text = generated_message.content
                if final_text:
                    input_message.append(generated_message)
                    final_response = final_text
                else:
                    final_response = "Empty response from model."
                break
        
        return {
            "model_response": final_response,
            "history": [
                msg.model_dump() if hasattr(msg, 'model_dump') else msg 
                for msg in input_message
            ],
            "total_tokens": all_tokens,
            "rounds": round_count
        }
