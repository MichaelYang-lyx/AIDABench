import os
import sys
import json
import ast
from typing import Dict, Any, List, Union
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import anthropic

@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResponse:
    """Response from the LLM."""
    think_text: Optional[str]
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def has_tool_calls(self):
        # type: () -> bool
        return len(self.tool_calls) > 0


def parse_response(response) -> LLMResponse:
        """Parse the Anthropic API response into LLMResponse."""
        content_text = ""
        think_text = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            if block.type == "thinking":
                think_text += block.thinking

            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {}
                ))
        
        # Determine finish reason
        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "end_turn":
            finish_reason = "stop"
        
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        
        return LLMResponse(
            think_text=think_text if think_text else None,
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )

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

class ClaudeSubprocessAgent:
    def __init__(self, api_key: str, base_url: str, model_name: str, data_root_path: str, max_rounds: int = 20):
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.data_root_path = data_root_path
        self.max_rounds = max_rounds
        
        # Define the tools (OpenAI format)
        self.tools = [
                {
                    "name": "execute_code",
                    "description": "在子进程环境中执行一段Python代码（注意：每次执行都是独立的，不保留状态）。",
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
            
                raw_response= self.client.messages.create(
                    model=self.model_name,  
                    system=system_msg,
                    messages=messages[1:],  
                    max_tokens=16000,
                    temperature=1,
                    tools=self.tools,
                    tool_choice={"type": "auto"},
                )
                
            else:
                raw_response= self.client.messages.create(
                    model=self.model_name,  
                    messages=messages,  
                    max_tokens=16000,
                    temperature=1,
                    tools=self.tools,
                    tool_choice={"type": "auto"},
                )
            response = parse_response(raw_response)
            return response, response.usage['completion_tokens']
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
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments  # already a dict from parse_response
                    })

                input_message.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Process each tool call and collect results
                tool_results_content = []
                for tool_call in tool_calls:
                    function_name = tool_call.name
                    function_args = tool_call.arguments  # already a dict
                    tool_call_id = tool_call.id

                    if function_name == "execute_code":
                        try:
                            code = function_args.get('code', '')

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
                                
                            # Collect tool result
                            tool_results_content.append({
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": execution_result
                            })

                        except Exception as e:
                             error_msg = f"Error processing tool call: {e}"
                             tool_results_content.append({
                                 "type": "tool_result",
                                 "tool_use_id": tool_call_id,
                                 "content": error_msg
                             })
                    else:
                         tool_results_content.append({
                             "type": "tool_result",
                             "tool_use_id": tool_call_id,
                             "content": "Error: Unknown function."
                         })

                # Append all tool results in a single user message
                input_message.append({
                    "role": "user",
                    "content": tool_results_content
                })
            else:
                # No tool calls -> Final Answer
                final_text = generated_message.content
                if final_text:
                    input_message.append({"role": "assistant", "content": final_text})
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
