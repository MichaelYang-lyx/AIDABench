import os
import sys
import argparse

import re

# Add project root to path so we can import from evaluation and infer
# infer/runner/run_chart.py -> infer/runner -> infer -> root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from infer.framework import InferenceRunner
try:
    from agents.openai_jupyter_agent import OpenAIJupyterAgent
    from agents.claude_jupyter_agent import ClaudeJupyterAgent
    from agents.openai_subprocess_agent import OpenAISubporcessAgent
    from agents.claude_subprocess_agent import ClaudeSubprocessAgent
    from toolkits import CodeExecutionToolkit, generate_file_info_string, extract_workbook_summary3b
    from infer.dataset_info import DATASET_INFO
except ImportError:
    print("Warning: Could not import OpenAIJupyterAgent or Evaluation Toolkit. Ensure dependencies are met.")
    OpenAIJupyterAgent = None
    OpenAISubporcessAgent = None
    ClaudeSubprocessAgent = None
    CodeExecutionToolkit = None
    generate_file_info_string = None
    extract_workbook_summary3b = None
    DATASET_INFO = None

# Dataset Mapping
DATASET_MAPPING = {
    "chart_mini": os.path.join("chart", "chart_mini", "chart_mini.jsonl"),
    # Add more datasets here as needed
}

def process_row(row: dict, agent: OpenAIJupyterAgent, prompt_path: str = None, get_sys_msg_func=None, need_info: bool = False, picture_path: str = None) -> dict:
    """
    Process a single row using the agent.
    """
    question = row.get('question', '')
    file_path = row.get('input_file', '') # e.g. "chart/chart_mini/data/0.xlsx"
    task_id = row.get('id', 'unknown')
    
    # 1. Initialize Messages
    if prompt_path and get_sys_msg_func:
        try:
            sys_content = get_sys_msg_func(prompt_path, question)
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            sys_content = "You are a helpful assistant capable of writing and executing Python code to solve data analysis tasks."
    else:
        sys_content = "You are a helpful assistant capable of writing and executing Python code to solve data analysis tasks."

    # 2. Enhance Prompt with File Info (if available)
    real_file_path_list=[]
    if file_path:
        # Split file paths by newline or semicolon (English or Chinese)
        pattern = r'[\n;；]+'
        file_list = [f.strip() for f in re.split(pattern, str(file_path).strip()) if f.strip()]
        
        all_info1 = []
        all_info2 = []
        
        for fname in file_list:
            real_file_path = ""
            if not os.path.isabs(fname):
                real_file_path = os.path.join(agent.data_root_path, str(task_id), fname)
            else:
                real_file_path = fname

            if os.path.exists(real_file_path):
                real_file_path_list.append(real_file_path)
                try:
                    if need_info and generate_file_info_string:
                        info1 = generate_file_info_string(real_file_path)
                        info2 = extract_workbook_summary3b(real_file_path)
                        
                        all_info1.append(f"文件 {fname} 描述: {info1}")
                        all_info2.append(f"文件 {fname} 摘要: {info2}")
                    
                except Exception as e:
                    print(f"Error generating file info for {fname}: {e}")
        
        if all_info1 or all_info2:
             info1_str = "\n".join(all_info1)
             info2_str = "\n".join(all_info2)
             new_system_text = f"[\"{sys_content}\",\"{'所有文件描述:'+info1_str}\",\"{'所有文件摘要:'+info2_str}\"]"
             sys_content = new_system_text
    
    # 3. Initialize Code Execution Toolkit (Persistent Session per Task)
    toolkit = None
    if CodeExecutionToolkit:
        sandbox_type = "jupyter"
        if OpenAISubporcessAgent and isinstance(agent, OpenAISubporcessAgent):
             sandbox_type = "subprocess"
        if ClaudeSubprocessAgent and isinstance(agent, ClaudeSubprocessAgent):
             sandbox_type = "subprocess"
        
        toolkit = CodeExecutionToolkit(sandbox=sandbox_type, namespace=f"task_{task_id}", timeout=30)
        run_code = toolkit.get_tools()[0] # execute_code function tool
    else:
            def run_code(code, **kwargs): return "Execution Environment Not Available"
    
    # 4. Run Interaction Loop
    # breakpoint()
    try:
        path_info = {}
        mnt_dir_path='/mnt/data'
        mnt_dir_result_path='/mnt/result'
        
        if real_file_path_list:
            file_paths_str = ", ".join([os.path.join(mnt_dir_path, os.path.basename(f)) for f in real_file_path_list])
            question = f"{question}\n\n 你所用到的文件在: {file_paths_str}"
            first_file_dir = os.path.dirname(real_file_path_list[0])
            path_info = {'real_input_dir': first_file_dir,
                         'mnt_input_dir': mnt_dir_path}

        if picture_path:
            picture_path = os.path.join(picture_path, str(task_id))
            os.makedirs(picture_path, exist_ok=True)
            path_info['real_output_dir'] = picture_path
            path_info['mnt_output_dir'] = mnt_dir_result_path

        output_files_str = row.get('output_file', '')
        if output_files_str:
            file_out_paths_str = ", ".join([os.path.join(mnt_dir_result_path, os.path.basename(f)) for f in output_files_str.split('\n') if f.strip()])
            question = f"{question}\n\n 你的输出结果保存到: {file_out_paths_str}"
        elif picture_path:
            # If output_path is not specified in row, prompt to save to picture_path (mapped to mnt_dir_result_path)
            # Assuming one picture output if not specified? Or just generic "save to /mnt/result"
            question = f"{question}\n\n 你的输出结果保存到: {mnt_dir_result_path}"
        
        interaction_result = agent.interact(query=question, system_prompt=sys_content, run_code_func=run_code, path_info=path_info)
        
        # 5. Prepare Result
        result = row.copy()
        result.update(interaction_result)
        
        return result
    finally:
        if toolkit:
            toolkit.reset_session()



def run(args):
    """
    Main entry point for chart inference.
    args can be an argparse.Namespace object or a similar object with attributes.
    """
    # Resolve Data Path (prefer infer/dataset_info.py)
    if args.data_path:
        data_path = args.data_path
        dataset_name = args.dataset or "custom"
    else:
        dataset_name = args.dataset
        rel_path = None
        if DATASET_INFO and dataset_name in DATASET_INFO:
            rel_path = DATASET_INFO[dataset_name].get("file_path")
        if not rel_path:
            rel_path = DATASET_MAPPING.get(dataset_name)
        if not rel_path:
            print(f"Warning: Dataset '{dataset_name}' not in mapping or dataset_info. Trying default path construction.")
            rel_path = os.path.join("data_visualization", f"{dataset_name}.jsonl")
        data_path = os.path.join(args.data_root, rel_path)

    # Resolve Output Path
    if args.output_path:
        OUTPUT_PATH = os.path.abspath(args.output_path)
        PICTURE_PATH = os.path.join(os.path.dirname(OUTPUT_PATH), "pictures")
    else:
        save_name = getattr(args, 'save_name', None) or args.model_name
        OUTPUT_PATH = os.path.abspath(os.path.join("output", "preds", save_name, dataset_name, "conv"))
        PICTURE_PATH = os.path.join(os.path.dirname(OUTPUT_PATH), "pictures")

    print(f"Starting Inference with Agent...\nModel: {args.model_name}\nDataset: {dataset_name}\nData Path: {data_path}\nOutput Path: {OUTPUT_PATH}\nPicture Path: {PICTURE_PATH}\nConcurrency: {args.num_workers}\nMax Rounds: {getattr(args, 'max_rounds', 20)}")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    if OpenAIJupyterAgent is None:
        print("Error: OpenAIJupyterAgent not found. Cannot proceed.")
        sys.exit(1)

    # Initialize Agent
    agent_data_root = args.data_root
    if DATASET_INFO and dataset_name in DATASET_INFO:
        # DATASET_INFO paths are relative to args.data_root
        rel_root = DATASET_INFO[dataset_name].get("data_root_path")
        if rel_root:
            agent_data_root = os.path.join(args.data_root, rel_root)
            
    agent_class = OpenAIJupyterAgent
    if hasattr(args, 'agent_type'):
        if 'jupyter' in args.agent_type:
            if args.agent_type == 'claude_jupyter_agent':
                agent_class = ClaudeJupyterAgent
            else:
                agent_class = OpenAIJupyterAgent
        else:
            if args.agent_type == 'claude_subprocess_agent':
                agent_class = ClaudeSubprocessAgent
            else:
                agent_class = OpenAISubporcessAgent
            
    print(f"Using Agent: {agent_class.__name__}")

    agent = agent_class(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        data_root_path=agent_data_root,
        max_rounds=getattr(args, 'max_rounds', 20)
    )

    # Resolve Prompt Path
    prompt_path = None
    if hasattr(args, 'prompt_file') and args.prompt_file:
        if os.path.isabs(args.prompt_file):
            prompt_path = args.prompt_file
        else:
            # Check in infer/prompts/
            # current file is infer/runner/run_chart.py
            # infer/prompts is ../prompts/ relative to infer/runner/ ? No, infer/prompts
            # infer/runner/../prompts/ -> infer/prompts/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(os.path.dirname(current_dir), "prompts", args.prompt_file)
            if os.path.exists(possible_path):
                prompt_path = possible_path
            else:
                prompt_path = args.prompt_file # Fallback to relative to cwd

    # Run Inference
    runner = InferenceRunner(num_workers=args.num_workers)
    
    # Extract helper function from args if available
    get_sys_msg_func = getattr(args, 'get_sys_msg_func', None)
    
    runner.run(
        data_path=data_path,
        output_path=OUTPUT_PATH,
        process_func=process_row,
        model_kwargs={
            'agent': agent, 
            'prompt_path': prompt_path,
            'get_sys_msg_func': get_sys_msg_func,
            'need_info': getattr(args, 'need_info', False),
            'picture_path': PICTURE_PATH
        }
    )

if __name__ == "__main__":
    pass
