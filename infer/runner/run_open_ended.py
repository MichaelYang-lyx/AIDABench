import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from infer.framework import InferenceRunner

OpenAIJupyterAgent = None
HermesAgent = None
SkillJupyterAgent = None
LightLLMJupyterAgent = None
XHXPipelineAgent = None
CodeExecutionToolkit = None
generate_file_info_string = None
DATASET_INFO = None

try:
    from agents.openai_jupyter_agent import OpenAIJupyterAgent
except ImportError:
    pass
try:
    from agents.hermes_agent import HermesAgent
except ImportError:
    pass
try:
    from agents.skill_jupyter_agent import SkillJupyterAgent
except ImportError:
    pass
try:
    from agents.lightllm_jupyter_agent import LightLLMJupyterAgent
except ImportError:
    pass
try:
    from agents.xhx_pipeline import XHXPipelineAgent
except ImportError:
    pass
try:
    from toolkits import CodeExecutionToolkit, generate_file_info_string
except ImportError:
    pass
try:
    from infer.dataset_info import DATASET_INFO
except ImportError:
    pass


DEFAULT_SYSTEM_PROMPT = """You are an expert data analyst. You have access to a Python environment with pandas, matplotlib, seaborn, scipy, numpy, and scikit-learn.

Your task is to analyze the given dataset thoroughly and provide actionable insights. You should:
1. Load and explore the data (shape, columns, types, missing values)
2. Perform statistical analysis to identify patterns, trends, and anomalies
3. Create visualizations where helpful
4. Identify root causes and key factors
5. Provide a comprehensive summary of your findings

Be thorough and quantitative - include specific numbers, percentages, and statistical measures in your analysis."""


def _read_workspace_output(workspace_dir: str) -> str:
    """Read output files from workspace when model_response is empty or invalid."""
    if not os.path.isdir(workspace_dir):
        return ""
    # Look for generated report/output files (exclude inputs/ directory)
    output_extensions = ('.md', '.txt', '.csv', '.json')
    candidates = []
    for item in os.listdir(workspace_dir):
        item_path = os.path.join(workspace_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith(output_extensions):
            candidates.append(item_path)
    # Also check outputs/ or output_report/ subdirectories
    for subdir in ['outputs', 'output_report', 'output']:
        sub_path = os.path.join(workspace_dir, subdir)
        if os.path.isdir(sub_path):
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if os.path.isfile(item_path) and item.lower().endswith(output_extensions):
                    candidates.append(item_path)
    if not candidates:
        return ""
    # Prefer .md files, then .txt, then others; pick the largest
    candidates.sort(key=lambda p: (
        0 if p.endswith('.md') else 1 if p.endswith('.txt') else 2,
        -os.path.getsize(p)
    ))
    try:
        with open(candidates[0], 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""


def process_row(row: dict, agent, prompt_path: str = None, get_sys_msg_func=None, need_info: bool = False, output_base_path: str = None) -> dict:
    question = row.get('question', '')
    file_path = row.get('input_file', '')
    task_id = row.get('id', 'unknown')

    if prompt_path and get_sys_msg_func:
        try:
            sys_content = get_sys_msg_func(prompt_path, question)
        except Exception:
            sys_content = DEFAULT_SYSTEM_PROMPT
    else:
        sys_content = DEFAULT_SYSTEM_PROMPT

    real_file_path_list = []
    if file_path:
        pattern = r'[\n;；]+'
        file_list = [f.strip() for f in re.split(pattern, str(file_path).strip()) if f.strip()]

        for fname in file_list:
            if not os.path.isabs(fname):
                path_input_id = os.path.join(agent.data_root_path, "input", str(task_id), fname)
                path_with_id = os.path.join(agent.data_root_path, str(task_id), fname)
                fallback = os.path.join(agent.data_root_path, fname)
                if os.path.exists(path_input_id):
                    real_file_path_list.append(path_input_id)
                elif os.path.exists(path_with_id):
                    real_file_path_list.append(path_with_id)
                elif os.path.exists(fallback):
                    real_file_path_list.append(fallback)
            else:
                if os.path.exists(fname):
                    real_file_path_list.append(fname)

    toolkit = None
    run_code = None
    if HermesAgent and isinstance(agent, HermesAgent):
        run_code = None
    elif CodeExecutionToolkit:
        toolkit = CodeExecutionToolkit(sandbox="jupyter", namespace=f"task_{task_id}", timeout=60)
        run_code = toolkit.get_tools()[0]
    else:
        def run_code(code, **kwargs):
            return "Execution Environment Not Available"

    try:
        path_info = {}
        mnt_dir_path = '/mnt/data'

        if real_file_path_list:
            file_paths_str = ", ".join([os.path.join(mnt_dir_path, os.path.basename(f)) for f in real_file_path_list])
            question = f"{question}\n\nThe data file(s) are located at: {file_paths_str}"
            first_file_dir = os.path.dirname(real_file_path_list[0])
            path_info = {
                'real_input_dir': first_file_dir,
                'mnt_input_dir': mnt_dir_path,
                'task_id': str(task_id)
            }
        else:
            path_info = {
                'real_input_dir': agent.data_root_path,
                'mnt_input_dir': mnt_dir_path,
                'task_id': str(task_id)
            }

        # Add round limit reminder
        max_rounds = getattr(agent, 'max_rounds', 60)
        question += f"\n\n(注意：你最多有 {max_rounds} 轮对话机会，请合理规划步骤，确保在轮次用完之前输出最终结果。你只能在当前所在目录下进行工作，数据文件已在当前目录中，请勿cd到其他目录。)"

        if output_base_path:
            path_info['workspace_dir'] = os.path.join(os.path.dirname(output_base_path), 'workspace', str(task_id))

        interaction_result = agent.interact(
            query=question,
            system_prompt=sys_content,
            run_code_func=run_code,
            path_info=path_info
        )

        # If model_response is empty/polluted, or workspace has a richer output file, use that
        model_resp = interaction_result.get('model_response', '')
        workspace_dir = path_info.get('workspace_dir')
        if workspace_dir:
            output_content = _read_workspace_output(workspace_dir)
            if output_content and (
                not model_resp.strip()
                or 'SUDO PASSWORD' in model_resp.upper()
                or len(output_content) > len(model_resp) * 2
            ):
                interaction_result['model_response'] = output_content

        # Fallback: if response is still empty/useless and we have a session, resume to extract summary
        model_resp = interaction_result.get('model_response', '')
        _is_useless = (
            not model_resp.strip()
            or 'SUDO PASSWORD' in model_resp.upper()
            or model_resp.strip().startswith('Hello')
            or len(model_resp.strip()) < 100
        )
        session_id = interaction_result.get('session_id')
        if _is_useless and session_id and HermesAgent and isinstance(agent, HermesAgent):
            summary_prompt = (
                "请总结你刚才对数据的所有分析结果。列出你发现的所有关键模式、趋势、异常和洞察。"
                "请用结构化的方式呈现，包含具体数字和百分比。"
            )
            work_dir = path_info.get('workspace_dir') or agent.data_root_path
            summary_result = agent.continue_session(
                session_id=session_id,
                query=summary_prompt,
                work_dir=work_dir,
            )
            summary_resp = summary_result.get('model_response', '')
            if summary_resp and len(summary_resp) > len(model_resp):
                interaction_result['model_response'] = summary_resp

        result = row.copy()
        result.update(interaction_result)
        return result
    finally:
        if toolkit:
            toolkit.reset_session()


def run(args):
    if args.data_path:
        data_path = args.data_path
        dataset_name = args.dataset or "open_ended_test"
    else:
        dataset_name = args.dataset
        rel_path = None
        if DATASET_INFO and dataset_name in DATASET_INFO:
            rel_path = DATASET_INFO[dataset_name].get("file_path")
        if not rel_path:
            rel_path = os.path.join("open_ended_test", "open_ended_test.jsonl")
        data_path = os.path.join(args.data_root, rel_path)

    if args.output_path:
        OUTPUT_PATH = os.path.abspath(args.output_path)
    else:
        save_name = getattr(args, 'save_name', None) or args.model_name
        OUTPUT_PATH = os.path.abspath(os.path.join("output", "preds", save_name, dataset_name, "conv"))

    print(f"Starting Open-Ended Inference with Agent...")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Data Path: {data_path}")
    print(f"  Output Path: {OUTPUT_PATH}")
    print(f"  Concurrency: {args.num_workers}")
    print(f"  Max Rounds: {getattr(args, 'max_rounds', 20)}")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    agent_data_root = args.data_root
    if DATASET_INFO and dataset_name in DATASET_INFO:
        rel_root = DATASET_INFO[dataset_name].get("data_root_path")
        if rel_root:
            agent_data_root = os.path.join(args.data_root, rel_root)

    agent_class = OpenAIJupyterAgent
    use_hermes = False
    if hasattr(args, 'agent_type'):
        if args.agent_type in ('hermes_agent', 'hermes'):
            agent_class = HermesAgent
            use_hermes = True
        elif args.agent_type == 'skill_jupyter_agent':
            agent_class = SkillJupyterAgent
        elif args.agent_type == 'lightllm_jupyter_agent':
            agent_class = LightLLMJupyterAgent
        elif args.agent_type == 'xhx_pipeline':
            agent_class = XHXPipelineAgent
        elif 'jupyter' in args.agent_type:
            agent_class = OpenAIJupyterAgent

    if agent_class is None:
        print(f"Error: Agent class for '{args.agent_type}' not found.")
        sys.exit(1)

    print(f"  Agent: {agent_class.__name__}")

    if use_hermes:
        agent = agent_class(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            data_root_path=agent_data_root,
            save_name=getattr(args, 'save_name', None) or args.model_name,
            max_rounds=getattr(args, 'max_rounds', 90),
        )
    elif XHXPipelineAgent and agent_class is XHXPipelineAgent:
        agent = agent_class(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            data_root_path=agent_data_root,
            max_rounds=getattr(args, 'max_rounds', 20),
            raccoon_project_uuid=getattr(args, 'raccoon_project_uuid', ''),
            enable_web_search=getattr(args, 'enable_web_search', False),
            deep_think=getattr(args, 'deep_think', False),
        )
    else:
        agent = agent_class(
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            data_root_path=agent_data_root,
            max_rounds=getattr(args, 'max_rounds', 20),
            enable_thinking=getattr(args, 'enable_thinking', None),
            temperature=getattr(args, 'temperature', 0.0),
            top_p=getattr(args, 'top_p', 1.0)
        )

    prompt_path = None
    if hasattr(args, 'prompt_file') and args.prompt_file:
        if os.path.isabs(args.prompt_file):
            prompt_path = args.prompt_file
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_path = os.path.join(os.path.dirname(current_dir), "prompts", args.prompt_file)
            prompt_path = possible_path if os.path.exists(possible_path) else args.prompt_file

    runner = InferenceRunner(num_workers=args.num_workers)
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
            'output_base_path': OUTPUT_PATH
        }
    )
