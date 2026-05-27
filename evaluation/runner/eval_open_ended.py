import os
import sys
import json
import re
import csv
import time
import concurrent.futures
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from consensus.rubric_generator import RubricGenerator
from consensus.consensus_extractor import ConsensusExtractor
from consensus.llm_judge import LLMJudge
from evaluation.utils import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_json_response(content: str) -> Any:
    """Parse JSON from LLM response, handling markdown code blocks."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def load_reference_models_config(config_path: str = "configs/reference_models.json") -> Dict:
    """Load reference models configuration."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, 'r') as f:
        return json.load(f)


def load_ground_truth_insights(task_config_path: str = "data/open_ended_test/test_tasks.json") -> Dict[str, Dict]:
    """Load task config (queries, metadata, csv paths) for test tasks."""
    if not os.path.isabs(task_config_path):
        task_config_path = os.path.join(PROJECT_ROOT, task_config_path)

    with open(task_config_path, 'r') as f:
        tasks = json.load(f)

    ground_truth_map = {}
    for task in tasks:
        task_id = task.get('task_id')
        ground_truth_map[task_id] = {
            "query": task.get('query', ''),
            "metadata": task.get('metadata', {}),
            "dataset_csv_path": task.get('dataset_csv_path', '')
        }

    return ground_truth_map


def get_data_description(task_id: str, ground_truth_map: Dict) -> str:
    """Get data description from metadata and CSV header."""
    task_info = ground_truth_map.get(task_id, {})
    metadata = task_info.get('metadata', {})
    csv_path = task_info.get('dataset_csv_path', '')

    parts = []
    if metadata.get('dataset_description'):
        parts.append(f"Description: {metadata['dataset_description']}")
    if metadata.get('category'):
        parts.append(f"Category: {metadata['category']}")
    if metadata.get('goal'):
        parts.append(f"Goal: {metadata['goal']}")

    if csv_path:
        full_csv_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
        if os.path.exists(full_csv_path):
            try:
                with open(full_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= 6:
                            break
                        rows.append(row)
                if rows:
                    parts.append(f"Columns: {', '.join(rows[0])}")
                    if len(rows) > 1:
                        parts.append(f"Sample rows:\n" + "\n".join([", ".join(r) for r in rows[1:4]]))
            except Exception:
                pass

    return "\n".join(parts) if parts else "No data description available."


# =============================================================================
# Cache helpers
# =============================================================================

def get_task_cache_dir(cache_path: str, task_id: str) -> str:
    return os.path.join(cache_path, task_id)


def get_model_cache_dir(cache_path: str, task_id: str, model_name: str) -> str:
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    return os.path.join(get_task_cache_dir(cache_path, task_id), safe_name)


def cache_model_response(cache_path: str, task_id: str, model_name: str,
                         response_data: Dict, trace_data: Dict, data_files: List[str]):
    """Cache a single reference model's response and trace."""
    model_dir = get_model_cache_dir(cache_path, task_id, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "response.json"), 'w', encoding='utf-8') as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "trace_infer.json"), 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, ensure_ascii=False, indent=2)


def load_cached_model_response(cache_path: str, task_id: str, model_name: str) -> Dict:
    """Load cached response for a model. Returns None if not cached."""
    model_dir = get_model_cache_dir(cache_path, task_id, model_name)
    resp_path = os.path.join(model_dir, "response.json")
    if os.path.exists(resp_path):
        with open(resp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def cache_consensus_findings(cache_path: str, task_id: str,
                             consensus_findings: List, non_consensus_findings: List):
    """Cache consensus findings only (before rubric generation)."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    os.makedirs(task_dir, exist_ok=True)

    with open(os.path.join(task_dir, "consensus_findings.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "consensus_findings": consensus_findings,
            "non_consensus_findings": non_consensus_findings
        }, f, ensure_ascii=False, indent=2)


def cache_rubric(cache_path: str, task_id: str, rubric: Dict):
    """Cache rubric only (after rubric generation)."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    os.makedirs(task_dir, exist_ok=True)

    with open(os.path.join(task_dir, "rubric.json"), 'w', encoding='utf-8') as f:
        json.dump(rubric, f, ensure_ascii=False, indent=2)


def cache_consensus_and_rubric(cache_path: str, task_id: str,
                               consensus_findings: List, non_consensus_findings: List, rubric: Dict):
    """Cache consensus findings and rubric at task level."""
    cache_consensus_findings(cache_path, task_id, consensus_findings, non_consensus_findings)
    cache_rubric(cache_path, task_id, rubric)


def load_cached_consensus_and_rubric(cache_path: str, task_id: str) -> Tuple:
    """Load cached consensus and rubric. Returns (consensus, non_consensus, rubric) or (None, None, None)."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    consensus_path = os.path.join(task_dir, "consensus_findings.json")
    rubric_path = os.path.join(task_dir, "rubric.json")

    if os.path.exists(consensus_path) and os.path.exists(rubric_path):
        with open(consensus_path, 'r', encoding='utf-8') as f:
            c_data = json.load(f)
        with open(rubric_path, 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        return c_data.get("consensus_findings", []), c_data.get("non_consensus_findings", []), rubric

    return None, None, None


# =============================================================================
# L3 Cross-validation
# =============================================================================

def cross_validate_findings(
    non_consensus_findings: List[Dict],
    cache_path: str,
    task_id: str,
    config: Dict,
    language: str = "zh",
) -> List[Dict]:
    """Cross-validate non-consensus findings by resuming other models' Hermes sessions.

    For each finding (proposed by model A), resume models B and C's sessions
    and ask them to score (0-5) whether the finding is factually correct and valuable.
    Returns findings with average score >= 3, sorted by score descending.
    """
    from agents.hermes_docker_agent import HermesDockerAgent as HermesAgent

    if not non_consensus_findings:
        return []

    # Load trace data for each reference model to get session_id and workspace
    model_sessions = {}
    task_cache_dir = get_task_cache_dir(cache_path, task_id)
    if not os.path.isdir(task_cache_dir):
        print(f"    Warning: Task cache dir not found: {task_cache_dir}")
        return []

    for item in os.listdir(task_cache_dir):
        trace_path = os.path.join(task_cache_dir, item, "trace_infer.json")
        if not os.path.isfile(trace_path):
            trace_path = os.path.join(task_cache_dir, item, "trace.json")
        if os.path.isfile(trace_path):
            try:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    trace = json.load(f)
                session_id = trace.get("hermes_session_id")
                workspace = os.path.join(task_cache_dir, item, "workspace")
                model_name = trace.get("model_name", item)
                model_id = trace.get("model_id", "")
                api_url = trace.get("api_url", "")
                if session_id and os.path.isdir(workspace):
                    # Find api_key from config
                    api_key = ""
                    provider = None
                    for m in config.get("models", []):
                        if m["name"] == model_name:
                            api_key = m["api_key"]
                            provider = m.get("provider")
                            break
                    # interact() creates profile as: f"{save_name}_{task_id}" where save_name = f"ref_{task_id}_{safe_model_id}"
                    safe_model_id = model_id.replace('/', '_').replace(' ', '_').replace('.', '-').lower()
                    fallback_profile = re.sub(r'[^a-z0-9_-]', '-', f"ref_{task_id}_{safe_model_id}_{task_id}".lower())[:64]
                    hermes_profile = trace.get("hermes_profile") or fallback_profile
                    # docker 模式：从 trace 提取消息历史作为 prefill 注入新容器
                    history_messages = []
                    fr_history = trace.get("first_round_history") or []
                    if isinstance(fr_history, list) and fr_history:
                        elem = fr_history[0]
                        if isinstance(elem, dict) and isinstance(elem.get("messages"), list):
                            history_messages = elem["messages"]
                        elif isinstance(elem, dict) and elem.get("role"):
                            history_messages = fr_history
                    model_sessions[model_name] = {
                        "session_id": session_id,
                        "workspace": workspace,
                        "model_id": model_id,
                        "api_url": api_url,
                        "api_key": api_key,
                        "provider": provider,
                        "profile": hermes_profile,
                        "messages_history": history_messages,
                    }
            except Exception:
                continue

    if len(model_sessions) < 2:
        print(f"    Warning: Need at least 2 model sessions for cross-validation, found {len(model_sessions)}")
        return []

    print(f"    Loaded {len(model_sessions)} model sessions for cross-validation")

    def _build_batch_prompt(batch_findings: List[Dict], lang: str) -> str:
        n = len(batch_findings)
        items = []
        for idx, fnd in enumerate(batch_findings, start=1):
            patt = fnd.get("pattern", "")
            ev_list = fnd.get("evidence", [])
            ev_str = "\n".join(str(e) for e in ev_list if e) or "N/A"
            if lang == "zh":
                items.append(f"## 发现 {idx}：\n{patt}\n\n### 证据 {idx}：\n{ev_str}")
            else:
                items.append(f"## Finding {idx}:\n{patt}\n\n### Evidence {idx}:\n{ev_str}")
        body = "\n\n".join(items)

        if lang == "zh":
            return f"""你之前已经分析过这份数据。现在请评估以下 {n} 个发现是否正确且有价值。
你可以编写代码验证这些发现的数值是否准确。

{body}

请按以下标准对每个发现打分（0-5分）并简要说明理由：
- 5分：发现完全正确，数值准确，且揭示了非显而易见的重要洞察
- 4分：发现正确，数值基本准确，具有一定分析价值
- 3分：发现大致正确，但数值有小幅偏差或结论较为常规
- 2分：发现部分正确，但存在明显的数值错误或逻辑漏洞
- 1分：发现大部分不正确，或证据严重不足
- 0分：发现完全错误，或与数据无关

请严格按以下 JSON 数组格式输出 {n} 个评分（顺序与上面一致，不要输出其他内容）：
```json
[
  {{"index": 1, "score": X, "reason": "..."}},
  {{"index": 2, "score": X, "reason": "..."}}
]
```"""
        else:
            return f"""You have already analyzed this dataset. Now evaluate whether the following {n} findings are correct and valuable.
You may write code to verify the numerical claims.

{body}

Score each finding using the following rubric and briefly explain:
- 5: Completely correct, numerically accurate, reveals a non-obvious and important insight
- 4: Correct, numerically sound, provides meaningful analytical value
- 3: Mostly correct, minor numerical deviations or somewhat routine conclusion
- 2: Partially correct, but has notable numerical errors or logical gaps
- 1: Mostly incorrect, or severely lacking in supporting evidence
- 0: Entirely wrong, or unrelated to the data

Output strictly as a JSON array of {n} scores (in the same order as above, no other content):
```json
[
  {{"index": 1, "score": X, "reason": "..."}},
  {{"index": 2, "score": X, "reason": "..."}}
]
```"""

    # Pre-compute per-model finding indices (skip findings the validator itself proposed)
    BATCH_SIZE = 5
    per_model_indices: Dict[str, List[int]] = {mname: [] for mname in model_sessions}
    for fi, finding in enumerate(non_consensus_findings):
        source_models = finding.get("models", [])
        for mname in model_sessions:
            if mname not in source_models:
                per_model_indices[mname].append(fi)

    # finding_index -> list of (model_name, score, raw_response)
    per_finding_validations: Dict[int, List[Dict]] = {fi: [] for fi in range(len(non_consensus_findings))}
    cv_traces = {mname: [] for mname in model_sessions}

    total_findings = len(non_consensus_findings)
    for mname, minfo in model_sessions.items():
        indices = per_model_indices[mname]
        if not indices:
            continue
        n_batches = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE
        for bi in range(n_batches):
            batch_idx_slice = indices[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
            batch_findings = [non_consensus_findings[i] for i in batch_idx_slice]
            n = len(batch_findings)
            prompt = _build_batch_prompt(batch_findings, language)

            print(f"      [{mname}] batch {bi+1}/{n_batches}: validating {n} findings (orig idx {batch_idx_slice})...")

            agent = HermesAgent(
                api_key=minfo["api_key"],
                base_url=minfo["api_url"],
                model_name=minfo["model_id"],
                data_root_path=minfo["workspace"],
                save_name=minfo["profile"],
                max_rounds=10,
                provider=minfo.get("provider"),
            )

            result = agent.continue_session(
                session_id=minfo["session_id"],
                query=prompt,
                work_dir=minfo["workspace"],
                messages_history=minfo.get("messages_history") or [],
            )

            resp = result.get("model_response", "")
            scores = _parse_batch_validation_scores(resp, n)
            if all(s == 0 for s in scores) and resp:
                print(f"        [DEBUG] {mname} batch response (first 300): {resp[:300]}")
            elif not resp:
                print(f"        [DEBUG] {mname} returned empty response for batch")

            for local_i, fi in enumerate(batch_idx_slice):
                score = scores[local_i] if local_i < len(scores) else 0.0
                per_finding_validations[fi].append({
                    "model": mname,
                    "score": score,
                    "raw_response": resp[:500],
                })
                cv_traces[mname].append({
                    "finding_index": fi,
                    "batch_index": bi,
                    "batch_position": local_i,
                    "pattern": non_consensus_findings[fi].get("pattern", ""),
                    "score": score,
                    "session_id": minfo["session_id"],
                })
            # One trace entry per batch for full prompt/response
            cv_traces[mname].append({
                "batch_index": bi,
                "batch_finding_indices": batch_idx_slice,
                "prompt": prompt,
                "response": resp,
                "session_id": minfo["session_id"],
                "history": result.get("history", []),
            })

    validated = []
    for fi, finding in enumerate(non_consensus_findings):
        pattern = finding.get("pattern", "")
        evidence_list = finding.get("evidence", [])
        source_models = finding.get("models", [])
        validations = per_finding_validations[fi]
        scores_for_finding = [v["score"] for v in validations]
        avg_score = sum(scores_for_finding) / len(scores_for_finding) if scores_for_finding else 0

        validated.append({
            "pattern": pattern,
            "evidence": evidence_list,
            "source_models": source_models,
            "avg_score": avg_score,
            "validations": validations,
        })

        print(f"      [{fi+1}/{total_findings}] \"{pattern[:50]}...\" avg_score={avg_score:.1f}")

    # Save per-model cross-validation traces
    for mname, traces in cv_traces.items():
        if not traces:
            continue
        model_dir = get_model_cache_dir(cache_path, task_id, mname)
        if os.path.isdir(model_dir):
            trace_cv_path = os.path.join(model_dir, "trace_cross_validation.json")
            with open(trace_cv_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "model_name": mname,
                    "session_id": model_sessions[mname]["session_id"],
                    "num_validations": len(traces),
                    "interactions": traces,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, ensure_ascii=False, indent=2)

    # Filter findings with avg_score >= 3, sort by score descending
    validated.sort(key=lambda x: x["avg_score"], reverse=True)
    qualified = [v for v in validated if v["avg_score"] >= 3.0]

    print(f"    Cross-validation complete: {len(validated)} findings evaluated, {len(qualified)} qualified (avg_score >= 3)")
    return qualified


def _parse_validation_score(response: str) -> float:
    """Parse a 0-5 score from the validation response."""
    import re as _re
    # Try JSON parse
    try:
        match = _re.search(r'```(?:json)?\s*(.*?)```', response, _re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return float(data.get("score", 0))
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        match = _re.search(r'\{[^{}]*"score"\s*:\s*(\d+(?:\.\d+)?)[^{}]*\}', response)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass

    # Fallback: look for "X/5" or "score: X"
    try:
        match = _re.search(r'(\d+(?:\.\d+)?)\s*/\s*5', response)
        if match:
            return float(match.group(1))
        match = _re.search(r'[Ss]core\s*[:=：]\s*(\d+(?:\.\d+)?)', response)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass

    return 0.0


def _parse_batch_validation_scores(response: str, expected_n: int) -> List[float]:
    """Parse a JSON array of {index, score, reason} entries.

    Returns a list of length expected_n. Missing/unparseable slots default to 0.0.
    Handles common deviations: array wrapped in ```json``` block, single dict
    instead of array (n==1), or score-only objects without explicit index.
    """
    import re as _re

    scores = [0.0] * expected_n

    def _apply(items):
        """Populate `scores` from a list of dicts using 1-based 'index' if present, else position."""
        for pos, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                s = float(item.get("score", 0))
            except (ValueError, TypeError):
                s = 0.0
            idx = item.get("index")
            if isinstance(idx, (int, float)) and 1 <= int(idx) <= expected_n:
                scores[int(idx) - 1] = s
            elif pos < expected_n:
                scores[pos] = s

    # Try fenced JSON block first
    try:
        match = _re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, _re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                _apply(data)
                return scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Try raw JSON array anywhere in the response
    try:
        match = _re.search(r'\[\s*\{.*?\}\s*\]', response, _re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                _apply(data)
                return scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Fenced JSON containing a single object (n==1 case)
    if expected_n == 1:
        try:
            match = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, _re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    _apply([data])
                    return scores
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: enumerate "score": X occurrences in order
    matches = _re.findall(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response)
    for pos, m in enumerate(matches[:expected_n]):
        try:
            scores[pos] = float(m)
        except ValueError:
            pass

    return scores


def cache_cross_validation(cache_path: str, task_id: str, validated_findings: List[Dict]):
    """Cache cross-validation results."""
    task_dir = get_task_cache_dir(cache_path, task_id)
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, "cross_validation.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(validated_findings, f, ensure_ascii=False, indent=2)


def load_cached_cross_validation(cache_path: str, task_id: str) -> List[Dict]:
    """Load cached cross-validation results. Returns None if not cached."""
    path = os.path.join(get_task_cache_dir(cache_path, task_id), "cross_validation.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# =============================================================================
# Reference model execution
# =============================================================================

def run_reference_models(query: str, data_description: str, config: Dict,
                         cache_path: str, task_id: str, data_files: List[str],
                         language: str = "zh", max_workers: int = 1) -> List[Dict]:
    """Run reference models with hermes agent, then extract insights in second round.

    Args:
        max_workers: Number of parallel workers for running reference models (default: 1 for serial)
    """
    from agents.hermes_docker_agent import HermesDockerAgent as HermesAgent

    def _run_single_model(model: Dict) -> Dict:
        model_name = model['name']

        # Check cache first
        cached = load_cached_model_response(cache_path, task_id, model_name)
        if cached is not None:
            print(f"    [Cache HIT] {model_name}")
            return cached

        print(f"    [Cache MISS] Running {model_name} with hermes...")
        t_start = time.time()
        try:
            safe_model_name = model['model_id'].replace('/', '_').replace(' ', '_').replace('.', '-').lower()
            profile_name = f"ref_{task_id}_{safe_model_name}"[:64]

            model_dir = os.path.join(cache_path, task_id, model_name.replace('/', '_').replace(' ', '_'))
            os.makedirs(model_dir, exist_ok=True)
            data_input_dir = os.path.dirname(data_files[0]) if data_files else model_dir

            agent = HermesAgent(
                api_key=model['api_key'],
                base_url=model['api_url'],
                model_name=model['model_id'],
                data_root_path=model_dir,
                save_name=profile_name,
                max_rounds=model.get('max_rounds', 20),
                provider=model.get('provider'),
            )

            path_info = {
                "task_id": task_id,
                "real_input_dir": data_input_dir,
                "real_output_dir": os.path.join(model_dir, "outputs"),
                "mnt_input_dir": "./inputs",
                "mnt_output_dir": "./outputs",
                "workspace_dir": os.path.join(model_dir, "workspace")
            }

            file_names = [os.path.basename(f) for f in data_files]
            files_str = ", ".join(file_names) if file_names else "the provided data"
            hermes_query = f"{query}\n\nThe data file(s) are located at: ./inputs/{files_str}"

            hermes_result = agent.interact(
                query=hermes_query,
                system_prompt="You are an expert data analyst. Analyze the dataset thoroughly.",
                run_code_func=None,
                path_info=path_info
            )

            first_round_response = hermes_result.get('model_response', '')
            session_id = hermes_result.get('session_id', '')

            if language == "zh":
                insight_extraction_prompt = (
                    "请基于你上面的分析，总结所有关键发现、规律、趋势和洞察。"
                    "请以要点形式清晰简洁地列出。"
                    "重点关注可操作的洞察和数据中的重要规律。"
                    "同时请列出你生成的关键交付产物（如图表、报告等），并简要说明其内容。\n\n"
                    "总条数最多为15条。"
                    "输出格式示例：\n"
                    "1. 关键发现1：[具体发现内容]\n"
                    "2. 关键发现2：[规律描述]\n"
                    "3. 规律1：[洞察内容]\n"
                    "...\n"
                    "15. 交付产物3：[产物名称] - [简要说明]"
                )
            else:
                insight_extraction_prompt = (
                    "Based on your analysis above, please summarize all key findings, patterns, trends, "
                    "and insights you discovered. List them clearly and concisely as bullet points. "
                    "Focus on actionable insights and important patterns in the data. "
                    "Also list the key deliverables you produced (e.g. charts, reports) with a brief description.\n\n"
                    "Output format example:\n"
                    "1. Key Finding: [specific finding]\n"
                    "2. Important Pattern: [pattern description]\n"
                    "3. Actionable Insight: [insight content]\n"
                    "4. Deliverable: [name] - [brief description]"
                )

            if not session_id:
                print(f"    Warning: No session_id captured for {model_name}, falling back to combined query")
                combined_query = hermes_query + "\n\n" + insight_extraction_prompt
                fallback_result = agent.interact(
                    query=combined_query,
                    system_prompt="You are an expert data analyst. Analyze the dataset thoroughly.",
                    run_code_func=None,
                    path_info=path_info
                )
                t_end = time.time()
                insights_response = fallback_result.get('model_response', '')
                insight_result = {'history': fallback_result.get('history', [])}
            else:
                work_dir = os.path.join(model_dir, "workspace")
                # docker 模式：把上一轮的消息历史作为 prefill 注入新容器
                first_round_messages = hermes_result.get('history') or []
                insight_result = agent.continue_session(
                    session_id=session_id,
                    query=insight_extraction_prompt,
                    work_dir=work_dir,
                    profile=hermes_result.get('profile'),
                    messages_history=first_round_messages,
                )

                t_end = time.time()
                insights_response = insight_result.get('model_response', '')
                if insights_response.startswith('↻'):
                    insights_response = '\n'.join(insights_response.split('\n')[1:]).strip()

            resp_data = {
                "model_name": model_name,
                "model_response": insights_response
            }

            trace_data = {
                "model_name": model_name,
                "model_id": model['model_id'],
                "api_url": model['api_url'],
                "first_round_query": hermes_query,
                "first_round_response": first_round_response,
                "first_round_history": hermes_result.get('history', []),
                "hermes_session_id": session_id,
                "hermes_profile": hermes_result.get('profile', ''),
                "second_round_prompt": insight_extraction_prompt,
                "insights_response": insights_response,
                "second_round_history": insight_result.get('history', []),
                "elapsed_seconds": round(t_end - t_start, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            cache_model_response(cache_path, task_id, model_name, resp_data, trace_data, data_files)
            return resp_data

        except Exception as e:
            t_end = time.time()
            print(f"    Warning: Reference model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

            resp_data = {"model_name": model_name, "model_response": "", "error": str(e)}
            trace_data = {
                "model_name": model_name,
                "error": str(e),
                "elapsed_seconds": round(t_end - t_start, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            cache_model_response(cache_path, task_id, model_name, resp_data, trace_data, data_files)
            return resp_data

    models = config['models']
    if max_workers <= 1 or len(models) <= 1:
        return [_run_single_model(m) for m in models]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_model, m): m['name'] for m in models}
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"    Warning: Reference model {model_name} unexpected error: {e}")
                results.append({"model_name": model_name, "model_response": "", "error": str(e)})
    return results


# =============================================================================
# Build reference cache command
# =============================================================================

def build_reference_cache(args):
    """Build reference cache (consensus + rubric) for all tasks."""
    config = load_reference_models_config(args.config_path)
    ground_truth_map = load_ground_truth_insights(args.task_config)

    dataset_name = getattr(args, 'dataset', 'open_ended_test')
    cache_path = getattr(args, 'cache_path', None)
    if not cache_path:
        cache_path = os.path.join(PROJECT_ROOT, "output", "reference_cache", dataset_name)

    os.makedirs(cache_path, exist_ok=True)
    print(f"Building reference cache at: {cache_path}")
    print(f"Tasks to process: {len(ground_truth_map)}")

    max_workers = getattr(args, 'max_workers', 1)

    def process_single_cache_task(task_id, task_info, idx, total):
        print(f"\n[{idx+1}/{total}] Processing task: {task_id}")

        # Check if already cached
        _, _, rubric = load_cached_consensus_and_rubric(cache_path, task_id)
        if rubric is not None:
            print(f"  Task {task_id} already has cached rubric, skipping.")
            return

        query = task_info.get('query', '')

        # Step 1: Get data description
        print(f"  [{task_id}] Getting data description...")
        data_description = get_data_description(task_id, ground_truth_map)

        # Resolve data files
        csv_path = task_info.get('dataset_csv_path', '')
        data_files = []
        if csv_path:
            full_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
            if os.path.exists(full_path):
                parent_dir = os.path.dirname(full_path)
                data_files = [
                    os.path.join(parent_dir, f) for f in sorted(os.listdir(parent_dir))
                    if os.path.isfile(os.path.join(parent_dir, f))
                ]

        # Step 2: Run reference models
        print(f"  [{task_id}] Running {len(config['models'])} reference models...")
        reference_responses = run_reference_models(
            query, data_description, config, cache_path, task_id, data_files,
            language=getattr(args, 'language', 'zh'),
            max_workers=getattr(args, 'ref_max_workers', 1)
        )
        valid_responses = [r for r in reference_responses if r.get('model_response')]
        print(f"  [{task_id}] Got {len(valid_responses)}/{len(reference_responses)} valid responses")

        # Step 3: Extract consensus
        print(f"  [{task_id}] Extracting consensus findings...")
        consensus_cfg = config.get('consensus_model', config['judge_model'])
        extractor = ConsensusExtractor(
            api_key=consensus_cfg['api_key'],
            base_url=consensus_cfg['api_url'],
            model_name=consensus_cfg['model_id'],
            consensus_threshold=config.get('consensus_threshold', 0.6),
            language=getattr(args, 'language', 'zh'),
        )
        consensus_findings, non_consensus_findings = extractor.extract(valid_responses)
        print(f"  [{task_id}] Found {len(consensus_findings)} consensus, {len(non_consensus_findings)} non-consensus findings")

        # Save consensus findings
        cache_consensus_findings(cache_path, task_id, consensus_findings, non_consensus_findings)

        # Step 4: Cross-validate non-consensus findings
        print(f"  [{task_id}] Cross-validating non-consensus findings...")
        validated_l3 = cross_validate_findings(
            non_consensus_findings=non_consensus_findings,
            cache_path=cache_path,
            task_id=task_id,
            config=config,
            language=getattr(args, 'language', 'zh'),
        )
        print(f"  [{task_id}] Cross-validation: {len(validated_l3)} findings passed")
        cache_cross_validation(cache_path, task_id, validated_l3)

        # Step 5: Generate rubric
        print(f"  [{task_id}] Generating rubric...")
        all_findings = consensus_findings + validated_l3
        rubric_cfg = config.get('rubric_model', config['judge_model'])
        generator = RubricGenerator(
            api_key=rubric_cfg['api_key'],
            base_url=rubric_cfg['api_url'],
            model_name=rubric_cfg['model_id'],
            language=getattr(args, 'language', 'zh'),
        )
        rubric = generator.generate(query, all_findings)
        print(f"  [{task_id}] Generated rubric with {len(rubric)} criteria")
        cache_rubric(cache_path, task_id, rubric)

        print(f"  ✓ Task {task_id} cache complete")

    # Execute tasks
    task_items = list(ground_truth_map.items())
    if max_workers == 1:
        for idx, (task_id, task_info) in enumerate(task_items):
            process_single_cache_task(task_id, task_info, idx, len(task_items))
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_cache_task, task_id, task_info, idx, len(task_items))
                for idx, (task_id, task_info) in enumerate(task_items)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Error processing task: {e}")

    print(f"\n✓ Reference cache build complete: {cache_path}")


# =============================================================================
# Main task processing
# =============================================================================

def process_single_task(row: Dict, i: int, args, config: Dict, ground_truth_map: Dict) -> Dict:
    """Process a single open-ended task using full ConsensusEval pipeline with caching."""
    row.pop('history', None)

    result_path = get_result_file_path(row, i, args)
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if 'score' in existing and 'error' not in existing:
                print(f"  [Task {i+1}] Already evaluated, skipping.")
                return existing
        except (json.JSONDecodeError, IOError):
            pass

    model_response = row.get('answer') or row.get('model_response') or row.get('analysis', '')
    task_id = row.get('id') or row.get('task_id', f"task_{i}")
    query = row.get('query', '')

    task_info = ground_truth_map.get(task_id, {})
    if not query:
        query = task_info.get('query', '')

    dataset_name = getattr(args, 'dataset', 'open_ended_test')
    use_cache = getattr(args, 'use_cache', False)
    cache_path = getattr(args, 'cache_path', None)
    if not cache_path:
        cache_path = os.path.join(PROJECT_ROOT, "output", "reference_cache", dataset_name)

    try:
        # Check if we can use cached consensus + rubric
        consensus_findings, non_consensus_findings, rubric = None, None, None
        if use_cache:
            consensus_findings, non_consensus_findings, rubric = load_cached_consensus_and_rubric(cache_path, task_id)
            if rubric is not None:
                print(f"  [Task {i+1}] Using cached consensus + rubric")

        if rubric is None:
            # Step 1: Get data description
            print(f"  [Task {i+1}] Getting data description...")
            data_description = get_data_description(task_id, ground_truth_map)

            # Resolve data files for caching
            # First try dataset_csv_path from task config, then scan input directory
            csv_path = task_info.get('dataset_csv_path', '')
            data_files = []
            if csv_path:
                full_path = os.path.join(PROJECT_ROOT, csv_path) if not os.path.isabs(csv_path) else csv_path
                if os.path.exists(full_path):
                    # If it points to a file, also include all sibling files in the same directory
                    parent_dir = os.path.dirname(full_path)
                    data_files = [
                        os.path.join(parent_dir, f) for f in sorted(os.listdir(parent_dir))
                        if os.path.isfile(os.path.join(parent_dir, f))
                    ]

            # Step 2: Run reference models (with per-model cache)
            print(f"  [Task {i+1}] Running {len(config['models'])} reference models...")
            reference_responses = run_reference_models(
                query, data_description, config, cache_path, task_id, data_files,
                language=getattr(args, 'language', 'zh'),
                max_workers=getattr(args, 'ref_max_workers', 1)
            )
            valid_responses = [r for r in reference_responses if r.get('model_response')]
            print(f"  [Task {i+1}] Got {len(valid_responses)}/{len(reference_responses)} valid responses")

            # Step 3: Extract consensus
            print(f"  [Task {i+1}] Extracting consensus findings...")
            consensus_cfg = config.get('consensus_model', config['judge_model'])
            extractor = ConsensusExtractor(
                api_key=consensus_cfg['api_key'],
                base_url=consensus_cfg['api_url'],
                model_name=consensus_cfg['model_id'],
                consensus_threshold=config.get('consensus_threshold', 0.6),
                language=getattr(args, 'language', 'zh'),
            )
            consensus_findings, non_consensus_findings = extractor.extract(valid_responses)
            print(f"  [Task {i+1}] Found {len(consensus_findings)} consensus, {len(non_consensus_findings)} non-consensus findings")

            # Save consensus findings immediately (before cross-validation and rubric generation)
            if use_cache:
                cache_consensus_findings(cache_path, task_id, consensus_findings, non_consensus_findings)

            # Step 3.5: Cross-validate non-consensus findings (L3)
            print(f"  [Task {i+1}] Cross-validating non-consensus findings...")
            validated_l3 = None
            if use_cache:
                validated_l3 = load_cached_cross_validation(cache_path, task_id)
                if validated_l3 is not None:
                    print(f"  [Task {i+1}] Using cached cross-validation ({len(validated_l3)} findings)")

            if validated_l3 is None:
                validated_l3 = cross_validate_findings(
                    non_consensus_findings=non_consensus_findings,
                    cache_path=cache_path,
                    task_id=task_id,
                    config=config,
                    language=getattr(args, 'language', 'zh'),
                )
                cache_cross_validation(cache_path, task_id, validated_l3)

            # Step 4: Generate rubric
            print(f"  [Task {i+1}] Generating three-layer rubric...")
            rubric_gen = RubricGenerator(
                api_key=config['judge_model']['api_key'],
                base_url=config['judge_model']['api_url'],
                model_name=config['judge_model']['model_id'],
                language=args.language
            )
            rubric = rubric_gen.generate(
                task_description=query,
                consensus_findings=consensus_findings,
                data_characteristics=task_info.get('metadata'),
                non_consensus_findings=non_consensus_findings,
                validated_l3_findings=validated_l3 if validated_l3 else None,
            )

            # Save rubric to cache (consensus_findings already saved after Step 3)
            if use_cache:
                cache_rubric(cache_path, task_id, rubric)

        print(f"  [Task {i+1}] Rubric: L1={len(rubric.get('layer1_must_find',[]))} items, "
              f"L2={len(rubric.get('layer2_process_quality',[]))} items, "
              f"L3={len(rubric.get('layer3_bonus_discovery',[]))} items")

        # Step 5: Judge the model response
        print(f"  [Task {i+1}] Judging model response...")
        judge = LLMJudge(
            api_key=config['judge_model']['api_key'],
            base_url=config['judge_model']['api_url'],
            model_name=config['judge_model']['model_id'],
            use_hermes=True,
            max_rounds=config.get('judge_max_rounds', 30),
            provider=config['judge_model'].get('provider'),
        )

        # Resolve data files and model workspace for Hermes judge
        judge_data_files = []
        model_workspace_path = None

        # Get original data files
        csv_path_for_judge = task_info.get('dataset_csv_path', '')
        if csv_path_for_judge:
            full_path = os.path.join(PROJECT_ROOT, csv_path_for_judge) if not os.path.isabs(csv_path_for_judge) else csv_path_for_judge
            if os.path.exists(full_path):
                parent_dir = os.path.dirname(full_path)
                judge_data_files = [
                    os.path.join(parent_dir, f) for f in sorted(os.listdir(parent_dir))
                    if os.path.isfile(os.path.join(parent_dir, f))
                ]

        # Try to find model's workspace from infer output
        input_path = args.input_path
        if os.path.isdir(input_path):
            # input_path is like output/preds/{model}/{dataset}/conv
            # workspace is at output/preds/{model}/{dataset}/workspace/{task_id}
            parent_of_conv = os.path.dirname(input_path.rstrip('/'))
            candidate = os.path.join(parent_of_conv, 'workspace', task_id)
            if os.path.isdir(candidate):
                model_workspace_path = candidate

        judge_workspace_dir = os.path.join(args.output_path, "workspace", task_id)

        eval_result = judge.evaluate(
            analysis_output=model_response,
            rubric=rubric,
            task_description=query,
            num_runs=config.get('num_judge_runs', 1),
            data_files=judge_data_files,
            model_output_workspace=model_workspace_path,
            judge_workspace_dir=judge_workspace_dir,
        )

        # Save judge trace
        judge_traces = eval_result.pop('judge_traces', [])
        if judge_traces:
            trace_judge_path = os.path.join(judge_workspace_dir, "trace_judge.json")
            os.makedirs(judge_workspace_dir, exist_ok=True)
            with open(trace_judge_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_id": task_id,
                    "judge_model": config['judge_model']['model_id'],
                    "num_runs": len(judge_traces),
                    "runs": judge_traces,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, ensure_ascii=False, indent=2)
            print(f"  [Task {i+1}] Judge trace saved to: {trace_judge_path}")

        row['score'] = eval_result.get('mean_score', 0)
        row['reason'] = eval_result.get('detailed_feedback', '')
        row['rubric'] = rubric
        row['consensus_findings'] = consensus_findings
        row['non_consensus_findings'] = non_consensus_findings
        row['eval_details'] = eval_result

    except Exception as e:
        import traceback
        print(f"  [Task {i+1}] Error: {str(e)}")
        traceback.print_exc()
        row['score'] = 0
        row['reason'] = f"Evaluation failed: {str(e)}"
        row['error'] = str(e)

    save_result(row, args, i)
    return row


def get_result_file_path(row: Dict, i: int, args) -> str:
    item_id = row.get('id') or row.get('task_id', f"eval_{i}")
    file_name = f"{item_id}.json"
    return os.path.join(args.output_path, file_name)


def save_result(row: Dict, args, i: int):
    file_path = get_result_file_path(row, i, args)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


def run(args):
    """Run open-ended evaluation using ConsensusEval."""
    print("\n" + "="*50)
    print("Starting Open-Ended Evaluation (ConsensusEval)")
    print("="*50)

    config_path = getattr(args, 'config_path', 'configs/reference_models.json')
    print(f"\nLoading reference models config from: {config_path}")
    config = load_reference_models_config(config_path)
    print(f"  - Loaded {len(config['models'])} reference models")
    print(f"  - Judge model: {config['judge_model']['model_id']}")
    print(f"  - Consensus threshold: {config.get('consensus_threshold', 0.6)}")
    print(f"  - Number of judge runs: {config.get('num_judge_runs', 1)}")

    use_cache = getattr(args, 'use_cache', False)
    cache_path = getattr(args, 'cache_path', None)
    if not cache_path:
        dataset_name = getattr(args, 'dataset', 'open_ended_test')
        cache_path = os.path.join(PROJECT_ROOT, "output", "reference_cache", dataset_name)
    print(f"  - Use cache: {use_cache}")
    print(f"  - Cache path: {cache_path}")

    task_config_path = getattr(args, 'task_config', None) or 'data/open_ended_test/test_tasks.json'
    print(f"\nLoading task config from: {task_config_path}")
    ground_truth_map = load_ground_truth_insights(task_config_path)
    print(f"  - Loaded config for {len(ground_truth_map)} tasks")

    os.makedirs(args.output_path, exist_ok=True)

    print(f"\nLoading predictions from: {args.input_path}")
    dataset = load_dataset(args.input_path)
    print(f"  - Found {len(dataset)} tasks to evaluate")

    if len(dataset) == 0:
        print("\nNo tasks found. Exiting.")
        return

    print("\nStarting evaluation...\n")
    max_workers = getattr(args, 'max_workers', 1)
    # Pre-allocate so results stay aligned with `dataset` order regardless of
    # completion order — the `scores` array in summary.json must line up with
    # the task list (callers compare positionally).
    results: List[Any] = [None] * len(dataset)

    if max_workers == 1:
        for i, row in enumerate(tqdm(dataset, desc="Evaluating tasks")):
            results[i] = process_single_task(row, i, args, config, ground_truth_map)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_task, row, i, args, config, ground_truth_map): i
                for i, row in enumerate(dataset)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Evaluating tasks"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    # scores: dict {task_id: score} — explicit keys avoid any positional misalignment
    scores: Dict[str, float] = {}
    for i, r in enumerate(results):
        tid = r.get('id') or r.get('task_id') or f"task_{i}"
        scores[tid] = r.get('score', 0)
    # Sort keys so qa_open_001 → qa_open_064 read in natural order in summary.json
    scores = dict(sorted(scores.items()))
    avg_score = sum(scores.values()) / len(scores) if scores else 0

    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)
    print(f"Total tasks evaluated: {len(results)}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Results saved to: {args.output_path}")
    print("="*50 + "\n")

    summary_path = os.path.join(args.output_path, "summary.json")
    summary = {
        "total_tasks": len(results),
        "average_score": avg_score,
        "scores": scores,
        "config": {
            "reference_models": [m['name'] for m in config['models']],
            "judge_model": config['judge_model']['model_id'],
            "consensus_threshold": config.get('consensus_threshold', 0.6),
            "num_judge_runs": config.get('num_judge_runs', 1),
            "use_cache": use_cache
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run open-ended evaluation")

    # Common arguments
    parser.add_argument("--input_path", default=None, help="Path to model predictions")
    parser.add_argument("--output_path", default=None, help="Path to save evaluation results")
    parser.add_argument("--config_path", default="configs/reference_models.json", help="Path to reference models config")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--use_cache", action="store_true", help="Reuse cached consensus findings and rubrics")
    parser.add_argument("--cache_path", default=None, help="Path for consensus/rubric cache (default: output/reference_cache/{dataset})")
    parser.add_argument("--dataset", default="open_ended_test", help="Dataset name for cache directory")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Language for rubric generation (zh or en)")
    parser.add_argument("--ref_max_workers", type=int, default=1, help="Number of parallel workers for reference models")
    parser.add_argument("--task_config", default="data/open_ended_test/test_tasks.json", help="Path to task config JSON")

    parser.add_argument("command", nargs="?", default="eval", choices=["eval", "build-cache"],
                        help="Command to run (default: eval)")

    args = parser.parse_args()

    if args.command == 'eval':
        if not args.input_path or not args.output_path:
            parser.error("--input_path and --output_path are required for eval")
        args.use_hermes_judge = True
        run(args)
    elif args.command == 'build-cache':
        build_reference_cache(args)
