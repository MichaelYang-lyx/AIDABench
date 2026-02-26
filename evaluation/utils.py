import json
import re
from typing import Tuple, List, Dict

def extract_json_from_response(response: str) -> Tuple[int, str]:
    """Extract score and reason from a JSON-formatted LLM response."""
    try:
        judgement = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if judgement:
            answer = judgement.group(1)
            content = json.loads(answer)
            score = content.get('分数', 0)
            if score == 0:
                score = content.get('score', 0)
            
            reason = content.get('得分原因', "")
            if not reason:
                reason = content.get('reason', "")
        else:
            # Fallback for simple JSON without markdown code blocks
            try:
                # Try to find just the JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    content = json.loads(json_match.group(0))
                else:
                    content = json.loads(response)
                    
                score = content.get('分数', 0)
                if score == 0:
                    score = content.get('score', 0)
                    
                reason = content.get('得分原因', "")
                if not reason:
                    reason = content.get('reason', "")
            except:
                score = 0
                reason = response
    except Exception:
        score = 0
        reason = response
    
    # Normalize score to int if possible
    try:
        score = int(score)
    except:
        pass
        
    return score, reason

def load_dataset(data_path: str) -> List[Dict]:
    """Load dataset from a JSON file, JSONL file, or directory of JSON files."""
    import os
    
    if os.path.isdir(data_path):
        data = []
        for fname in os.listdir(data_path):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(data_path, fname), 'r', encoding='utf-8') as f:
                        data.append(json.load(f))
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
        return data
        
    # File handling
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        return data
    else:
        # Assume standard JSON
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
