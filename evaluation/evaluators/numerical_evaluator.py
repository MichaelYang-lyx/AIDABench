import json
from typing import Dict, Any, List
from openai import OpenAI
from .base_evaluator import BaseEvaluator
from ..config import NUMERICAL_EVAL_API_URL, NUMERICAL_EVAL_API_KEY, NUMERICAL_EVAL_MODEL_NAME
from ..utils import extract_json_from_response, load_dataset

PROMPT_TEMPLATE = """你是一个擅长打分的小助手。现在请你给AI模型的答案进行打分。我们会给你提供以下信息：
1. 问题、参考答案
2. 需要打分的AI模型答案

答案的打分标准：分数总共有2档，0分，1分。请根据问题和参考答案，对AI模型的答案进行判断：
1. 统计类题目：若AI模型的计算结果与参考答案基本一致（统计数据需更严格对应），则判定为1分；若计算结果与参考答案存在明显差异，判定为0分。
2. 知识问答与分析类题目：若AI模型答案能完整覆盖参考答案的核心要点，且表述逻辑与原意相符，判定为1分；若答案与参考答案存在显著偏差，核心要点缺失，则判定为0分。

注意：
1. 以上是答案打分的通用标准，但是请注意，有些题目会提供[答题点]，提供了[答题点]的题目，评分时需要考虑[答题点]。
2. 模型的答案可能包含其分析过程，但主要关注最终的结论。
3. 数据可能保留小数不一样，四舍五入后相同也算数据对。
4. 对于表格的对照，需要仔细理解题目后进行对比判断。
5. 只需关注答案逻辑，任何回答语言都能接受（中、英...）

最后，你需要按照以下步骤来进行打分
1. 理解题目和参考答案
2. 根据上面介绍的标准，给AI模型的答案打分，并解释这样打分的原因。
3. 最后将用下面json格式输出: 
```json
{{"分数": x, "得分原因": "[解释你这样打分的原因]"}}.
```
我们会给你提供用户的问题、参考答案和AI模型的答案。
<问题>:
{question}
<参考答案>:
{reference}
<答题点>:
{rubrics}
<AI模型答案>:
{AI_answer}
"""

class NumericalEvaluator(BaseEvaluator):
    def __init__(self):
        self.client = OpenAI(base_url=NUMERICAL_EVAL_API_URL, api_key=NUMERICAL_EVAL_API_KEY)

    def evaluate_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            'question':    row.get('question', ''),
            'reference':   row.get('reference', ''),
            'rubrics':     row.get('rubrics', ''),
            'AI_answer':   row.get('answer', ''),
        }
        
        query = PROMPT_TEMPLATE.format_map(entry)
        
        try:
            response = self.client.chat.completions.create(
                model=NUMERICAL_EVAL_MODEL_NAME,
                messages=[{"role": "user", "content": query}],
                temperature=0.01,
            )
            content = response.choices[0].message.content
            score, reason = extract_json_from_response(content)
            return {'score': score, 'reason': reason}
        except Exception as e:
            return {'score': 0, 'reason': f"Error: {str(e)}"}

    def evaluate_dataset(self, data_path: str, pred_path: str) -> List[Dict[str, Any]]:
        print(f"Loading numerical dataset from {data_path}...")
        data = load_dataset(data_path)
        print(f"Loading predictions from {pred_path}...")
        preds = load_dataset(pred_path)
        
        if len(data) != len(preds):
            print(f"Warning: Data length ({len(data)}) != Preds length ({len(preds)})")
        
        results = []
        total_score = 0
        
        for i, row in enumerate(data):
            if i < len(preds):
                row['answer'] = preds[i].get('answer', '')
            else:
                row['answer'] = ""
                
            print(f"Evaluating item {i+1}/{len(data)}...")
            res = self.evaluate_single(row)
            print(f"  Score: {res['score']}, Reason: {str(res['reason'])[:50]}...")
            results.append({**row, **res})
            try:
                total_score += int(res['score'])
            except:
                pass
                
        print(f"\nNumerical Evaluation Complete. Total Score: {total_score}/{len(data)}")
        return results
