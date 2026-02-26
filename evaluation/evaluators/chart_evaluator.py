import json
import os
import base64
from typing import Dict, Any, List
from openai import OpenAI
from .base_evaluator import BaseEvaluator
from ..config import CHART_EVAL_API_URL, CHART_EVAL_API_KEY, CHART_EVAL_MODEL_NAME
from ..utils import extract_json_from_response, load_dataset

PROMPT_TEMPLATE = """你是“图表正确性评分器”。你会收到：
- question：用户原始需求
- rubrics：答案要点列表（每条为一个可验证的检查点，满足=1分）
- chart：待评分的图表（可能是图片、图表截图、或图表的文字描述/规范）

你的任务：判断 chart 是否满足每条 rubrics，并给出总分。

评分规则：
1) 每条 rubric 只能给 0 或 1 分：满足=1，不满足/无法从图表中验证=0。
2) 只能基于 chart 中可见信息判断；不要猜测或脑补。看不出来就算未满足。
3) 评分是基于数据的正确性而评分，不是基于图表的美观度。
4) total_rubrics = rubrics 的条数；score = 满足条数。
5) reason 要简洁：说明主要扣分原因。

信息如下：
question: {question}
rubrics: {rubrics}
chart: 见下方图片。

输出要求（非常重要）：
- 只输出 JSON，不要输出任何额外文字、不要 markdown。
- JSON 字段必须且只能包含：
  {{
    "correct_rubrics": <int>,
    "total_rubrics": <int>,
    "reason": "<string>"
  }}
"""

VISUAL_PROMPT_TEMPLATE = """你是“图表美学评分器”。你会收到一张图表。
你的任务：根据以下维度评估图表的美学质量。

评估维度：
1. labels_and_titles: Clear, concise, and correctly placed.
2. layout_spacing: Well-organized and not cluttered.
3. color_accessibility: Distinct and colorblind-friendly where possible.
4. axis_scaling: Axes labeled correctly with proportional scaling.
5. chart_type_suitability: Appropriate chart type for the data/task.
6. font_and_legends: Readable fonts and properly aligned legends.
7. annotation_readability: Data labels/annotations/callouts/leader lines are clear, appropriately placed, and non-overlapping.
8. visual_hierarchy_and_emphasis: The key takeaway is immediately apparent; comparisons are highlighted effectively, and secondary information is appropriately de-emphasized (avoiding distraction from the main message).

评分规则：
1) 每个维度评分 0 或 1：1=满足/好，0=不满足/差/无法判断。
2) 给出每个维度的简要原因。

输出要求（非常重要）：
- 只输出 JSON，不要输出任何额外文字、不要 markdown。
- JSON 字段必须包含：
  {{
    "labels_and_titles": <0/1>,
    "labels_and_titles_reason": "<string>",
    "layout_spacing": <0/1>,
    "layout_spacing_reason": "<string>",
    "color_accessibility": <0/1>,
    "color_accessibility_reason": "<string>",
    "axis_scaling": <0/1>,
    "axis_scaling_reason": "<string>",
    "chart_type_suitability": <0/1>,
    "chart_type_suitability_reason": "<string>",
    "font_and_legends": <0/1>,
    "font_and_legends_reason": "<string>",
    "annotation_readability": <0/1>,
    "annotation_readability_reason": "<string>",
    "visual_hierarchy_and_emphasis": <0/1>,
    "visual_hierarchy_and_emphasis_reason": "<string>"
  }}
"""

class ChartEvaluator(BaseEvaluator):
    def __init__(self):
        self.client = OpenAI(
            base_url=CHART_EVAL_API_URL,
            api_key=CHART_EVAL_API_KEY
        )
        self.model = CHART_EVAL_MODEL_NAME

    def evaluate_visual(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image_path = row.get('image_path', '')
        
        if not os.path.exists(image_path):
            return {'reason': f"Image not found: {image_path}"}
            
        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            prompt = VISUAL_PROMPT_TEMPLATE
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = json.loads(content)
                return result
            except:
                return {'reason': f"Failed to parse JSON: {content}"}
                
        except Exception as e:
            return {'reason': f"Error: {str(e)}"}

    def evaluate_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        question = row.get('question', '')
        rubrics = row.get('rubrics', '')
        image_path = row.get('image_path', '')
        
        if not os.path.exists(image_path):
            return {'correct_rubrics': 0, 'reason': f"Image not found: {image_path}", 'total_rubrics': 3}
            
        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            prompt = PROMPT_TEMPLATE.format(question=question, rubrics=rubrics)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            
            # Extract score and reason using utility (it handles various JSON formats)
            # But here we expect "total_rubrics" too, so we might need custom parsing or just use the utility for basic fields
            # Let's try to parse directly first as we expect specific format
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    result = json.loads(content)
                return result
            except:
                return {'correct_rubrics': 0, 'reason': f"Failed to parse JSON: {content}", 'total_rubrics': 3}
                
        except Exception as e:
            return {'correct_rubrics': 0, 'reason': f"Error: {str(e)}", 'total_rubrics': 3}

    def evaluate_dataset(self, data_path: str, pred_path: str) -> List[Dict[str, Any]]:
        print(f"Loading chart dataset from {data_path}...")
        data = load_dataset(data_path)
        
        results = []
        total_score = 0
        total_items = 0
        
        for i, row in enumerate(data):
            print(f"Evaluating item {i+1}/{len(data)}...")
            
            # Construct image path from pred_path (directory) and image_name
            image_name = row.get('image_name', '')
            if not image_name:
                print("  Skipping: No image_name in data")
                continue
                
            row['image_path'] = os.path.join(pred_path, image_name)
            
            res = self.evaluate_single(row)
            print(f"  Score: {res['score']}, Reason: {str(res.get('reason', ''))[:50]}...")
            
            results.append({**row, **res})
            try:
                
                total_score += int(res['score'])
                total_items += 1
            except:
                pass
                
        print(f"\nChart Evaluation Complete. Total Score: {total_score}/{total_items}")
        return results
