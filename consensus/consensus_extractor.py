"""Consensus extraction module.

Extracts common patterns/findings from multiple model analysis results.
Uses semantic alignment to identify what ≥60% of models discovered.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI


class ConsensusExtractor:
    """Extracts consensus findings from multiple analysis results."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        consensus_threshold: float = 0.6,
    ):
        """
        Args:
            api_key: API key for the extraction model
            base_url: Base URL for the extraction model
            model_name: Model to use for semantic alignment
            consensus_threshold: Minimum fraction of models that must mention a finding (default: 0.6)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.consensus_threshold = consensus_threshold

    def extract(
        self, analysis_results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract consensus and non-consensus findings.

        Args:
            analysis_results: List of analysis results from MultiModelAnalyzer

        Returns:
            Tuple of (consensus_findings, non_consensus_findings)
            Each finding is a dict with:
            - pattern: str (description of the finding)
            - evidence: list of str (supporting evidence from models)
            - frequency: float (fraction of models that mentioned it)
            - models: list of str (which models found it)
        """
        # Build prompt for semantic alignment
        n_models = len(analysis_results)
        analyses_text = self._format_analyses(analysis_results)

        prompt = f"""You are analyzing {n_models} independent data analysis results from different AI models.
Your task is to identify ALL distinct patterns and findings across these analyses, including those mentioned by only a single model.

# Analysis Results:
{analyses_text}

# Instructions:
1. Identify ALL distinct patterns/findings mentioned across the analyses
2. For each pattern, determine which models mentioned it (use semantic similarity, not exact wording)
3. Calculate the frequency (fraction of models that mentioned it)
4. Extract supporting evidence from each model's analysis
5. IMPORTANT: Do NOT omit findings that are only mentioned by one model. These unique insights are valuable and must be included with their correct frequency (e.g., 1/{n_models} = {1/n_models:.2f})

Output a JSON object with this structure:
{{
  "findings": [
    {{
      "pattern": "Brief description of the finding",
      "evidence": ["Quote from model 1", "Quote from model 2", ...],
      "frequency": 0.8,
      "models": ["model_name_1", "model_name_2", ...]
    }},
    ...
  ]
}}

Be precise. Only group findings that are semantically equivalent. Include ALL findings regardless of how many models mentioned them.
"""

        # Call LLM for extraction
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and comparing data analysis results."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        # Parse response
        try:
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    raise
            all_findings = result.get("findings", [])
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse consensus extraction result as JSON: {e}")
            all_findings = []

        # Split into consensus and non-consensus
        consensus_findings = [
            f for f in all_findings if f["frequency"] >= self.consensus_threshold
        ]
        non_consensus_findings = [
            f for f in all_findings if f["frequency"] < self.consensus_threshold
        ]

        return consensus_findings, non_consensus_findings

    def _format_analyses(self, analysis_results: List[Dict[str, Any]]) -> str:
        """Format analysis results for the prompt."""
        formatted = []
        for i, result in enumerate(analysis_results, 1):
            model_name = result.get("model_name", f"Model {i}")
            response = result.get("model_response", "")
            formatted.append(f"## Analysis {i} ({model_name}):\n{response}\n")
        return "\n".join(formatted)
