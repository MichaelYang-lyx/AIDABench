from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def evaluate_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single data item."""
        pass

    @abstractmethod
    def evaluate_dataset(self, data_path: str, pred_path: str) -> List[Dict[str, Any]]:
        """Evaluate an entire dataset."""
        pass
