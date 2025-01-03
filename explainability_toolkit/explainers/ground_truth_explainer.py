from typing import Dict, Any, Union
import numpy as np
import torch
from .base_explainer import BaseExplainer

class GroundTruthExplainer(BaseExplainer):
    """Ground truth explainer for 3x3 boolean functions."""
    
    def __init__(self, function_type: str):
        """
        Args:
            function_type: Type of boolean function ('first_example', 'min_x_amount', 'consecutive')
        """
        super().__init__(None)
        self.function_type = function_type
    
    def _first_example_attribution(self, x: np.ndarray) -> np.ndarray:
        """Attribution based on actual input pattern."""
        attributions = np.zeros(9)
        for i in range(9):
            if x[i] == 1:
                attributions[i] = 1.0
        return attributions
    
    def _get_pattern_expression(self, x: np.ndarray) -> str:
        terms = []
        if x[0] and x[1]:  # x1 ∧ x2
            terms.append("(x1 ∧ x2)")
        if x[3] and x[4] and x[5]:  # x4 ∧ x5 ∧ x6
            terms.append("(x4 ∧ x5 ∧ x6)")
        if x[6] and x[7] and x[8]:  # x7 ∧ x8 ∧ x9
            terms.append("(x7 ∧ x8 ∧ x9)")
        return " ∨ ".join(terms) if terms else "False"
    
    
    def explain(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Generate ground truth explanation based on input pattern."""
        x = self._validate_input(x)
        x_np = x.cpu().numpy().reshape(-1)
        
        if self.function_type == 'first_example':
            return {
                'attributions': self._first_example_attribution(x_np).reshape(3, 3),
                'function_type': self.function_type,
                'boolean_expression': self._get_pattern_expression(x_np)
            }
        else:
            raise ValueError(f"Only first_example is implemented")