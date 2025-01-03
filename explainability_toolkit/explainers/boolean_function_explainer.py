from typing import List, Dict, Set, Union, Optional, Tuple, Any
import numpy as np
import torch
from itertools import combinations
import torch.nn as nn
from .base_explainer import BaseExplainer

class BooleanFunctionExplainer(BaseExplainer):
    def __init__(self, 
                 model: Union[nn.Module, Any],
                 device: str = 'cpu',
                 threshold: float = 0.99):
        super().__init__(model, device)
        self.threshold = threshold
        self._reset_state()
    
    def _generate_all_patterns(self, n: int = 9) -> List[np.ndarray]:
        """Generate all possible binary patterns up to n variables."""
        patterns = []
        # Start with smaller combinations first
        for r in range(1, n + 1):
            for combo in combinations(range(n), r):
                pattern = np.zeros(n, dtype=np.float32)
                pattern[list(combo)] = 1
                patterns.append(pattern)
        return patterns
    
    def _test_pattern(self, pattern: np.ndarray) -> bool:
        """Test if a pattern triggers a positive prediction."""
        x = torch.FloatTensor(pattern).reshape(1, 3, 3)
        if self.is_torch_model:
            x = x.to(self.device)
            
        with torch.no_grad():
            if self.is_torch_model:
                outputs = self.model(x)
                prediction = outputs.argmax(dim=1).cpu().numpy()[0]
            else:
                x_numpy = x.cpu().numpy().reshape(1, -1)
                prediction = self.model.predict(x_numpy)[0]
                
        return bool(prediction == 1)
    
    def _is_minimal_pattern(self, pattern: np.ndarray) -> bool:
        """Check if pattern is minimal and necessary."""
        active_indices = np.where(pattern == 1)[0]
        
        # Check if any subset triggers positive prediction
        for r in range(1, len(active_indices)):
            for sub_combo in combinations(active_indices, r):
                sub_pattern = np.zeros_like(pattern)
                sub_pattern[list(sub_combo)] = 1
                pattern_tuple = tuple(sub_pattern.tolist())
                
                if pattern_tuple in self.pattern_responses:
                    if self.pattern_responses[pattern_tuple]:
                        return False
                else:
                    response = self._test_pattern(sub_pattern)
                    self.pattern_responses[pattern_tuple] = response
                    if response:
                        return False
        
        return True
    
    def explain(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Generate boolean function explanation for the model."""
        self._reset_state()
        x = self._validate_input(x)
        
        # Generate and test all patterns
        all_patterns = self._generate_all_patterns()
        minimal_positive_patterns = []
        
        # First find all positive patterns
        for pattern in all_patterns:
            pattern_tuple = tuple(pattern.tolist())
            
            if pattern_tuple not in self.pattern_responses:
                self.pattern_responses[pattern_tuple] = self._test_pattern(pattern)
                
            if self.pattern_responses[pattern_tuple]:
                if self._is_minimal_pattern(pattern):
                    minimal_positive_patterns.append(pattern)
        
        # Convert patterns to boolean expression
        if not minimal_positive_patterns:
            boolean_function = "False"
        else:
            terms = []
            for pattern in minimal_positive_patterns:
                active_vars = [f"x{i+1}" for i, val in enumerate(pattern) if val == 1]
                terms.append(" ∧ ".join(active_vars))
            boolean_function = " ∨ ".join(f"({term})" for term in terms)
        
        return {
            'boolean_function': boolean_function,
            'minimal_patterns': minimal_positive_patterns,
            'pattern_responses': self.pattern_responses
        }
    
    def _reset_state(self):
        """Reset the internal state for new explanation."""
        self.pattern_responses = {}
        
    def batch_explain(self, 
                     x: Union[np.ndarray, torch.Tensor],
                     target_classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Generate explanations for a batch of inputs."""
        explanation = self.explain(x[0])
        return [explanation] * len(x)