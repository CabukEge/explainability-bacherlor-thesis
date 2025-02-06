# base_explainer.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

class BaseExplainer(ABC):
    def __init__(self, model: Any):
        self.model = model
        self.is_torch = isinstance(model, nn.Module)
        if self.is_torch:
            self.model.eval()

    @abstractmethod
    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """Main explanation method to be implemented by each explainer."""
        pass
    
    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Helper method for model prediction."""
        if self.is_torch:
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).view(-1, 1, 3, 3)
                output = self.model(x_tensor)
                return torch.softmax(output, dim=1)[:, 1].numpy()
        return self.model.predict_proba(x.reshape(-1, 9))[:, 1]