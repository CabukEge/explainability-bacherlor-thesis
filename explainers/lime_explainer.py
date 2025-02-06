# lime_explainer.py

import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Union, Any
import torch
import torch.nn as nn
from .base_explainer import BaseExplainer

class LIMEExplainer(BaseExplainer):
    def __init__(self, model: Union[nn.Module, Any], num_samples: int = 1000):
        super().__init__(model)
        self.num_samples = num_samples

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Helper method for model prediction."""
        if self.is_torch:
            with torch.no_grad():
                # For CNN we need to reshape to include channel dimension
                if isinstance(self.model, nn.Conv2d) or hasattr(self.model, 'conv1'):
                    x_tensor = torch.FloatTensor(x).view(-1, 1, 3, 3)
                else:
                    x_tensor = torch.FloatTensor(x).view(-1, 3, 3)
                output = self.model(x_tensor)
                return torch.softmax(output, dim=1)[:, 1].numpy()
        return self.model.predict_proba(x.reshape(-1, 9))[:, 1]

    def _generate_samples(self, x: np.ndarray) -> np.ndarray:
        """Generate perturbed samples around x."""
        samples = np.tile(x, (self.num_samples, 1))
        flip_probs = np.random.rand(*samples.shape)
        samples = np.where(flip_probs < 0.3, 1 - samples, samples)
        return samples

    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using LIME."""
        x = x.flatten()
        samples = self._generate_samples(x)
        predictions = self._predict(samples)
        
        # Weight samples by L1 distance
        distances = np.sum(np.abs(samples - x), axis=1)
        weights = np.exp(-distances)
        
        # Fit local model
        explainer = Ridge(alpha=1.0)
        explainer.fit(samples, predictions, sample_weight=weights)
        
        # Get prediction from the original model
        pred = self._predict(x.reshape(1, -1))[0]
        
        return {
            'coefficients': explainer.coef_,
            'prediction': pred,
            'local_accuracy': explainer.score(samples, predictions, sample_weight=weights)
        }