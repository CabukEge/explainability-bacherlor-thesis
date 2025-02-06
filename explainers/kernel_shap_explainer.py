# kernel_shap_explainer.py

import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Any
from .base_explainer import BaseExplainer
import scipy.special

class KernelSHAPExplainer(BaseExplainer):
    def __init__(self, model: Any, num_samples: int = 1000):
        super().__init__(model)
        self.num_samples = num_samples

    def _shapley_kernel(self, M: int, s: int) -> float:
        """Computes the Shapley kernel weight.
        Args:
            M: Total number of features
            s: Size of the current subset
        """
        if s == 0 or s == M:
            return float('inf')
        return (M - 1) / (scipy.special.comb(M, s) * s * (M - s))

    def _generate_samples(self, x: np.ndarray) -> np.ndarray:
        """Generates samples by setting random features to their background values."""
        M = x.shape[0]  # Number of features
        samples = []
        weights = []

        # Generate random coalitions
        for _ in range(self.num_samples):
            coalition = np.random.binomial(1, 0.5, M)
            sample = np.copy(x)
            sample[coalition == 0] = 0  # Set masked features to 0
            samples.append(sample)
            weights.append(self._shapley_kernel(M, np.sum(coalition)))

        return np.array(samples), np.array(weights)

    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using KernelSHAP."""
        x = x.flatten()
        samples, weights = self._generate_samples(x)
        
        # Get model predictions for samples
        predictions = np.array([self._predict(s.reshape(1, -1))[0] for s in samples])
        
        # Replace infinite weights with large finite values
        max_weight = np.max(weights[~np.isinf(weights)])
        weights[np.isinf(weights)] = max_weight * 1000
        
        # Prepare feature matrix for weighted linear regression
        # Each sample becomes a row in the design matrix
        X = samples
        
        # Solve weighted linear regression
        # This directly gives us the SHAP values as coefficients
        explainer = Ridge(alpha=0.01, fit_intercept=True)
        explainer.fit(X, predictions, sample_weight=weights)
        
        # Get prediction for the current input
        prediction = self._predict(x.reshape(1, -1))[0]
        
        return {
            'shap_values': explainer.coef_,  # This is what evaluate.py expects
            'prediction': prediction,
            'base_value': explainer.intercept_,  # Base value (average prediction)
            'local_accuracy': explainer.score(X, predictions, sample_weight=weights)
        }