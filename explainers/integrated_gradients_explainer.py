# integrated_gradients_explainer.py

import numpy as np
import torch
from typing import Dict, Any
from .base_explainer import BaseExplainer

class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, model, steps: int = 50, baseline: np.ndarray = None):
        """
        model: The PyTorch model (FCN or CNN).
        steps: Number of steps in the Riemann approximation of IG.
        baseline: If None, defaults to np.zeros_like(x).
        """
        super().__init__(model)
        self.steps = steps
        self.baseline = baseline

    def _compute_gradients(self, inputs: torch.Tensor, target_class: int = 1) -> torch.Tensor:
        """
        Computes gradients of the model output w.r.t. inputs for a specific class index.
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = self.model(inputs)
        # We take the logit for class=1, or you can do probabilities if you prefer
        logit = outputs[:, target_class]
        grad = torch.autograd.grad(torch.sum(logit), inputs)[0]
        return grad

    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Return integrated gradient attributions for the positive class (class index = 1).
        The shape of x is (9,) for FCN or (1,3,3) for CNN if you prefer.
        We'll flatten x to 9 features either way, then reshape inside if CNN.
        """
        # Flatten x for consistency
        x = x.flatten()
        x_tensor = torch.FloatTensor(x).unsqueeze(0)  # shape [1, 9]

        # Decide baseline
        if self.baseline is None:
            baseline = np.zeros_like(x)
        else:
            baseline = self.baseline
        baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0)

        # If CNN, reshape to [1,1,3,3]
        is_cnn = hasattr(self.model, 'conv1')
        if is_cnn:
            x_tensor = x_tensor.view(-1, 1, 3, 3)
            baseline_tensor = baseline_tensor.view(-1, 1, 3, 3)

        # Create scaled inputs
        scaled_inputs = [
            baseline_tensor + (float(i) / self.steps) * (x_tensor - baseline_tensor)
            for i in range(self.steps + 1)
        ]

        # Accumulate gradients
        total_grad = torch.zeros_like(x_tensor)
        for scaled_input in scaled_inputs:
            grad = self._compute_gradients(scaled_input, target_class=1)
            total_grad += grad

        # Average gradients
        avg_grad = total_grad / (self.steps + 1)

        # Integrated gradients = (x - x0) * avg_grad
        ig = (x_tensor - baseline_tensor) * avg_grad

        # Reshape back to [9] if CNN
        if is_cnn:
            ig = ig.view(-1, 9)

        # Return as numpy array
        ig_values = ig.squeeze(0).detach().numpy()

        # Also get the model's prediction for x
        pred = self._predict(x_tensor)

        return {
            # We'll store them in the same key used for SHAP so that
            # the evaluate.py logic can handle them in a similar way
            'shap_values': ig_values,
            'prediction': pred,
            'baseline': baseline
        }
