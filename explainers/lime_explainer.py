from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import torch
from sklearn.linear_model import LassoLars
from .base_explainer import BaseExplainer

class LimeExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) implementation for 3x3 grid inputs.
    This implementation is specifically designed for binary classification of 3x3 boolean grids.
    """
    
    def __init__(self, 
                 model: Union[torch.nn.Module, Any],
                 num_samples: int = 1000,
                 kernel_width: float = 0.25,
                 device: str = 'cpu'):
        """
        Initialize LIME explainer.
        
        Args:
            model: The model to explain
            num_samples: Number of samples to generate for local approximation
            kernel_width: Kernel width for exponential kernel
            device: Device to use for PyTorch models
        """
        super().__init__(model, device)
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.surrogate_model = LassoLars(alpha=0.01)
        
    def _generate_neighborhood(self, 
                             x: np.ndarray, 
                             n_samples: int) -> np.ndarray:
        """
        Generate neighborhood samples around the input instance.
        For 3x3 boolean grid, we randomly flip bits with probability based on distance.
        
        Args:
            x: Original input instance (9-dimensional binary vector)
            n_samples: Number of samples to generate
            
        Returns:
            np.ndarray: Generated samples of shape (n_samples, 9)
        """
        neighborhood = np.zeros((n_samples, 9))
        for i in range(n_samples):
            # Random probability for each cell
            probs = np.random.random(9)
            # Flip bits with probability proportional to distance from border
            flips = probs < 0.3  # 30% chance to flip each bit
            sample = x.copy()
            sample[flips] = 1 - sample[flips]  # Flip selected bits
            neighborhood[i] = sample
            
        return neighborhood
        
    def _compute_distances(self, 
                          original: np.ndarray, 
                          samples: np.ndarray) -> np.ndarray:
        """
        Compute distances between original instance and samples.
        Uses Hamming distance since we're dealing with binary features.
        
        Args:
            original: Original instance
            samples: Generated neighborhood samples
            
        Returns:
            np.ndarray: Distances between original and samples
        """
        return np.array([np.sum(original != sample) for sample in samples])
    
    def _compute_kernel_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights based on distances.
        
        Args:
            distances: Distances between original and samples
            
        Returns:
            np.ndarray: Kernel weights for samples
        """
        return np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))
    
    def explain(self, 
                x: Union[np.ndarray, torch.Tensor],
                target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single input.
        
        Args:
            x: Input to explain (3x3 grid)
            target_class: Target class to explain (if None, uses model prediction)
            
        Returns:
            Dict containing:
                - attributions: Feature importance scores (3x3)
                - prediction: Model's prediction
                - local_model_accuracy: R² score of local surrogate model
                - local_model_coef: Coefficients of local surrogate model
        """
        # Validate and prepare input
        x_tensor = self._validate_input(x)
        x_numpy = x_tensor.cpu().numpy().reshape(-1)
        
        # Get model prediction if target_class not specified
        if target_class is None:
            target_class = self._model_predict(x_tensor)[0]
        
        # Generate neighborhood samples
        neighborhood = self._generate_neighborhood(x_numpy, self.num_samples)
        
        # Get model predictions for neighborhood
        neighborhood_tensor = torch.FloatTensor(neighborhood).reshape(-1, 3, 3)
        if self.is_torch_model:
            neighborhood_tensor = neighborhood_tensor.to(self.device)
        neighborhood_preds = self._model_predict(neighborhood_tensor)
        
        # Compute distances and weights
        distances = self._compute_distances(x_numpy, neighborhood)
        weights = self._compute_kernel_weights(distances)
        
        # Fit local surrogate model
        self.surrogate_model.fit(
            neighborhood, 
            neighborhood_preds == target_class,
            sample_weight=weights
        )
        
        # Get local model accuracy
        local_accuracy = self.surrogate_model.score(
            neighborhood,
            neighborhood_preds == target_class,
            sample_weight=weights
        )
        
        # Format explanation
        attributions = self.surrogate_model.coef_
        additional_info = {
            'local_model_accuracy': float(local_accuracy),
            'local_model_coef': attributions.tolist(),
            'target_class': int(target_class)
        }
        
        return self._format_explanation(attributions, target_class, additional_info)
    
    def batch_explain(self,
                     x: Union[np.ndarray, torch.Tensor],
                     target_classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations for a batch of inputs.
        
        Args:
            x: Batch of inputs to explain
            target_classes: List of target classes to explain
            
        Returns:
            List of explanation dictionaries
        """
        x_tensor = self._validate_input(x)
        batch_size = x_tensor.shape[0]
        
        if target_classes is None:
            target_classes = self._model_predict(x_tensor)
        
        return [
            self.explain(
                x_tensor[i], 
                target_classes[i] if target_classes else None
            )
            for i in range(batch_size)
        ]