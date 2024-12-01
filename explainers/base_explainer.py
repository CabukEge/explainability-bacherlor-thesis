from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from torch import nn

class BaseExplainer(ABC):
    """
    Abstract base class for all explainability methods.
    
    This class defines the common interface that all explainers must implement.
    It provides basic functionality for model input/output handling and result
    validation.
    """
    
    def __init__(self, model: Union[nn.Module, Any], device: str = 'cpu'):
        """
        Initialize the explainer.
        
        Args:
            model: The model to explain (PyTorch model or sklearn-compatible model)
            device: The device to use for PyTorch models ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.is_torch_model = isinstance(model, nn.Module)
        
        if self.is_torch_model:
            self.model.to(self.device)
            self.model.eval()
    
    def _validate_input(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Validate and prepare input for explanation.
        
        Args:
            x: Input to validate (3x3 grid as numpy array or torch tensor)
            
        Returns:
            torch.Tensor: Validated and prepared input
            
        Raises:
            ValueError: If input shape is invalid
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Check if input is a single sample or batch
        if len(x.shape) == 1:
            if x.shape[0] != 9:
                raise ValueError(f"Single input must be 9-dimensional, got shape {x.shape}")
            x = x.reshape(1, 3, 3)
        elif len(x.shape) == 2:
            if x.shape[1] != 9:
                raise ValueError(f"Each input must be 9-dimensional, got shape {x.shape}")
            x = x.reshape(-1, 3, 3)
        elif len(x.shape) == 3:
            if x.shape[1:] != (3, 3):
                raise ValueError(f"Input must have shape (batch_size, 3, 3), got shape {x.shape}")
        else:
            raise ValueError(f"Invalid input shape {x.shape}")
        
        return x.to(self.device)
    
    def _model_predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Get model predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            np.ndarray: Model predictions
        """
        with torch.no_grad():
            if self.is_torch_model:
                outputs = self.model(x)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            else:
                # Handle non-PyTorch models (e.g., sklearn)
                x_numpy = x.cpu().numpy().reshape(x.shape[0], -1)
                predictions = self.model.predict(x_numpy)
                
        return predictions
    
    @abstractmethod
    def explain(self, 
                x: Union[np.ndarray, torch.Tensor],
                target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate explanation for the input.
        
        Args:
            x: Input to explain (3x3 grid)
            target_class: Target class to explain (optional)
            
        Returns:
            Dict containing explanation details:
                - 'attributions': Feature importance scores
                - 'prediction': Model's prediction
                - Additional method-specific information
        """
        pass
    
    @abstractmethod
    def batch_explain(self,
                     x: Union[np.ndarray, torch.Tensor],
                     target_classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of inputs.
        
        Args:
            x: Batch of inputs to explain
            target_classes: List of target classes to explain (optional)
            
        Returns:
            List of explanation dictionaries
        """
        pass
    
    def _format_explanation(self,
                          attributions: np.ndarray,
                          prediction: int,
                          additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format the explanation results consistently.
        
        Args:
            attributions: Feature importance scores
            prediction: Model's prediction
            additional_info: Additional method-specific information
            
        Returns:
            Dict containing formatted explanation
        """
        explanation = {
            'attributions': attributions.reshape(3, 3),
            'prediction': prediction,
            'attribution_sum': float(np.sum(attributions)),
            'attribution_max': float(np.max(np.abs(attributions))),
            'attribution_min': float(np.min(np.abs(attributions)))
        }
        
        if additional_info:
            explanation.update(additional_info)
            
        return explanation
    
    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"{self.__class__.__name__}(model={type(self.model).__name__}, device='{self.device}')"