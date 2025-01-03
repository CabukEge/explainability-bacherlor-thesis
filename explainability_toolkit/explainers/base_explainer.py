from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from torch import nn

class BaseExplainer(ABC):
    def __init__(self, model: Union[nn.Module, Any], device: str = 'cpu'):
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
            x: Input to validate (3x3 grid or 9-dimensional vector)
            
        Returns:
            torch.Tensor: Validated and prepared input
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Handle different input shapes
        if len(x.shape) == 1:
            if x.shape[0] != 9:
                raise ValueError(f"Single input must be 9-dimensional, got shape {x.shape}")
            x = x.reshape(1, 3, 3)
        elif len(x.shape) == 2:
            if x.shape == (3, 3):
                x = x.reshape(1, 3, 3)
            elif x.shape[1] == 9:
                x = x.reshape(-1, 3, 3)
            else:
                raise ValueError(f"Invalid input shape {x.shape}")
        elif len(x.shape) == 3:
            if x.shape[1:] != (3, 3):
                raise ValueError(f"Input must have shape (batch_size, 3, 3), got shape {x.shape}")
        else:
            raise ValueError(f"Invalid input shape {x.shape}")
        
        return x.to(self.device)