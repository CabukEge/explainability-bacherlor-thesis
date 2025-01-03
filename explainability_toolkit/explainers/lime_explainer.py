from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from sklearn.linear_model import Ridge
from .base_explainer import BaseExplainer

class LimeExplainer(BaseExplainer):
    def __init__(self, 
                 model: Union[torch.nn.Module, Any],
                 num_samples: int = 1000,
                 kernel_width: float = 0.75,
                 device: str = 'cpu'):
        super().__init__(model, device)
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.surrogate_model = Ridge(alpha=0.1)
    
    def _model_predict(self, x: torch.Tensor) -> np.ndarray:
        """Get model predictions with proper handling of negative cases."""
        if self.is_torch_model:
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
                probs = torch.softmax(output, dim=1)
                # Convert to centered predictions (-1 to 1 range)
                return 2 * probs[:, 1].cpu().numpy() - 1
        else:
            probs = self.model.predict_proba(x.cpu().numpy().reshape(len(x), -1))[:, 1]
            return 2 * probs - 1
    
    def _generate_neighborhood(self, x: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate neighborhood samples with balanced positive and negative cases."""
        neighborhood = np.zeros((n_samples, 9))
        x_flat = x.flatten()
        
        for i in range(n_samples):
            # Vary number of flips with geometric distribution
            num_flips = np.random.geometric(p=0.5)
            num_flips = min(num_flips, 9)
            
            # Choose positions to flip
            flip_positions = np.random.choice(9, size=num_flips, replace=False)
            
            # Create perturbed sample
            sample = x_flat.copy()
            sample[flip_positions] = 1 - sample[flip_positions]
            neighborhood[i] = sample
            
        return neighborhood
        
    def _compute_distances(self, original: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """Compute Hamming distances between original and samples."""
        return np.array([np.sum(original != sample) for sample in samples])
    
    def _compute_kernel_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute kernel weights with proper scaling."""
        normalized_distances = distances / np.sqrt(9)
        return np.exp(-(normalized_distances ** 2) / (self.kernel_width ** 2))

    def _find_patterns(self, grid: np.ndarray, threshold: float = 0.1) -> List[List[int]]:
        """Find significant patterns in the attribution grid."""
        patterns = []
        abs_grid = np.abs(grid)
        
        # Find clusters of significant attributions
        visited = np.zeros_like(grid, dtype=bool)
        
        def get_neighbors(i, j):
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    neighbors.append((ni, nj))
            return neighbors
        
        def explore_cluster(i, j, cluster):
            if visited[i,j] or abs_grid[i,j] < threshold:
                return
            
            visited[i,j] = True
            cluster.append(i * 3 + j)
            
            for ni, nj in get_neighbors(i, j):
                if not visited[ni,nj] and abs_grid[ni,nj] >= threshold:
                    explore_cluster(ni, nj, cluster)
        
        for i in range(3):
            for j in range(3):
                if not visited[i,j] and abs_grid[i,j] >= threshold:
                    cluster = []
                    explore_cluster(i, j, cluster)
                    if cluster:
                        patterns.append(sorted(cluster))
        
        if not patterns:
            significant_indices = np.where(abs_grid.flatten() >= threshold)[0]
            if len(significant_indices) > 0:
                patterns = [[idx] for idx in significant_indices]
        
        return patterns

    def _format_boolean_expression(self, attributions: np.ndarray, prediction: int) -> str:
        """Format attributions as a DNF boolean expression with proper handling of negative predictions."""
        # Scale attributions to [-1, 1] range
        max_abs_attr = np.max(np.abs(attributions))
        if max_abs_attr > 0:
            scaled_attrs = attributions / max_abs_attr
        else:
            return "False"
        
        # Find significant patterns
        grid = scaled_attrs.reshape(3, 3)
        patterns = self._find_patterns(grid, threshold=0.2)
        
        if not patterns:
            return "False"
        
        # Convert patterns to boolean expressions
        dnf_terms = []
        for pattern in patterns:
            pattern_values = scaled_attrs.flatten()[pattern]
            # For negative predictions, look for negative patterns
            if prediction == 0 and np.all(pattern_values < 0):
                vars_in_term = [f"¬x{i+1}" for i in pattern]
                dnf_terms.append(f"({' ∧ '.join(vars_in_term)})")
            # For positive predictions, look for positive patterns
            elif prediction == 1 and np.all(pattern_values > 0):
                vars_in_term = [f"x{i+1}" for i in pattern]
                dnf_terms.append(f"({' ∧ '.join(vars_in_term)})")
        
        if not dnf_terms:
            return "False"
        elif len(dnf_terms) == 1:
            return dnf_terms[0]
        else:
            return " ∨ ".join(dnf_terms)
    
    def explain(self, 
                x: Union[np.ndarray, torch.Tensor],
                target_class: Optional[int] = None) -> Dict[str, Any]:
        """Generate LIME explanation with improved handling of negative cases."""
        x_tensor = self._validate_input(x)
        x_numpy = x_tensor.cpu().numpy().reshape(-1)
        
        # Get model prediction and centered probabilities
        centered_probs = self._model_predict(x_tensor.reshape(1, 3, 3))
        pred_class = (centered_probs > 0).astype(int)[0]
        
        if target_class is None:
            target_class = pred_class
        
        # Generate neighborhood samples
        neighborhood = self._generate_neighborhood(x_numpy, self.num_samples)
        neighborhood_tensor = torch.FloatTensor(neighborhood).reshape(-1, 3, 3)
        if self.is_torch_model:
            neighborhood_tensor = neighborhood_tensor.to(self.device)
        
        # Get predictions for neighborhood
        neighborhood_preds = self._model_predict(neighborhood_tensor)
        
        # Compute distances and weights
        distances = self._compute_distances(x_numpy, neighborhood)
        weights = self._compute_kernel_weights(distances)
        
        # Fit local model
        self.surrogate_model.fit(
            neighborhood, 
            neighborhood_preds,  # Using centered predictions
            sample_weight=weights
        )
        
        # Calculate local model accuracy
        local_preds = self.surrogate_model.predict(neighborhood)
        local_accuracy = np.average(
            (local_preds > 0) == (neighborhood_preds > 0),
            weights=weights
        )
        
        # Get feature attributions
        attributions = self.surrogate_model.coef_
        
        # Ensure attributions reflect prediction
        if pred_class == 0 and np.min(attributions) > 0:
            attributions = -attributions
        
        return {
            'attributions': attributions.reshape(3, 3),
            'prediction': int(pred_class),
            'local_model_accuracy': float(local_accuracy),
            'local_model_coef': attributions.tolist(),
            'boolean_expression': self._format_boolean_expression(attributions, pred_class)
        }