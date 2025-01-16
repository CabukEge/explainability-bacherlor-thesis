# treeshap_explainer.py

import numpy as np
from typing import Dict, List, Tuple, Any
from .base_explainer import BaseExplainer
from sklearn.tree import DecisionTreeClassifier

class TreeSHAPExplainer(BaseExplainer):
    def __init__(self, model: Any):
        super().__init__(model)
        # Verify model type
        if not isinstance(model, DecisionTreeClassifier):
            raise ValueError(
                "TreeSHAP is only implemented for tree-based models. "
                "Got model type: {}. For neural networks, please use "
                "KernelSHAP or DeepSHAP instead.".format(type(model))
            )

    def _get_node_value(self, node_value: np.ndarray) -> float:
        """Extract probability of class 1 from node value array."""
        values = node_value.ravel()  # Flatten array
        return float(values[1]) / float(np.sum(values))  # Convert to probability

    def _get_path_weights(self, x: np.ndarray, tree) -> Tuple[List[int], List[float]]:
        """Compute weights for a single path through the tree."""
        path = []
        weights = []
        node = 0  # Start at root
        
        while True:
            if tree.children_left[node] == -1:  # Leaf node
                path.append(node)
                weights.append(1.0)
                break
                
            feature = tree.feature[node]
            threshold = tree.threshold[node]
            
            if x[feature] <= threshold:
                path.append(node)
                weights.append(float(tree.n_node_samples[tree.children_left[node]]) / 
                             float(tree.n_node_samples[node]))
                node = tree.children_left[node]
            else:
                path.append(node)
                weights.append(float(tree.n_node_samples[tree.children_right[node]]) / 
                             float(tree.n_node_samples[node]))
                node = tree.children_right[node]
                
        return path, weights

    def _compute_shap_values(self, x: np.ndarray, tree) -> np.ndarray:
        """Compute SHAP values for a single tree."""
        path, weights = self._get_path_weights(x, tree)
        shap_values = np.zeros(x.shape[0], dtype=np.float64)
        
        M = len(x)  # Number of features
        for i, node in enumerate(path[:-1]):  # Exclude leaf
            feature = tree.feature[node]
            
            # Get sets of features before this split
            zero_inds = set(tree.feature[path[:i]])
            one_inds = set(range(M)) - zero_inds - {feature}
            
            # Compute weight for this feature
            weight = 1.0
            for j in range(i + 1, len(path)):
                weight *= weights[j]
            
            # Add SHAP value contribution
            contrib = weight * (
                self._get_node_value(tree.value[path[-1]]) -  # Leaf value
                self._get_node_value(tree.value[node])        # Expected value at this node
            )
            shap_values[feature] += contrib
            
        return shap_values

    def explain(self, x: np.ndarray) -> Dict[str, Any]:
        """Explain prediction using TreeSHAP."""
        x = x.flatten()
        shap_values = self._compute_shap_values(x, self.model.tree_)
        
        # Get expected value (probability of class 1 at root node)
        expected_value = self._get_node_value(self.model.tree_.value[0])
        
        return {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'prediction': self._predict(x.reshape(1, -1))[0]
        }