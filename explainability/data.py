import numpy as np
import torch
from itertools import product
from typing import Tuple, Callable
from boolean_functions import dnf_example

def generate_data(func: Callable = dnf_example) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor]
]:
    """Generate complete dataset for 3x3 grid problems"""
    # Generate all possible 9-bit combinations
    X = np.array(list(product([0, 1], repeat=9)))
    y = np.array([func(x) for x in X])
    
    # Shuffle consistently
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Split: 70% train, 15% val, 15% test
    n_train = int(0.7 * len(X))
    n_val = int(0.85 * len(X))
    
    X_train = torch.FloatTensor(X[:n_train]).reshape(-1, 3, 3)
    y_train = torch.LongTensor(y[:n_train])
    
    X_val = torch.FloatTensor(X[n_train:n_val]).reshape(-1, 3, 3)
    y_val = torch.LongTensor(y[n_train:n_val])
    
    X_test = torch.FloatTensor(X[n_val:]).reshape(-1, 3, 3)
    y_test = torch.LongTensor(y[n_val:])
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)