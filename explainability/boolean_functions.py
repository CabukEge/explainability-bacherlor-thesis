import numpy as np
from typing import Union

def validate_input(x: Union[np.ndarray, list]) -> np.ndarray:
    """Validates boolean input vector."""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.shape != (9,):
        raise ValueError("Input must be 9-dimensional")
    if not np.all(np.isin(x, [0, 1])):
        raise ValueError("Input must contain only 0s and 1s")
    return x

def dnf_example(x: Union[np.ndarray, list]) -> bool:
    """Implementation of (x₁ ∧ x₂) ∨ (x₄ ∧ x₅ ∧ x₆) ∨ (x₇ ∧ x₈ ∧ x₉)"""
    x = validate_input(x)
    return (x[0] and x[1]) or \
           (x[3] and x[4] and x[5]) or \
           (x[6] and x[7] and x[8])

def min_ones(x: Union[np.ndarray, list], k: int) -> bool:
    """Returns true if at least k elements are 1."""
    x = validate_input(x)
    return np.sum(x) >= k

def consecutive_ones(x: Union[np.ndarray, list], k: int) -> bool:
    """Returns true if k consecutive elements are 1."""
    x = validate_input(x)
    return any(all(x[i:i+k]) for i in range(len(x)-k+1))