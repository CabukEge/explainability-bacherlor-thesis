# boolean_functions.py

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

def dnf_simple(x: Union[np.ndarray, list]) -> bool:
    """Implementation of (x₁ ∧ x₂) ∨ (x₅ ∧ x₆)"""
    x = validate_input(x)
    return (x[0] and x[1]) or \
           (x[4] and x[5])

def dnf_complex(x: Union[np.ndarray, list]) -> bool:
    """Implementation of (x₁ ∧ x₂ ∧ x₃) ∨ (x₄ ∧ x₅) ∨ (x₇ ∧ x₈ ∧ x₉)"""
    x = validate_input(x)
    return (x[0] and x[1] and x[2]) or \
           (x[3] and x[4]) or \
           (x[6] and x[7] and x[8])