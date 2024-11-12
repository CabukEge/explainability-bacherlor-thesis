import numpy as np
from typing import Union

def validate_input(vector: Union[np.ndarray, list]) -> np.ndarray:
    """
    Validates the input vector for boolean functions.
    
    Args:
        vector: Input vector to validate
        
    Returns:
        np.ndarray: Validated numpy array
        
    Raises:
        IndexError: If vector length is not 9
        ValueError: If vector contains non-binary values
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector)
        
    if vector.shape != (9,):
        raise IndexError(f"Input vector must have exactly 9 elements, got {len(vector)}")
        
    if not np.all(np.logical_or(vector == 0, vector == 1)):
        raise ValueError("Input vector must contain only binary values (0 or 1)")
        
    return vector

def first_example(vector: Union[np.ndarray, list]) -> bool:
    """
    Implementation of the boolean function: (x₁ ∧ x₂) ∨ (x₄ ∧ x₅ ∧ x₆) ∨ (x₇ ∧ x₈ ∧ x₉)
    
    Args:
        vector: Binary input vector of length 9
        
    Returns:
        bool: Result of the boolean function
        
    Raises:
        IndexError: If vector length is not 9
        ValueError: If vector contains non-binary values
    """
    vector = validate_input(vector)
    
    # (x₁ ∧ x₂)
    clause1 = vector[0] and vector[1]
    
    # (x₄ ∧ x₅ ∧ x₆)
    clause2 = vector[3] and vector[4] and vector[5]
    
    # (x₇ ∧ x₈ ∧ x₉)
    clause3 = vector[6] and vector[7] and vector[8]
    
    return clause1 or clause2 or clause3

def min_x_amount_equals_one(vector: Union[np.ndarray, list], x: int) -> bool:
    """
    Check if at least x elements in the vector are 1.
    
    Args:
        vector: Binary input vector of length 9
        x: Minimum number of ones required
        
    Returns:
        bool: True if at least x elements are 1, False otherwise
    """
    vector = validate_input(vector)
    return np.sum(vector) >= x

def x_many_consecutively_one(vector: Union[np.ndarray, list], x: int) -> bool:
    """
    Check if there are x consecutive ones in the vector.
    
    Args:
        vector: Binary input vector of length 9
        x: Number of consecutive ones required
        
    Returns:
        bool: True if there are x consecutive ones, False otherwise
    """
    vector = validate_input(vector)
    
    if x > len(vector):
        return False
        
    for i in range(len(vector) - x + 1):
        if np.all(vector[i:i+x] == 1):
            return True
    return False