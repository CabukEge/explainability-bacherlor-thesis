import numpy as np
from typing import Union

def validate_input(vector: Union[np.ndarray, list]) -> np.ndarray:
    """
    Validates the input vector for boolean functions.
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
    """
    vector = validate_input(vector)
    return np.sum(vector) >= x

def x_many_consecutively_one(vector: Union[np.ndarray, list], x: int) -> bool:
    """
    Check if there are x consecutive ones in the vector.
    Takes vector as row-major 1D array and checks for consecutive ones in that order.
    """
    vector = validate_input(vector)
    
    if x > len(vector):
        return False
    
    # Check for consecutive ones in row-major order
    count = 0
    for val in vector:
        if val == 1:
            count += 1
            if count >= x:
                return True
        else:
            count = 0
    return False

def test_x_many_consecutively_one():
    """
    Helper function to test consecutive ones detection
    """
    test_cases = [
        # True cases - horizontal consecutive
        ([1, 1, 1, 0, 0, 0, 0, 0, 0], True),  # First row
        ([0, 0, 0, 1, 1, 1, 0, 0, 0], True),  # Middle row
        ([0, 0, 0, 0, 0, 0, 1, 1, 1], True),  # Last row
        
        # True cases - vertical consecutive 
        ([1, 0, 0, 1, 0, 0, 1, 0, 0], True),  # First column
        ([0, 1, 0, 0, 1, 0, 0, 1, 0], True),  # Middle column
        ([0, 0, 1, 0, 0, 1, 0, 0, 1], True),  # Last column
        
        # False cases
        ([1, 1, 0, 1, 0, 0, 0, 0, 0], False),  # Only 2 consecutive
        ([1, 0, 1, 1, 0, 0, 0, 0, 0], False),  # Non-consecutive
        ([0, 0, 0, 0, 0, 0, 0, 0, 0], False),  # All zeros
    ]
    
    for input_vec, expected in test_cases:
        result = x_many_consecutively_one(np.array(input_vec), 3)
        print(f"Input: {input_vec}")
        print(f"Expected: {expected}, Got: {result}")
        print(f"Match: {result == expected}\n")