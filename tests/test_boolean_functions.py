import unittest
import numpy as np
from explainability_toolkit.boolean_functions_3x3 import (
    first_example,
    min_x_amount_equals_one,
    x_many_consecutively_one,
    validate_input
)

class TestBooleanFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        # True cases - each of these should return True
        self.true_cases = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),  # x1 AND x2
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),  # x4 AND x5 AND x6
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),  # x7 AND x8 AND x9
        ]
        
        # False cases - each of these should return False
        self.false_cases = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # All zeros
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # Only x1 true
            np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]),  # Only x4 and x5 true
        ]

    def test_true_cases(self):
        """Test cases where first_example should return True"""
        for i, test_vector in enumerate(self.true_cases):
            with self.subTest(case=i):
                result = first_example(test_vector)
                self.assertTrue(
                    result, 
                    f"Failed for true case {i}: {test_vector}"
                )

    def test_false_cases(self):
        """Test cases where first_example should return False"""
        for i, test_vector in enumerate(self.false_cases):
            with self.subTest(case=i):
                result = first_example(test_vector)
                self.assertFalse(
                    result, 
                    f"Failed for false case {i}: {test_vector}"
                )

    def test_input_validation(self):
        """Test input validation for various invalid inputs"""
        # Test vector too short
        with self.assertRaises(IndexError):
            first_example(np.array([1, 0]))
        
        # Test vector too long
        with self.assertRaises(IndexError):
            first_example(np.array([1] * 10))
        
        # Test non-binary values
        with self.assertRaises(ValueError):
            first_example(np.array([0, 1, 2, 0, 1, 0, 1, 0, 1]))
        
        # Test float values
        with self.assertRaises(ValueError):
            first_example(np.array([0.5] * 9))

    def test_list_input(self):
        """Test that the function works with list inputs"""
        # Test with list that should return True
        self.assertTrue(first_example([1, 1, 0, 0, 0, 0, 0, 0, 0]))
        
        # Test with list that should return False
        self.assertFalse(first_example([0, 0, 0, 0, 0, 0, 0, 0, 0]))

if __name__ == '__main__':
    unittest.main(verbosity=2)