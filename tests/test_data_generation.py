import unittest
import numpy as np
import torch
from explainability_toolkit.data_generation import prepare_data, generate_complete_dataset, split_dataset
from explainability_toolkit.boolean_functions_3x3 import first_example

class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test parameters"""
        self.test_function = first_example

    def test_complete_dataset(self):
        """Test if complete dataset generation works"""
        X, y = generate_complete_dataset()
        self.assertEqual(len(X), 512)  # 2^9 combinations
        self.assertTrue(np.all(np.isin(X, [0, 1])))

    def test_data_splits(self):
        """Test if dataset splitting works correctly"""
        X, y = generate_complete_dataset()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)
        
        # Check split sizes
        self.assertAlmostEqual(len(X_train) / len(X), 0.7, places=1)
        self.assertAlmostEqual(len(X_val) / len(X), 0.15, places=1)
        self.assertAlmostEqual(len(X_test) / len(X), 0.15, places=1)

    def test_prepare_data(self):
        """Test if data preparation works"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(use_complete_dataset=True)
        self.assertTrue(torch.is_tensor(X_train))
        self.assertTrue(torch.is_tensor(y_train))
        self.assertEqual(X_train.shape[1:], (3, 3))

if __name__ == '__main__':
    unittest.main(verbosity=2)