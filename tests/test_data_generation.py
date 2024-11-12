import unittest
import numpy as np
from explainability_toolkit.data_generation import generate_disjoint_datasets
from explainability_toolkit.boolean_functions_3x3 import first_example

class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test parameters"""
        self.num_samples = 100
        self.test_function = first_example

    def test_output_format(self):
        """Test if the output format is correct"""
        train_data, test_data = generate_disjoint_datasets(self.num_samples, self.test_function)
        
        # Check if outputs are lists
        self.assertIsInstance(train_data, list)
        self.assertIsInstance(test_data, list)
        
        # Check if each element is a tuple with correct types
        for dataset in [train_data, test_data]:
            for item in dataset:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
                self.assertIsInstance(item[0], np.ndarray)  # vector
                self.assertIsInstance(item[1], (bool, np.bool_, int))  # label

    def test_vector_properties(self):
        """Test properties of the generated vectors"""
        train_data, test_data = generate_disjoint_datasets(self.num_samples, self.test_function)
        
        for dataset in [train_data, test_data]:
            for vector, _ in dataset:
                # Check shape
                self.assertEqual(
                    vector.shape, 
                    (9,),
                    f"Vector has wrong shape: {vector.shape}"
                )
                
                # Check if binary
                unique_values = np.unique(vector)
                self.assertTrue(
                    np.all(np.isin(unique_values, [0, 1])),
                    f"Vector contains non-binary values: {unique_values}"
                )

    def test_label_consistency(self):
        """Test if labels are consistent with the boolean function"""
        train_data, test_data = generate_disjoint_datasets(self.num_samples, self.test_function)
        
        for dataset in [train_data, test_data]:
            for vector, label in dataset:
                expected_label = self.test_function(vector)
                self.assertEqual(
                    bool(label),
                    expected_label,
                    f"Label mismatch for vector {vector}"
                )

if __name__ == '__main__':
    unittest.main(verbosity=2)