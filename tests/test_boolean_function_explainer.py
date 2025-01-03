import unittest
import torch
import numpy as np
from explainability_toolkit.data_generation import prepare_data
from explainability_toolkit.models import SimpleNet
from explainability_toolkit.boolean_functions_3x3 import (
    first_example,
    min_x_amount_equals_one,
    x_many_consecutively_one
)
from explainability_toolkit.train import train_fully_connected, SEED_LIST
from explainability_toolkit.explainers.boolean_function_explainer import BooleanFunctionExplainer

class TestBooleanFunctionExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nSetting up test class...")
        # Prepare data
        (cls.X_train, cls.y_train), (cls.X_val, cls.y_val), (cls.X_test, cls.y_test) = \
            prepare_data(use_complete_dataset=True)
        
        # Train model for first_example
        losses, models = train_fully_connected(
            cls.X_train, 
            cls.y_train,
            seed_list=SEED_LIST[:1],
            num_epochs=500
        )
        cls.model = models[0]
        cls.model.eval()
        
        # Verify model accuracy
        with torch.no_grad():
            predictions = cls.model(cls.X_test).argmax(dim=1)
            accuracy = (predictions == cls.y_test).float().mean()
            assert accuracy == 1.0, f"Model accuracy must be 100%, got {accuracy*100:.2f}%"
        
        # Create explainer
        cls.explainer = BooleanFunctionExplainer(cls.model)
        print("Setup complete.")
    
    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")

    def _print_error_cases(self, X_test, predictions, y_test, max_cases=5):
        """Helper function to safely print error cases."""
        mismatches = (predictions != y_test).nonzero().cpu()
        if len(mismatches) == 0:
            return
            
        print("\nFailure cases:")
        # Handle both single and multiple error cases
        if mismatches.dim() == 0:
            # Single error case
            idx = mismatches.item()
            input_vec = X_test[idx].cpu().numpy().reshape(9)
            pred = predictions[idx].item()
            true = y_test[idx].item()
            print(f"Input: {input_vec}")
            print(f"Predicted: {pred}, True: {true}")
        else:
            # Multiple error cases
            for i, idx in enumerate(mismatches[:max_cases].flatten()):
                input_vec = X_test[idx].cpu().numpy().reshape(9)
                pred = predictions[idx].item()
                true = y_test[idx].item()
                print(f"Input: {input_vec}")
                print(f"Predicted: {pred}, True: {true}")
    
    def test_first_example_reconstruction(self):
        """Test reconstruction of first_example boolean function."""
        print("\nTesting first_example reconstruction...")
        explanation = self.explainer.explain(self.X_test[0].reshape(9))
        reconstructed = explanation['boolean_function']
        print("\nReconstructed boolean function:")
        print(reconstructed)
        
        test_cases = [
            torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0]).float(),
            torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0]).float(),
            torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1]).float(),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0]).float(),
        ]
        
        for test_case in test_cases:
            with torch.no_grad():
                pred = self.model(test_case.reshape(1, 3, 3)).argmax().item()
            truth = first_example(test_case.numpy())
            self.assertEqual(pred, int(truth))
    
    def test_min_x_amount_reconstruction(self):
        """Test reconstruction with min_x_amount_equals_one function."""
        print("\nTesting min_x_amount reconstruction...")
        y_train = torch.tensor([
            int(min_x_amount_equals_one(x.reshape(9), 3))
            for x in self.X_train.numpy()
        ]).long()
        
        losses, models = train_fully_connected(
            self.X_train, 
            y_train,
            seed_list=SEED_LIST[:1],
            num_epochs=500,
            learning_rate=0.01
        )
        
        model = models[0]
        model.eval()
        
        with torch.no_grad():
            y_test = torch.tensor([
                int(min_x_amount_equals_one(x.reshape(9), 3))
                for x in self.X_test.numpy()
            ]).long()
            predictions = model(self.X_test).argmax(dim=1)
            accuracy = (predictions == y_test).float().mean()
            if accuracy < 1.0:
                self._print_error_cases(self.X_test, predictions, y_test)
            self.assertEqual(accuracy, 1.0)
        
        explainer = BooleanFunctionExplainer(model)
        explanation = explainer.explain(self.X_test[0].reshape(9))
        print("\nReconstructed min_x_amount_equals_one function (x=3):")
        print(explanation['boolean_function'])
    
    def test_consecutive_ones_reconstruction(self):
        """Test reconstruction with x_many_consecutively_one function."""
        print("\nTesting consecutive_ones reconstruction...")
        y_train = torch.tensor([
            int(x_many_consecutively_one(x.reshape(9), 3))
            for x in self.X_train.numpy()
        ]).long()
        
        best_accuracy = 0
        best_model = None
        
        for seed in SEED_LIST[:10]:
            print(f"\nTrying seed {seed}...")
            losses, models = train_fully_connected(
                self.X_train, 
                y_train,
                seed_list=[seed],
                num_epochs=10000,
                learning_rate=0.0005
            )
            
            model = models[0]
            model.eval()
            
            with torch.no_grad():
                y_test = torch.tensor([
                    int(x_many_consecutively_one(x.reshape(9), 3))
                    for x in self.X_test.numpy()
                ]).long()
                predictions = model(self.X_test).argmax(dim=1)
                accuracy = (predictions == y_test).float().mean()
                print(f"Accuracy with seed {seed}: {accuracy*100:.2f}%")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    if accuracy < 1.0:
                        self._print_error_cases(self.X_test, predictions, y_test)
            
            if best_accuracy == 1.0:
                print(f"Achieved 100% accuracy with seed {seed}")
                break
        
        self.assertEqual(best_accuracy, 1.0, 
                        f"Model accuracy must be 100%, got {best_accuracy*100:.2f}%")
        
        explainer = BooleanFunctionExplainer(best_model)
        explanation = explainer.explain(self.X_test[0].reshape(9))
        print("\nReconstructed x_many_consecutively_one function (x=3):")
        print(explanation['boolean_function'])

if __name__ == '__main__':
    unittest.main(verbosity=2)