import unittest
import numpy as np
import torch
from explainability_toolkit.data_generation import prepare_data
from explainability_toolkit.models import SimpleNet
from explainability_toolkit.train import train_fully_connected, SEED_LIST
from explainability_toolkit.explainers.lime_explainer import LimeExplainer
from explainability_toolkit.explainers.ground_truth_explainer import GroundTruthExplainer
from explainability_toolkit.utils import ResultsLogger

class TestLIMEExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize logger
        cls.logger = ResultsLogger()
        
        # Prepare data and train model
        (cls.X_train, cls.y_train), (cls.X_val, cls.y_val), (cls.X_test, cls.y_test) = \
            prepare_data(use_complete_dataset=True)
            
        losses, models = train_fully_connected(
            cls.X_train, cls.y_train,
            seed_list=SEED_LIST[:1],
            num_epochs=500
        )
        cls.model = models[0]
        cls.model.eval()
        
        # Initialize explainers
        cls.lime_explainer = LimeExplainer(cls.model, num_samples=2000)
        cls.ground_truth = GroundTruthExplainer('first_example')
        
        # Test cases
        cls.test_cases = {
            'x1_x2': torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0]).float(),
            'x4_x5_x6': torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0]).float(),
            'x7_x8_x9': torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1]).float(),
            'all_zeros': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0]).float(),
            'all_ones': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]).float()
        }
    
    def test_basic_functionality(self):
        """Test basic LIME explanation generation."""
        for case_name, test_input in self.test_cases.items():
            lime_exp = self.lime_explainer.explain(test_input)
            gt_exp = self.ground_truth.explain(test_input)
            
            self.logger.log_explanation(
                input_data=test_input,
                ground_truth=gt_exp,
                predicted=lime_exp,
                method_name="LIME",
                case_name=f"basic_{case_name}"
            )
            
            self.assertIn('attributions', lime_exp)
            self.assertEqual(lime_exp['attributions'].shape, (3, 3))
    def test_explanation_fidelity(self):
        """Test if LIME explanations match model behavior."""
        for case_name, test_input in self.test_cases.items():
            explanation = self.lime_explainer.explain(test_input)
            
            # Get model prediction
            with torch.no_grad():
                pred = self.model(test_input.reshape(1, 3, 3)).argmax().item()
            
            # Check if highest attribution matches prediction
            if pred == 1:
                self.assertTrue(np.max(explanation['attributions']) > 0)
            else:
                self.assertTrue(np.min(explanation['attributions']) < 0)
                
    def test_ground_truth_comparison(self):
        """Compare LIME explanations with ground truth."""
        for case_name, test_input in self.test_cases.items():
            lime_exp = self.lime_explainer.explain(test_input)
            gt_exp = self.ground_truth.explain(test_input)
            
            lime_attrs = lime_exp['attributions']
            gt_attrs = gt_exp['attributions']
            
            # Compare attribution patterns
            if case_name in ['x1_x2', 'x4_x5_x6', 'x7_x8_x9']:
                # Check if LIME identifies the correct important features
                lime_important = np.where(np.abs(lime_attrs) > np.mean(np.abs(lime_attrs)))
                gt_important = np.where(gt_attrs != 0)
                self.assertTrue(
                    np.intersect1d(lime_important[0], gt_important[0]).size > 0,
                    f"LIME failed to identify important features for {case_name}"
                )
                
    def test_stability(self):
        """Test stability of LIME explanations."""
        test_input = self.test_cases['x1_x2']
        explanations = [
            self.lime_explainer.explain(test_input)['attributions']
            for _ in range(5)
        ]
        
        # Check consistency across runs
        base_exp = explanations[0]
        for exp in explanations[1:]:
            correlation = np.corrcoef(
                base_exp.flatten(),
                exp.flatten()
            )[0, 1]
            self.assertGreater(
                correlation,
                0.5,
                "LIME explanations show high variance across runs"
            )
            
    def test_edge_cases(self):
        """Test LIME behavior on edge cases."""
        edge_cases = {
            'alternating': torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1]).float(),
            'single_one': torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]).float(),
            'checkerboard': torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1]).float()
        }
        
        for case_name, test_input in edge_cases.items():
            explanation = self.lime_explainer.explain(test_input)
            self.assertIsNotNone(explanation['attributions'])
            self.assertTrue(np.isfinite(explanation['attributions']).all())

if __name__ == '__main__':
    unittest.main(verbosity=2)