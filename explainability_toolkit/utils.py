import os
import shutil
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Union, Dict, Any, Optional, List

def archive_old_results(output_dir: str = "results") -> None:
    """Archive old results into a timestamped folder."""
    if not os.path.exists(output_dir):
        return
        
    files = os.listdir(output_dir)
    if not files:
        return
        
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(output_dir, "archive", archive_timestamp)
    os.makedirs(archive_dir, exist_ok=True)
    
    for file in files:
        if file != "archive":
            src = os.path.join(output_dir, file)
            dst = os.path.join(archive_dir, file)
            shutil.move(src, dst)
    
    print(f"Archived old results to: {archive_dir}")

def vector_to_string(vector: np.ndarray) -> str:
    """Convert a binary vector to a readable string format."""
    return ''.join(map(str, vector.astype(int)))

def save_predictions(X: np.ndarray, 
                    y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    model_name: str,
                    seed: Union[int, None] = None,
                    output_dir: str = "results") -> Dict[str, str]:
    """Save predictions and evaluation metrics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'input_vector': [vector_to_string(v) for v in X],
        'input_grid': [str(v.reshape(3, 3)) for v in X],
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct_prediction': y_true == y_pred
    })
    
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / total_samples
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"_seed_{seed}" if seed is not None else ""
    filename = f"{model_name}{seed_str}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    results_df.to_csv(filepath, index=False)
    
    summary_file = os.path.join(output_dir, f"{model_name}_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {seed if seed is not None else 'N/A'}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        f.write(f"True Positive: {tp}\n")
        f.write(f"True Negative: {tn}\n")
        f.write(f"False Positive: {fp}\n")
        f.write(f"False Negative: {fn}\n")
    
    print(f"Results saved to {filepath}")
    print(f"Summary saved to {summary_file}")
    
    return {
        'results': filepath,
        'summary': summary_file
    }

class ResultsLogger:
    """Logger for explanation results with visualizations and metrics."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _find_conjunctive_terms(self, attributions: np.ndarray, threshold: float = 0.1) -> List[List[int]]:
        """Find groups of variables that should be conjoined based on similar attribution values."""
        terms = []
        processed = set()
        flat_attrs = attributions.flatten()
        scaled_attrs = flat_attrs / np.max(np.abs(flat_attrs)) if np.max(np.abs(flat_attrs)) > 0 else flat_attrs
        
        # Sort indices by attribution value
        sorted_indices = np.argsort(-np.abs(scaled_attrs))
        
        current_term = []
        current_value = None
        
        for idx in sorted_indices:
            if idx in processed:
                continue
                
            value = scaled_attrs[idx]
            if abs(value) < threshold:
                continue
                
            if current_value is None:
                current_term = [idx]
                current_value = value
            elif np.isclose(value, current_value, rtol=0.1):  # Similar magnitude and same sign
                current_term.append(idx)
            else:
                if current_term:
                    terms.append(current_term)
                current_term = [idx]
                current_value = value
                
            processed.add(idx)
            
        if current_term:
            terms.append(current_term)
            
        return terms
        
    def _format_boolean_expression(self, attributions: np.ndarray, threshold: float = 0.1) -> str:
        """Format attribution scores as DNF boolean expression."""
        # Find groups of variables that should be conjoined
        terms = self._find_conjunctive_terms(attributions)
        
        if not terms:
            return "False"
            
        # Format each conjunctive term
        dnf_terms = []
        for term_indices in terms:
            if not term_indices:
                continue
                
            # Get the variables in this term
            vars_in_term = [f"x{i+1}" for i in term_indices]
            
            # Create the conjunction
            if len(vars_in_term) == 1:
                dnf_terms.append(vars_in_term[0])
            else:
                dnf_terms.append(f"({' ∧ '.join(vars_in_term)})")
        
        # Join terms with OR
        if not dnf_terms:
            return "False"
        elif len(dnf_terms) == 1:
            return dnf_terms[0]
        else:
            return " ∨ ".join(dnf_terms)
    
    def log_explanation(self, 
                       input_data: torch.Tensor,
                       ground_truth: Dict[str, Any],
                       predicted: Dict[str, Any],
                       method_name: str,
                       case_name: Optional[str] = None):
        case_id = case_name or datetime.now().strftime("%H%M%S")
        case_dir = os.path.join(self.log_dir, f"{method_name}_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(ground_truth['attributions'], ax=ax1, cmap='RdBu', center=0,
                   annot=True, fmt='.2f')
        ax1.set_title('Ground Truth Attribution')

        sns.heatmap(predicted['attributions'], ax=ax2, cmap='RdBu', center=0,
                   annot=True, fmt='.2f')
        ax2.set_title(f'{method_name} Attribution')

        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, 'attribution_comparison.png'))
        plt.close()

        results = {
            'input': input_data.numpy().tolist(),
            'ground_truth': {
                'boolean_expression': self._format_boolean_expression(ground_truth['attributions']),
                'attributions': ground_truth['attributions'].tolist(),
                'function_type': ground_truth['function_type']
            },
            'predicted': {
                'boolean_expression': predicted['boolean_expression'],
                'attributions': predicted['attributions'].tolist(),
                'local_model_accuracy': predicted.get('local_model_accuracy', None),
                'prediction': predicted.get('prediction', None)
            },
            'metrics': {
                'correlation': float(np.corrcoef(
                    ground_truth['attributions'].flatten(),
                    predicted['attributions'].flatten()
                )[0, 1]),
                'mse': float(np.mean((
                    ground_truth['attributions'] - predicted['attributions']
                ) ** 2))
            }
        }

        with open(os.path.join(case_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(case_dir, 'summary.txt'), 'w') as f:
            f.write(f"Explanation Summary for {method_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Input Pattern:\n{input_data.numpy().reshape(3,3)}\n\n")
            f.write("Ground Truth Expression:\n")
            f.write(f"{results['ground_truth']['boolean_expression']}\n\n")
            f.write(f"Predicted Expression:\n")
            f.write(f"{results['predicted']['boolean_expression']}\n\n")
            f.write("Metrics:\n")
            f.write(f"Correlation: {results['metrics']['correlation']:.4f}\n")
            f.write(f"MSE: {results['metrics']['mse']:.4f}\n")
            if 'local_model_accuracy' in predicted:
                f.write(f"Local Model Accuracy: {predicted['local_model_accuracy']:.4f}\n")