import os
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
from typing import Union, Dict

def archive_old_results(output_dir: str = "results") -> None:
    """
    Archive old results into a timestamped folder.
    
    Args:
        output_dir: Directory containing results to archive
    """
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
    """
    Save predictions and evaluation metrics to files.
    
    Args:
        X: Input vectors
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        seed: Random seed used (if applicable)
        output_dir: Directory to save results
        
    Returns:
        Dict containing paths to saved files
    """
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