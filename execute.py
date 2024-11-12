
import numpy as np
import torch
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
import os
import shutil
from pathlib import Path
from explainability_toolkit.boolean_functions_3x3 import first_example
from explainability_toolkit.data_generation import (
    generate_complete_dataset,
    split_dataset,
    generate_sampled_datasets
)
from explainability_toolkit.train import (
    decision_tree,
    train_fully_connected,
    train_cnn,
    SEED_LIST,
    ST_NUM_EPOCHS
)

def prepare_data(num_samples: int = 200, 
                use_complete_dataset: bool = False,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15) -> Tuple[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor]]:
    """
    Prepare datasets for training, validation, and testing.
    
    Args:
        num_samples: Number of samples for sampled dataset
        use_complete_dataset: Whether to use all possible combinations
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    
    Returns:
        Three tuples containing (X, y) tensors for train, validation, and test sets
    """
    if use_complete_dataset:
        # Generate complete dataset
        X, y = generate_complete_dataset()
        # Split into train/val/test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
            X, y, train_ratio, val_ratio, test_ratio
        )
    else:
        # Use sampled datasets
        train_data, val_data, test_data = generate_sampled_datasets(
            num_samples, first_example, train_ratio, val_ratio, test_ratio
        )
        
        # Convert to numpy arrays
        X_train = np.array([data[0] for data in train_data])
        y_train = np.array([data[1] for data in train_data])
        X_val = np.array([data[0] for data in val_data])
        y_val = np.array([data[1] for data in val_data])
        X_test = np.array([data[0] for data in test_data])
        y_test = np.array([data[1] for data in test_data])
    
    # Convert to PyTorch tensors
    train_tensors = (
        torch.FloatTensor(X_train.reshape(-1, 3, 3)),
        torch.LongTensor(y_train)
    )
    val_tensors = (
        torch.FloatTensor(X_val.reshape(-1, 3, 3)),
        torch.LongTensor(y_val)
    )
    test_tensors = (
        torch.FloatTensor(X_test.reshape(-1, 3, 3)),
        torch.LongTensor(y_test)
    )
    
    return train_tensors, val_tensors, test_tensors

def archive_old_results(output_dir: str = "results"):
    """
    Archive old results into a timestamped folder.
    
    Args:
        output_dir: Directory containing results to archive
    """
    if not os.path.exists(output_dir):
        return
        
    # Get list of files in results directory
    files = os.listdir(output_dir)
    if not files:
        return
        
    # Create archive directory with timestamp
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(output_dir, "archive", archive_timestamp)
    os.makedirs(archive_dir, exist_ok=True)
    
    # Move all files to archive
    for file in files:
        if file != "archive":  # Don't move the archive directory itself
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
                    seed: int = None,
                    output_dir: str = "results"):
    """
    Save predictions to a CSV file.
    
    Args:
        X: Input vectors
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        seed: Random seed used (if applicable)
        output_dir: Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'input_vector': [vector_to_string(v) for v in X],
        'input_grid': [str(v.reshape(3, 3)) for v in X],
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct_prediction': y_true == y_pred
    })
    
    # Add statistics
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / total_samples
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_str = f"_seed_{seed}" if seed is not None else ""
    filename = f"{model_name}{seed_str}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f"{model_name}_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {seed if seed is not None else 'N/A'}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        
        # Calculate confusion matrix
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

def main(use_complete_dataset: bool = True):
    """
    Main execution function with result logging.
    
    Args:
        use_complete_dataset: Whether to use all possible input combinations
    """
    # Archive old results before starting new run
    archive_old_results()
    
    # Prepare data with splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(
        use_complete_dataset=use_complete_dataset
    )
    
    # Save dataset split information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_info_file = os.path.join("results", f"dataset_info_{timestamp}.txt")
    with open(dataset_info_file, 'w') as f:
        f.write("Dataset Information:\n")
        f.write(f"Using complete dataset: {use_complete_dataset}\n\n")
        
        f.write("Training Set:\n")
        f.write(f"Samples: {len(y_train)}\n")
        f.write(f"Positive samples: {torch.sum(y_train == 1).item()}\n")
        f.write(f"Negative samples: {torch.sum(y_train == 0).item()}\n\n")
        
        f.write("Validation Set:\n")
        f.write(f"Samples: {len(y_val)}\n")
        f.write(f"Positive samples: {torch.sum(y_val == 1).item()}\n")
        f.write(f"Negative samples: {torch.sum(y_val == 0).item()}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"Samples: {len(y_test)}\n")
        f.write(f"Positive samples: {torch.sum(y_test == 1).item()}\n")
        f.write(f"Negative samples: {torch.sum(y_test == 0).item()}\n")
    
    # Convert to numpy for decision tree
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    
    # Rest of the training and evaluation code...
    # Decision Tree
    print("\nTraining Decision Tree...")
    dt_model = decision_tree(X_train_np.reshape(len(X_train_np), -1), y_train_np)
    dt_predictions = dt_model.predict(X_test_np.reshape(len(X_test_np), -1))
    save_predictions(X_test_np, y_test_np, dt_predictions, "decision_tree")
    
    # Fully Connected Network
    print("\nTraining Fully Connected Network...")
    fcn_losses, fcn_models = train_fully_connected(X_train, y_train, SEED_LIST, ST_NUM_EPOCHS)
    for i, model in enumerate(fcn_models):
        with torch.no_grad():
            fcn_predictions = model(X_test).argmax(dim=1).numpy()
            save_predictions(X_test_np, y_test_np, fcn_predictions, "fcn", SEED_LIST[i])
    
    # CNN with validation
    print("\nTraining CNN...")
    train_losses, val_losses, cnn_models = train_cnn(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        seed_list=SEED_LIST, 
        num_epochs=ST_NUM_EPOCHS
    )
    
    for i, model in enumerate(cnn_models):
        with torch.no_grad():
            model.eval()
            X_test_cnn = X_test.unsqueeze(1)
            cnn_predictions = model(X_test_cnn).argmax(dim=1).numpy()
            save_predictions(X_test_np, y_test_np, cnn_predictions, "cnn", SEED_LIST[i])
    
    return fcn_models, cnn_models

if __name__ == "__main__":
    USE_COMPLETE_DATASET = True
    fcn_models, cnn_models = main(use_complete_dataset=USE_COMPLETE_DATASET)