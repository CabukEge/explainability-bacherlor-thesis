import itertools
import numpy as np
import torch
from typing import Callable, List, Tuple
import random
from . import boolean_functions_3x3

def generate_complete_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset containing all possible 9-bit binary combinations.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) where X contains all possible binary vectors
        and y contains their corresponding labels
    """
    # Generate all possible 9-bit binary combinations
    all_vectors = list(itertools.product([0, 1], repeat=9))
    X = np.array(all_vectors)
    y = np.array([boolean_functions_3x3.first_example(v) for v in all_vectors])
    
    # Print dataset statistics
    print(f"Complete Dataset Statistics:")
    print(f"Total samples: {len(y)} (2^9 = 512 combinations)")
    print(f"Positive samples: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
    
    return X, y

def split_dataset(X: np.ndarray, 
                 y: np.ndarray, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15) -> Tuple[
                     Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray],
                     Tuple[np.ndarray, np.ndarray]]:
    """
    Splits dataset into train, validation, and test sets.
    
    Args:
        X: Input data
        y: Labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Three tuples containing (X, y) for train, validation, and test sets
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
    
    # Generate random indices for splitting
    indices = np.random.permutation(len(X))
    
    # Calculate split points
    train_split = int(len(X) * train_ratio)
    val_split = int(len(X) * (train_ratio + val_ratio))
    
    # Create the splits
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    return (X[train_indices], y[train_indices]), \
           (X[val_indices], y[val_indices]), \
           (X[test_indices], y[test_indices])

def generate_sampled_datasets(num_samples: int, 
                            label_function: Callable[[np.ndarray], bool],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15) -> Tuple[
                                List[Tuple[np.ndarray, int]],
                                List[Tuple[np.ndarray, int]],
                                List[Tuple[np.ndarray, int]]]:
    """
    Generates balanced, sampled datasets split into train/val/test.
    
    Args:
        num_samples: Total number of samples to generate
        label_function: Function to label the data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Three lists containing (vector, label) tuples for train, val, and test sets
    """
    # Generate balanced samples
    all_vectors = [np.array(vector) for vector in itertools.product([0, 1], repeat=9)]
    
    positive_samples = []
    negative_samples = []
    
    for vector in all_vectors:
        if label_function(vector):
            positive_samples.append((vector, 1))
        else:
            negative_samples.append((vector, 0))
    
    # Sample equally from each class
    samples_per_class = num_samples // 2
    positive_samples = random.sample(positive_samples, samples_per_class)
    negative_samples = random.sample(negative_samples, samples_per_class)
    
    # Combine and shuffle
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    # Split into train/val/test
    train_end = int(len(all_samples) * train_ratio)
    val_end = int(len(all_samples) * (train_ratio + val_ratio))
    
    return (all_samples[:train_end], 
            all_samples[train_end:val_end], 
            all_samples[val_end:])

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
            num_samples, boolean_functions_3x3.first_example,
            train_ratio, val_ratio, test_ratio
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