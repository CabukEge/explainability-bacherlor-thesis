import itertools
import numpy as np
from typing import Callable, List, Tuple
import random
from . import boolean_functions_3x3

def generate_complete_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates all 512 possible combinations for a 9-bit input.
    """
    all_vectors = list(itertools.product([0, 1], repeat=9))
    X = np.array(all_vectors)
    y = np.array([boolean_functions_3x3.first_example(v) for v in all_vectors])
    
    print(f"Complete Dataset Statistics:")
    print(f"Total samples: {len(y)} (2^9 = 512 combinations)")
    print(f"Positive samples: {np.sum(y == 1)}")
    print(f"Negative samples: {np.sum(y == 0)}")
    
    return X, y

def split_dataset(X: np.ndarray, 
                 y: np.ndarray, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                Tuple[np.ndarray, np.ndarray],
                                                Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the dataset into train, validation, and test sets.
    
    Args:
        X: Input vectors
        y: Labels
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Three tuples containing (X, y) for train, validation, and test sets
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
        "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate random indices
    indices = np.random.permutation(len(X))
    
    # Calculate split points
    train_split = int(len(X) * train_ratio)
    val_split = int(len(X) * (train_ratio + val_ratio))
    
    # Split indices
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # Create the splits
    train_set = (X[train_indices], y[train_indices])
    val_set = (X[val_indices], y[val_indices])
    test_set = (X[test_indices], y[test_indices])
    
    # Print statistics
    print("\nDataset Split Statistics:")
    print(f"Training set: {len(train_indices)} samples "
          f"({len(train_indices)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(val_indices)} samples "
          f"({len(val_indices)/len(X)*100:.1f}%)")
    print(f"Test set: {len(test_indices)} samples "
          f"({len(test_indices)/len(X)*100:.1f}%)")
    
    return train_set, val_set, test_set

def generate_sampled_datasets(num_samples: int, 
                            label_function: Callable[[np.ndarray], bool],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15) -> Tuple[List[Tuple[np.ndarray, bool]], 
                                                             List[Tuple[np.ndarray, bool]],
                                                             List[Tuple[np.ndarray, bool]]]:
    """
    Generates balanced, sampled datasets split into train/val/test.
    
    Args:
        num_samples: Total number of samples to generate
        label_function: Function to label the data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
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
    
    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]
    
    print("\nSampled Dataset Statistics:")
    print(f"Total samples: {num_samples}")
    print(f"Training: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data