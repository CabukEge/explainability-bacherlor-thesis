import numpy as np
import torch
from typing import Tuple, Dict
import os
from explainability_toolkit.boolean_functions_3x3 import first_example
from explainability_toolkit.data_generation import (
    generate_complete_dataset,
    split_dataset,
    generate_sampled_datasets,
    prepare_data
)
from explainability_toolkit.train import (
    decision_tree,
    train_fully_connected,
    train_cnn,
    SEED_LIST,
    ST_NUM_EPOCHS
)
from explainability_toolkit.utils import (
    archive_old_results,
    save_predictions
)

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
    
    # Save dataset information
    os.makedirs("results", exist_ok=True)
    dataset_info = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for split_name, (X, y) in dataset_info.items():
        print(f"\n{split_name.capitalize()} Set Statistics:")
        print(f"Total samples: {len(y)}")
        print(f"Positive samples: {torch.sum(y == 1).item()}")
        print(f"Negative samples: {torch.sum(y == 0).item()}")
    
    # Convert to numpy for decision tree
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    
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
    # Set to True to use all possible input combinations
    # Set to False to use the original sampling method
    USE_COMPLETE_DATASET = True
    
    fcn_models, cnn_models = main(use_complete_dataset=USE_COMPLETE_DATASET)