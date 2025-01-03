"""
Explainability Toolkit for analyzing machine learning models
on boolean functions with 3x3 grid inputs.
"""

from .boolean_functions_3x3 import (
    first_example,
    min_x_amount_equals_one,
    x_many_consecutively_one
)
from .data_generation import (
    generate_complete_dataset,
    split_dataset,
    generate_sampled_datasets,
    prepare_data
)
from .models import (
    SimpleNet,
    ImprovedCNN
)
from .train import (
    decision_tree,
    train_fully_connected,
    train_cnn,
    SEED_LIST,
    ST_NUM_EPOCHS
)
from .utils import (
    archive_old_results,
    save_predictions,
    vector_to_string
)
from .explainers.base_explainer import BaseExplainer
from .explainers.lime_explainer import LimeExplainer
from .explainers.visualization_utils import (
    plot_grid_attribution,
    plot_multiple_attributions
)

__all__ = [
    # Boolean functions
    "first_example",
    "min_x_amount_equals_one",
    "x_many_consecutively_one",
    
    # Data generation
    "generate_complete_dataset",
    "split_dataset",
    "generate_sampled_datasets",
    "prepare_data",
    
    # Models
    "SimpleNet",
    "ImprovedCNN",
    
    # Training
    "decision_tree",
    "train_fully_connected",
    "train_cnn",
    "SEED_LIST",
    "ST_NUM_EPOCHS",
    
    # Utilities
    "archive_old_results",
    "save_predictions",
    "vector_to_string",
    
    # Explainers
    "BaseExplainer",
    "LimeExplainer",
    "plot_grid_attribution",
    "plot_multiple_attributions"
]

__version__ = "0.1.0"