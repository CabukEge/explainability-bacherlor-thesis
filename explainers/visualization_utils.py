import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import seaborn as sns

def plot_grid_attribution(attributions: np.ndarray,
                         original_input: Optional[np.ndarray] = None,
                         prediction: Optional[int] = None,
                         title: Optional[str] = None,
                         cmap: str = 'RdBu',
                         show_values: bool = True) -> plt.Figure:
    """
    Plot attribution scores for a 3x3 grid.
    
    Args:
        attributions: 3x3 array of attribution scores
        original_input: Original 3x3 input grid (optional)
        prediction: Model's prediction (optional)
        title: Plot title (optional)
        cmap: Colormap to use
        show_values: Whether to show attribution values in cells
        
    Returns:
        matplotlib.Figure: The generated figure
    """
    fig, axes = plt.subplots(1, 2 if original_input is not None else 1,
                            figsize=(10 if original_input is not None else 5, 5))
    
    if original_input is None:
        ax = axes
    else:
        ax = axes[1]
        # Plot original input
        sns.heatmap(original_input.reshape(3, 3),
                   ax=axes[0],
                   cmap='binary',
                   cbar=False,
                   square=True,
                   annot=True,
                   fmt='.0f')
        axes[0].set_title('Original Input')
    
    # Plot attribution scores
    vmax = max(abs(attributions.max()), abs(attributions.min()))
    vmin = -vmax
    sns.heatmap(attributions.reshape(3, 3),
                ax=ax,
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                annot=show_values,
                fmt='.3f')
    
    if title:
        ax.set_title(title)
    elif prediction is not None:
        ax.set_title(f'Attribution Scores\nPrediction: {prediction}')
    else:
        ax.set_title('Attribution Scores')
    
    plt.tight_layout()
    return fig

def plot_multiple_attributions(explanations: List[Dict[str, Union[np.ndarray, int]]],
                             original_inputs: Optional[np.ndarray] = None,
                             method_name: Optional[str] = None,
                             max_cols: int = 4) -> plt.Figure:
    """
    Plot attribution scores for multiple inputs.
    
    Args:
        explanations: List of explanation dictionaries
        original_inputs: Original input grids (optional)
        method_name: Name of explanation method (optional)
        max_cols: Maximum number of columns in the plot
        
    Returns:
        matplotlib.Figure: The generated figure
    """
    n_samples = len(explanations)
    n_cols = min(n_samples, max_cols)
    n_rows = (n_samples - 1) // max_cols + 1
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    
    for i, explanation in enumerate(explanations):
        plt.subplot(n_rows, n_cols, i + 1)
        
        attributions = explanation['attributions']
        prediction = explanation.get('prediction')
        original_input = original_inputs[i] if original_inputs is not None else None
        
        if original_input is not None:
            # Plot original input
            plt.subplot(n_rows, n_cols * 2, i * 2 + 1)
            sns.heatmap(original_input.reshape(3, 3),
                       cmap='binary',
                       cbar=False,
                       square=True,
                       annot=True,
                       fmt='.0f')
            plt.title('Original Input')
            
            # Plot attribution scores
            plt.subplot(n_rows, n_cols * 2, i * 2 + 2)
        
        vmax = max(abs(attributions.max()), abs(attributions.min()))
        vmin = -vmax
        sns.heatmap(attributions,
                    cmap='RdBu',
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    square=True,
                    annot=True,
                    fmt='.3f')
        
        if prediction is not None:
            plt.title(f'Attribution Scores\nPrediction: {prediction}')
        else:
            plt.title('Attribution Scores')
    
    if method_name:
        fig.suptitle(method_name, fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig