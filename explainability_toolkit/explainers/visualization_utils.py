import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_attribution(attributions: np.ndarray,
                         original_input: Optional[np.ndarray] = None,
                         prediction: Optional[int] = None,
                         title: Optional[str] = None,
                         cmap: str = 'RdBu',
                         show_values: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot attribution scores for a 3x3 grid.
    """
    plt.style.use('seaborn')
    
    fig, axes = plt.subplots(1, 2 if original_input is not None else 1,
                            figsize=(12 if original_input is not None else 6, 5))
    
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
                   fmt='.0f',
                   xticklabels=False,
                   yticklabels=False)
        axes[0].set_title('Original Input')
    
    # Plot attribution scores
    vmax = max(abs(attributions.max()), abs(attributions.min()))
    vmin = -vmax
    
    if show_values:
        annot = [[f'{x:.3f}' for x in row] for row in attributions.reshape(3, 3)]
    else:
        annot = False
    
    sns.heatmap(attributions.reshape(3, 3),
                ax=ax,
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                annot=annot,
                fmt='',
                xticklabels=False,
                yticklabels=False)
    
    if title:
        ax.set_title(title)
    elif prediction is not None:
        ax.set_title(f'Attribution Scores\nPrediction: {prediction}')
    else:
        ax.set_title('Attribution Scores')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        return None
    
    return fig

def plot_multiple_attributions(explanations: List[Dict[str, Union[np.ndarray, int]]],
                             original_inputs: Optional[np.ndarray] = None,
                             method_name: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot attribution scores for multiple inputs.
    """
    plt.style.use('seaborn')
    n_samples = len(explanations)
    
    # Create a figure with subplots for each example
    n_rows = n_samples
    fig = plt.figure(figsize=(12, 4 * n_rows))
    
    for i, explanation in enumerate(explanations):
        # Create side-by-side plots for each example
        original_ax = plt.subplot(n_rows, 2, 2*i + 1)
        attribution_ax = plt.subplot(n_rows, 2, 2*i + 2)
        
        # Plot original input
        if original_inputs is not None:
            sns.heatmap(original_inputs[i],
                       ax=original_ax,
                       cmap='binary',
                       cbar=False,
                       square=True,
                       annot=True,
                       fmt='.0f',
                       xticklabels=False,
                       yticklabels=False)
            original_ax.set_title(f'Original Input (Prediction: {explanation["prediction"]})')
        
        # Plot attribution scores
        attributions = explanation['attributions']
        vmax = max(abs(attributions.max()), abs(attributions.min()))
        vmin = -vmax
        
        annot = [[f'{x:.3f}' for x in row] for row in attributions.reshape(3, 3)]
        
        sns.heatmap(attributions.reshape(3, 3),
                   ax=attribution_ax,
                   cmap='RdBu',
                   center=0,
                   vmin=vmin,
                   vmax=vmax,
                   square=True,
                   annot=annot,
                   fmt='',
                   xticklabels=False,
                   yticklabels=False)
        attribution_ax.set_title(f'Attribution Scores (Local Acc: {explanation["local_model_accuracy"]:.3f})')
    
    if method_name:
        plt.suptitle(method_name, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        return None
    
    return fig