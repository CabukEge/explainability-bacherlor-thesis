"""
Visualization module for explainability experiments.

This module provides functions to visualize experiment results:
1. Training curves for model training
2. Random sampling accuracy comparisons
3. Method comparison radar charts
4. Training regime impact visualizations
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

def plot_training_curves(train_acc, val_acc, model_name, function_name, regime):
    """
    Generate training and validation accuracy curves.
    
    Args:
        train_acc: List of training accuracies over epochs
        val_acc: List of validation accuracies over epochs
        model_name: Name of the model architecture
        function_name: Name of the Boolean function
        regime: Training regime (normal or overfitted)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation accuracy
    plt.plot(range(1, len(train_acc) + 1), train_acc, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'r-', label='Validation Accuracy')
    
    # Add annotations and formatting
    plt.title(f'{model_name} Training on {function_name}: {regime} Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Add horizontal line at 100% accuracy
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # Save the figure
    plt.tight_layout()
    filename = f'artifacts/training_curves_{model_name}_{function_name}_{regime}.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Return the figure path for reference
    return filename

def plot_random_sampling_accuracy(results, model_names, explainer_names, boolean_functions):
    """
    Create a bar chart showing reconstruction accuracy for random sampling approach.
    
    Args:
        results: Dictionary containing accuracy results
        model_names: List of model architecture names
        explainer_names: List of explainer method names
        boolean_functions: List of Boolean function names
    """
    # Prepare data for visualization
    data = {'Model-Explainer': [], 'Accuracy': [], 'Function': []}
    
    for func_name, func_results in results.items():
        if 'approach2' in func_name:
            base_name = func_name.split('_approach2')[0]
            for model_explainer, accuracy in func_results.items():
                # Extract accuracy from tuple if needed
                if isinstance(accuracy, tuple) and len(accuracy) >= 1:
                    accuracy = accuracy[0]
                
                data['Model-Explainer'].append(model_explainer)
                data['Accuracy'].append(accuracy)
                data['Function'].append(base_name)
    
    df = pd.DataFrame(data)
    
    # If no data for approach2, try approach3
    if len(df) == 0:
        data = {'Model-Explainer': [], 'Accuracy': [], 'Function': []}
        for func_name, func_results in results.items():
            if 'approach3' in func_name:
                base_name = func_name.split('_approach3')[0]
                for model_explainer, result in func_results.items():
                    # Extract accuracy from tuple if needed
                    if isinstance(result, tuple) and len(result) >= 1:
                        accuracy = result[0]
                    else:
                        accuracy = result
                    
                    data['Model-Explainer'].append(model_explainer)
                    data['Accuracy'].append(accuracy)
                    data['Function'].append(base_name)
        df = pd.DataFrame(data)
    
    # Calculate mean and std for error bars
    summary = df.groupby('Model-Explainer')['Accuracy'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(summary['Model-Explainer'], summary['mean'], yerr=summary['std'], 
                   capsize=5, color=sns.color_palette('muted', len(summary)))
    
    plt.title('Reconstruction Accuracy for Random Sampling Approach')
    plt.xlabel('Model-Explainer Combination')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    # Add exact values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filename = 'artifacts/random_sampling_accuracy.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return filename

def radar_chart(results, model_names, explainer_names, training_regimes):
    """
    Create a radar chart comparing performance of different explainer methods.
    
    Args:
        results: Dictionary containing accuracy results
        model_names: List of model architecture names
        explainer_names: List of explainer methods
        training_regimes: List of training regimes
    """
    # Prepare data for radar chart
    # Get all valid model-explainer combinations
    model_explainer_combos = []
    for model in model_names:
        for explainer in explainer_names:
            # Skip invalid combinations
            if model in ['FCN', 'CNN'] and explainer == 'TreeSHAP':
                continue
            if model == 'Decision Tree' and explainer == 'Integrated Gradients':
                continue
            model_explainer_combos.append(f"{model}-{explainer}")
    
    # Aggregate results for exhaustive approach
    data = {}
    for combo in model_explainer_combos:
        data[combo] = {'normal': [], 'overfitted': []}
    
    for func_name, func_results in results.items():
        if 'approach3' in func_name:
            for model_explainer, result in func_results.items():
                # Extract score from different possible result formats
                if isinstance(result, (tuple, list)) and len(result) >= 1:
                    score = result[0]
                else:
                    score = result
                
                # Determine regime and clean model name
                if '(Overfitted)' in model_explainer:
                    regime = 'overfitted'
                    clean_name = model_explainer.replace('(Overfitted)', '').strip()
                else:
                    regime = 'normal'
                    clean_name = model_explainer
                
                if clean_name in model_explainer_combos:
                    data[clean_name][regime].append(score)
        
        # If no approach3 data, try approach2
        elif 'approach2' in func_name and not any(data[combo]['normal'] or data[combo]['overfitted'] 
                                                 for combo in model_explainer_combos):
            for model_explainer, result in func_results.items():
                # Extract score from different possible result formats
                if isinstance(result, tuple) and len(result) >= 1:
                    score = result[0]
                else:
                    score = result
                
                # Determine regime and clean model name
                if '(Overfitted)' in model_explainer:
                    regime = 'overfitted'
                    clean_name = model_explainer.replace('(Overfitted)', '').strip()
                else:
                    regime = 'normal'
                    clean_name = model_explainer
                
                if clean_name in model_explainer_combos:
                    data[clean_name][regime].append(score)
    
    # Calculate averages
    averages = {}
    for combo in model_explainer_combos:
        averages[combo] = {
            'normal': np.mean(data[combo]['normal']) if data[combo]['normal'] else 0,
            'overfitted': np.mean(data[combo]['overfitted']) if data[combo]['overfitted'] else 0,
        }
    
    # Setup radar chart
    categories = model_explainer_combos
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add lines for each training regime
    for regime, color, line_style in zip(['normal', 'overfitted'], ['blue', 'red'], ['-', '--']):
        values = [averages[combo][regime] for combo in model_explainer_combos]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle=line_style, 
                label=f"{regime.capitalize()} Training", color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=8)
    
    # Draw y-axis lines from center to outer edge
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Comparison of Explanation Methods Across Models and Training Regimes", size=12, y=1.1)
    
    plt.tight_layout()
    filename = 'artifacts/method_comparison.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return filename

def plot_overfitting_impact(results, model_names, explainer_names):
    """
    Create a grouped bar chart showing impact of training regime.
    
    Args:
        results: Dictionary containing accuracy results
        model_names: List of model architecture names
        explainer_names: List of explainer method names
    """
    # Prepare data
    data = {'Explainer': [], 'Accuracy (Normal)': [], 'Accuracy (Overfitted)': []}
    
    # Debug information to help identify issues
    print("Available keys in results dictionary:")
    for key in results.keys():
        print(f"  - {key}")
    
    # Gather data for explainers across all models for exhaustive approach
    for explainer in explainer_names:
        normal_acc = []
        overfitted_acc = []
        
        for func_name, func_results in results.items():
            # Focus on approach3 as it should have the most comprehensive data
            if 'approach3' in func_name:
                # Looking for overfitted data in the function name
                is_overfitted_data = 'overfitted' in func_name.lower()
                
                print(f"\nAnalyzing {func_name} for {explainer} (is_overfitted_data={is_overfitted_data})")
                
                for model_explainer, result in func_results.items():
                    # Only consider this explainer
                    if explainer in model_explainer:
                        # Extract score correctly
                        if isinstance(result, tuple) and len(result) >= 1:
                            score = result[0]
                        else:
                            score = result
                        
                        print(f"  Model-explainer: {model_explainer}, score: {score}")
                        
                        # The key issue: determine if this is an overfitted model result
                        # Method 1: Check explicit marker in model name (original)
                        has_overfitted_marker = '(Overfitted)' in model_explainer
                        
                        # Method 2: Use the function name marker (new)
                        if is_overfitted_data:
                            overfitted_acc.append(score)
                            print(f"    → Added to overfitted (from function name)")
                        else:
                            normal_acc.append(score)
                            print(f"    → Added to normal (from function name)")
        
        if normal_acc or overfitted_acc:
            normal_mean = np.mean(normal_acc) if normal_acc else 0
            overfitted_mean = np.mean(overfitted_acc) if overfitted_acc else 0
            
            print(f"\nExplainer {explainer} final means:")
            print(f"  Normal: {normal_mean:.4f} (from {len(normal_acc)} values)")
            print(f"  Overfitted: {overfitted_mean:.4f} (from {len(overfitted_acc)} values)")
            
            data['Explainer'].append(explainer)
            data['Accuracy (Normal)'].append(normal_mean)
            data['Accuracy (Overfitted)'].append(overfitted_mean)
    
    # Create the plot (with strict 0-1 y-axis)
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set position of bars on X axis
    r1 = np.arange(len(df['Explainer']))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, df['Accuracy (Normal)'], width=barWidth, edgecolor='grey', 
            label='Normal Training', color='skyblue')
    plt.bar(r2, df['Accuracy (Overfitted)'], width=barWidth, edgecolor='grey', 
            label='Overfitted Training', color='salmon')
    
    # Add labels and legend
    plt.xlabel('Explainability Method', fontweight='bold')
    plt.ylabel('Average Reconstruction Accuracy')
    plt.xticks([r + barWidth/2 for r in range(len(df['Explainer']))], df['Explainer'])
    plt.legend()
    
    # Add error bars
    if normal_acc and overfitted_acc:
        plt.errorbar(r1, df['Accuracy (Normal)'], 
                    yerr=[np.std(normal_acc) if len(normal_acc) > 1 else 0.05 for _ in range(len(r1))], 
                    fmt='none', ecolor='black', capsize=5)
        plt.errorbar(r2, df['Accuracy (Overfitted)'], 
                    yerr=[np.std(overfitted_acc) if len(overfitted_acc) > 1 else 0.05 for _ in range(len(r2))], 
                    fmt='none', ecolor='black', capsize=5)
    else:
        # Fallback to standard error if we don't have enough data
        std_dev = 0.05
        plt.errorbar(r1, df['Accuracy (Normal)'], yerr=std_dev, fmt='none', ecolor='black', capsize=5)
        plt.errorbar(r2, df['Accuracy (Overfitted)'], yerr=std_dev, fmt='none', ecolor='black', capsize=5)
    
    plt.title('Impact of Training Regime on Reconstruction Accuracy')
    plt.ylim(0, 1.0)  # Strict upper limit
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    filename = 'artifacts/overfitting_impact.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    
    return filename