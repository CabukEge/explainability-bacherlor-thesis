#!/usr/bin/env python
"""
run_tests.py

This script runs three different approaches to test the explainers.

Approach 1 (Selective):
  - Assumes the target DNF is known.
  - Uses the known terms to construct inputs that are guaranteed to be positive for each term.

Approach 2 (Random Sampling):
  - Uses 50 random samples chosen from the 512 possible 9-bit combinations.
  - Additionally, it aggregates summary statistics of the explanation values:
      • For inputs that trigger "true" (prediction ≥ 0.5): records the highest and lowest explanation value.
      • For inputs that trigger "false": records the highest explanation value.
      • Then, it computes an adaptive threshold as (true_max_mean + false_max_mean)/2 (if false_max_mean is available; otherwise 0.1).
      • This threshold is then used for extracting active features.
      
Approach 3 (Exhaustive):
  - Uses all 512 possible inputs.
  - Aggregates explanation stats in the same manner and uses the adaptive threshold.
  - If a timeout is reached, it logs how many inputs were processed.

All outputs are logged both to the terminal and to a log file in artifacts/run_tests.log.
Reconstruction matrices (the reconstructed DNF and metrics) are saved as artifacts.
"""

import os
import sys
import time
import signal
import argparse
import random
import torch
import numpy as np
import json
from itertools import product

import logging
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/run_tests.log", mode='w')
    ]
)
logger = logging.getLogger()

from boolean_functions import dnf_simple, dnf_example, dnf_complex
from data import generate_data
from models import FCN, CNN, train_model, train_tree
from evaluate import (
    parse_dnf_to_terms,
    get_function_str,
    verify_term,
    verify_term_is_minimal,
    compute_term_metrics,
    are_dnfs_equivalent,
    create_test_case_for_term
)
from explainers import (
    LIMEExplainer,
    KernelSHAPExplainer,
    IntegratedGradientsExplainer,
    TreeSHAPExplainer
)
from pprint import pformat  # For nicer printing at the end

DEBUG_EXPLANATION = False

##############################
# Helper functions
##############################
def load_model_weights(model: torch.nn.Module, weight_path: str) -> bool:
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info(f"Loaded weights from {weight_path}")
        return True
    else:
        logger.info(f"No weight file found at {weight_path}. Will train from scratch.")
        return False

def get_prediction(model, sample):
    if isinstance(sample, tuple):
        sample = np.array(sample)
    from sklearn.tree import DecisionTreeClassifier
    if isinstance(model, DecisionTreeClassifier):
        return model.predict_proba(sample.reshape(1, -1))[0, 1]
    elif isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'conv1'):
                x_tensor = torch.FloatTensor(sample).reshape(1, 1, 3, 3)
            else:
                x_tensor = torch.FloatTensor(sample).reshape(1, 3, 3)
            output = model(x_tensor)
            return torch.softmax(output, dim=1)[0, 1].item()
    else:
        return 0.0

def aggregate_explanation_stats(model, samples, explainer):
    """
    Enhanced function to collect detailed attribution statistics
    """
    true_max_vals = []
    true_min_vals = []
    false_max_vals = []
    all_attribution_values = []
    
    for sample in samples:
        p = get_prediction(model, sample)
        explanation = explainer.explain(sample)
        if 'coefficients' in explanation:
            values = np.array(explanation['coefficients'])
        elif 'shap_values' in explanation:
            values = np.array(explanation['shap_values'])
        else:
            continue
            
        all_attribution_values.extend(values)
        
        if p >= 0.5:
            true_max_vals.append(np.max(values))
            true_min_vals.append(np.min(values))
        else:
            false_max_vals.append(np.max(values))
    
    stats = {}
    if true_max_vals:
        stats['true_max_mean'] = np.mean(true_max_vals)
        stats['true_min_mean'] = np.mean(true_min_vals)
    else:
        stats['true_max_mean'] = stats['true_min_mean'] = None
    
    if false_max_vals:
        stats['false_max_mean'] = np.mean(false_max_vals)
    else:
        stats['false_max_mean'] = None
    
    # Additional statistics
    if all_attribution_values:
        stats['attribution_magnitude'] = np.mean([abs(v) for v in all_attribution_values])
        stats['attribution_stability'] = np.std([abs(v) for v in all_attribution_values])
    
    return stats, all_attribution_values

def evaluate_functional_equivalence(reconstructed_dnf, original_function):
    """
    Evaluate functional equivalence by checking if reconstructed DNF produces
    same outputs as original function for all possible inputs.
    """
    terms = parse_dnf_to_terms(reconstructed_dnf)
    
    def eval_dnf(x, terms):
        for term in terms:
            if all(x[i] == 1 for i in term):
                return True
        return False
    
    all_inputs = list(product([0, 1], repeat=9))
    correct = 0
    total = len(all_inputs)
    
    for x in all_inputs:
        x_array = np.array(x)
        pred = eval_dnf(x_array, terms)
        actual = original_function(x_array)
        if pred == (actual > 0.5):
            correct += 1
    
    return correct / total

##############################
# Training helper: uses different parameters for normal vs overtrained mode.
##############################
def train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained):
    if overtrained:
        # For overfitting, combine training and validation data
        # This lets the model see and memorize ALL available data
        X_combined = torch.cat([X_train, X_val], dim=0)
        y_combined = torch.cat([y_train, y_val], dim=0)
        
        # Use the combined data for both training and "validation"
        if hasattr(model, 'conv1'):  # Assume CNN
            return train_model(model, (X_combined, y_combined), (X_combined, y_combined), 
                              epochs=1000, lr=0.001, weight_decay=0.0)
        else:
            return train_model(model, (X_combined, y_combined), (X_combined, y_combined), 
                              epochs=1000, lr=0.001, weight_decay=0.0)
    else:
        # Normal training schedule.
        if hasattr(model, 'conv1'):  # CNN
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=500, lr=0.001, weight_decay=0.01)
        else:
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001, weight_decay=0.01)

##################################################################
# Approach 1: Selective (Known DNF)
##################################################################
def selective_approach_test(boolean_func, func_name, overtrained):
    """
    Enhanced selective approach with more detailed metrics.
    Now explicitly appends "(Overfitted)" or "(Normal)" to keys.
    """
    logger.info("=== Approach 1: Selective inputs based on known DNF ===")
    target_dnf = get_function_str(boolean_func)
    logger.info(f"Testing function: {func_name}")
    logger.info(f"Target DNF: {target_dnf}")

    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load or train each model
    for name in ['FCN', 'CNN']:
        model = models[name]
        # FIX: Include training regime in weight file name
        regime_suffix = "overfitted" if overtrained else "normal"
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}_{regime_suffix}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

    # Parse the known terms
    known_terms = parse_dnf_to_terms(target_dnf)
    samples = [create_test_case_for_term(term) for term in known_terms]
    total_terms = len(known_terms)
    known_terms_set = set(known_terms)
    
    results = {}
    explainer_metrics = {}
    
    model_performance = {}  # Track model accuracy on the known terms
    
    # Overfitted or normal label:
    regime_label = "(Overfitted)" if overtrained else "(Normal)"
    
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        found_terms = set()
        
        # Test prediction accuracy on all known terms
        correct_predictions = 0
        for idx, term in enumerate(known_terms):
            threshold_val = 0.5
            prediction = get_prediction(model, create_test_case_for_term(term))
            if prediction >= threshold_val:
                found_terms.add(term)
                correct_predictions += 1
                logger.info(f"Found term {term} at sample {idx+1} / {total_terms}")
            else:
                logger.info(f"Term {term} did NOT trigger positive prediction (pred={prediction:.4f}, thr={threshold_val}).")
        
        # Model accuracy on the known terms
        model_performance[model_name] = {
            'term_accuracy': correct_predictions / total_terms if total_terms > 0 else 0.0,
            'correct_terms': correct_predictions,
            'total_terms': total_terms
        }
        
        # Construct the reconstructed DNF
        if not found_terms:
            reconstructed_dnf = "False"
        else:
            dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
            reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
        
        logger.info(f"Reconstructed DNF (from model {model_name}): {reconstructed_dnf}")
        
        # Evaluate reconstruction
        term_metrics = compute_term_metrics(found_terms, known_terms_set)
        functional_equiv = evaluate_functional_equivalence(reconstructed_dnf, boolean_func)
        
        # Determine which explainers to test
        explainer_list = ['LIME', 'KernelSHAP']
        if hasattr(model, 'parameters'):
            explainer_list.append('IntegratedGradients')
        if model_name == 'Decision Tree':
            explainer_list.append('TreeSHAP')
        
        # Evaluate each explainer
        for explainer_name in explainer_list:
            if explainer_name == 'LIME':
                explainer = LIMEExplainer(model)
            elif explainer_name == 'KernelSHAP':
                explainer = KernelSHAPExplainer(model)
            elif explainer_name == 'IntegratedGradients':
                explainer = IntegratedGradientsExplainer(model)
            elif explainer_name == 'TreeSHAP':
                explainer = TreeSHAPExplainer(model)
            
            # Aggregate explanation stats
            stats, all_attribution_values = aggregate_explanation_stats(model, samples, explainer)
            logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}: {stats}")
            
            # Check for exact match
            correct = are_dnfs_equivalent(reconstructed_dnf, target_dnf)
            exact_match_score = 1.0 if correct else 0.0
            
            # **Add the regime label** so the chart sees "(Overfitted)" or "(Normal)"
            key = f"{model_name}-{explainer_name} {regime_label}"
            results[key] = exact_match_score
            
            explainer_metrics[key] = {
                'term_precision': term_metrics['precision'],
                'term_recall': term_metrics['recall'],
                'term_f1': term_metrics['f1'],
                'functional_equivalence': functional_equiv,
                'exact_match': exact_match_score,
                'attribution_stats': stats,
                'model_performance': model_performance[model_name],
                'training_regime': 'overfitted' if overtrained else 'normal'
            }
            
            logger.info(f"\nExplainer: {explainer_name}")
            logger.info(f"Exact Match: {'✓' if correct else '✗'}")
            logger.info(f"Term Precision: {term_metrics['precision']:.4f}")
            logger.info(f"Term Recall: {term_metrics['recall']:.4f}")
            logger.info(f"Term F1: {term_metrics['f1']:.4f}")
            logger.info(f"Functional Equivalence: {functional_equiv:.4f}")
            
            # Save metrics
            artifact_file = os.path.join(
                "artifacts", 
                f"reconstruction_{func_name}_{model_name}_{explainer_name}_selective.txt"
            )
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Original terms: {known_terms_set}\n")
                f.write(f"Exact Match: {exact_match_score}\n")
                f.write(f"Term Precision: {term_metrics['precision']:.4f}\n")
                f.write(f"Term Recall: {term_metrics['recall']:.4f}\n")
                f.write(f"Term F1: {term_metrics['f1']:.4f}\n")
                f.write(f"Functional Equivalence: {functional_equiv:.4f}\n")
                f.write(f"Attribution Statistics: {stats}\n")
    
    logger.info("\nRanking of explainers (Selective):")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (key, score) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Exact Match)")
    
    return results, explainer_metrics


##################################################################
# Approach 2: Random Sampling (50 samples)
##################################################################
def approach2_test(boolean_func, func_name, num_samples, overtrained):
    """
    Enhanced random sampling approach with comprehensive metrics.
    Now explicitly appends "(Overfitted)" or "(Normal)" to keys.
    """
    logger.info(f"\n=== Approach 2: Reconstruction with {num_samples} random samples for {func_name} ===")
    
    target_dnf = get_function_str(boolean_func)
    known_terms_set = set(parse_dnf_to_terms(target_dnf))
    
    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load or train
    for name in ['FCN', 'CNN']:
        model = models[name]
        # FIX: Include training regime in weight file name
        regime_suffix = "overfitted" if overtrained else "normal"
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}_{regime_suffix}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

    expl_summary = {}
    results = {}
    explainer_metrics = {}
    
    all_inputs = list(product([0, 1], repeat=9))
    samples = [np.array(random.choice(all_inputs)) for _ in range(num_samples)]
    
    # Overfitted or normal label:
    regime_label = "(Overfitted)" if overtrained else "(Normal)"
    
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        
        # Select applicable explainers
        explainers = {
            'LIME': LIMEExplainer(model),
            'KernelSHAP': KernelSHAPExplainer(model)
        }
        if hasattr(model, 'parameters'):
            explainers['IntegratedGradients'] = IntegratedGradientsExplainer(model)
        if model_name == 'Decision Tree':
            explainers['TreeSHAP'] = TreeSHAPExplainer(model)
        
        for explainer_name, explainer in explainers.items():
            logger.info(f"\nUsing explainer: {explainer_name}")
            
            # Collect explanation statistics
            stats, all_attribution_values = aggregate_explanation_stats(model, samples, explainer)
            expl_summary.setdefault(model_name, {})[explainer_name] = stats
            logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}: {stats}")
            
            # Determine an adaptive threshold
            if stats.get('false_max_mean') is not None and stats['true_max_mean'] is not None:
                adaptive_threshold = (stats['true_max_mean'] + stats['false_max_mean']) / 2
            else:
                adaptive_threshold = 0.1
                
            logger.info(f"Adaptive threshold for {explainer_name} on {model_name}: {adaptive_threshold:.4f}")
            
            # Reconstruct DNF from random samples
            found_terms = set()
            all_predictions = []
            all_thresholds = []
            
            for idx, sample in enumerate(samples):
                # Explanation
                explanation = explainer.explain(sample)
                
                # Extract significant variables
                if 'coefficients' in explanation:
                    significant_vars = tuple(sorted(
                        i for i, coef in enumerate(explanation['coefficients'])
                        if coef > adaptive_threshold and sample[i] == 1
                    ))
                    all_thresholds.extend(
                        [coef for i, coef in enumerate(explanation['coefficients']) if sample[i] == 1]
                    )
                elif 'shap_values' in explanation:
                    significant_vars = tuple(sorted(
                        i for i, val in enumerate(explanation['shap_values'])
                        if val > adaptive_threshold and sample[i] == 1
                    ))
                    all_thresholds.extend(
                        [val for i, val in enumerate(explanation['shap_values']) if sample[i] == 1]
                    )
                else:
                    significant_vars = None
                
                # Get model prediction
                prediction = get_prediction(model, sample)
                all_predictions.append(prediction)
                
                # Add term if it triggers positive prediction and is minimal
                if prediction >= 0.5 and significant_vars and verify_term_is_minimal(significant_vars, list(found_terms), model):
                    found_terms.add(significant_vars)
                    logger.info(f"Found term {significant_vars} at sample {idx+1} / {num_samples}")
            
            # Construct final DNF
            if not found_terms:
                reconstructed_dnf = "False"
            else:
                dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
                reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
                
            logger.info(f"Reconstructed DNF: {reconstructed_dnf}")
            
            # Evaluate
            term_metrics = compute_term_metrics(found_terms, known_terms_set)
            functional_equiv = evaluate_functional_equivalence(reconstructed_dnf, boolean_func)
            exact_match = are_dnfs_equivalent(reconstructed_dnf, target_dnf)
            
            # Build the final key
            key = f"{model_name}-{explainer_name} {regime_label}"
            
            # Use functional equivalence as the main "score"
            results[key] = functional_equiv
            
            # Store detailed metrics
            explainer_metrics[key] = {
                'term_precision': term_metrics['precision'],
                'term_recall': term_metrics['recall'],
                'term_f1': term_metrics['f1'],
                'functional_equivalence': functional_equiv,
                'exact_match': 1.0 if exact_match else 0.0,
                'attribution_stats': stats,
                'training_regime': 'overfitted' if overtrained else 'normal',
                'threshold_stats': {
                    'mean': float(np.mean(all_thresholds)) if all_thresholds else None,
                    'std': float(np.std(all_thresholds)) if all_thresholds else None,
                    'min': float(np.min(all_thresholds)) if all_thresholds else None,
                    'max': float(np.max(all_thresholds)) if all_thresholds else None,
                    'used_threshold': float(adaptive_threshold)
                },
                'prediction_stats': {
                    'mean': float(np.mean(all_predictions)),
                    'std': float(np.std(all_predictions)),
                    'positive_ratio': float(sum(1 for p in all_predictions if p >= 0.5) / len(all_predictions))
                }
            }
            
            logger.info(f"Exact Match: {'✓' if exact_match else '✗'}")
            logger.info(f"Term Precision: {term_metrics['precision']:.4f}")
            logger.info(f"Term Recall: {term_metrics['recall']:.4f}")
            logger.info(f"Term F1: {term_metrics['f1']:.4f}")
            logger.info(f"Functional Equivalence: {functional_equiv:.4f}")
            
            # Save artifact
            artifact_file = os.path.join("artifacts", f"reconstruction_{func_name}_{model_name}_{explainer_name}_approach2.txt")
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Original terms: {known_terms_set}\n")
                f.write(f"Exact Match: {'Yes' if exact_match else 'No'}\n")
                f.write(f"Term Precision: {term_metrics['precision']:.4f}\n")
                f.write(f"Term Recall: {term_metrics['recall']:.4f}\n")
                f.write(f"Term F1: {term_metrics['f1']:.4f}\n")
                f.write(f"Functional Equivalence: {functional_equiv:.4f}\n")
                f.write(f"Sample Coverage: {num_samples}/512 ({(num_samples/512)*100:.1f}%)\n")
                f.write(f"Attribution Statistics: {stats}\n")
                f.write(f"Threshold Statistics: {explainer_metrics[key]['threshold_stats']}\n")
                f.write(f"Prediction Statistics: {explainer_metrics[key]['prediction_stats']}\n")
    
    logger.info("\nRanking of explainers (Random Sampling):")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (key, score) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Functional Equivalence)")
    
    return results, explainer_metrics


##################################################################
# Approach 3: Exhaustive (All 512 inputs)
##################################################################
def approach3_test(boolean_func, func_name, timeout_sec, overtrained):
    """
    Enhanced exhaustive approach with comprehensive metrics.
    Now explicitly appends "(Overfitted)" or "(Normal)" to keys.
    """
    logger.info(f"\n=== Approach 3: Full reconstruction over all inputs with timeout {timeout_sec}s for {func_name} ===")
    
    target_dnf = get_function_str(boolean_func)
    known_terms_set = set(parse_dnf_to_terms(target_dnf))
    
    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load or train
    for name in ['FCN', 'CNN']:
        model = models[name]
        # FIX: Include training regime in weight file name
        regime_suffix = "overfitted" if overtrained else "normal"
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}_{regime_suffix}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model
    
    results = {}
    explainer_metrics = {}
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    all_inputs = list(product([0, 1], repeat=9))
    total_inputs = len(all_inputs)
    
    # Overfitted or normal label:
    regime_label = "(Overfitted)" if overtrained else "(Normal)"
    
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        
        # Select applicable explainers
        explainers = {
            'LIME': LIMEExplainer(model),
            'KernelSHAP': KernelSHAPExplainer(model)
        }
        if hasattr(model, 'parameters'):
            explainers['IntegratedGradients'] = IntegratedGradientsExplainer(model)
        if model_name == 'Decision Tree':
            explainers['TreeSHAP'] = TreeSHAPExplainer(model)
        
        for explainer_name, explainer in explainers.items():
            logger.info(f"\nUsing explainer: {explainer_name}")
            
            # Process all inputs with timeout
            found_terms = set()
            processed = 0
            agg_stats = []
            all_attributions = []
            start_time = time.time()
            
            try:
                signal.alarm(timeout_sec)
                for sample in all_inputs:
                    processed += 1
                    if isinstance(sample, tuple):
                        sample = np.array(sample)
                    
                    explanation = explainer.explain(sample)
                    
                    # Extract values
                    if 'coefficients' in explanation:
                        values = np.array(explanation['coefficients'])
                        significant_vars = tuple(sorted(
                            i for i, coef in enumerate(values)
                            if coef > 0.1 and sample[i] == 1
                        ))
                    elif 'shap_values' in explanation:
                        values = np.array(explanation['shap_values'])
                        significant_vars = tuple(sorted(
                            i for i, val in enumerate(values)
                            if val > 0.1 and sample[i] == 1
                        ))
                    else:
                        values = None
                        significant_vars = None
                    
                    # Collect stats
                    if values is not None:
                        all_attributions.extend(values)
                        p = get_prediction(model, sample)
                        agg_stats.append({
                            'prediction': p,
                            'max': float(np.max(values)),
                            'min': float(np.min(values)),
                            'sum': float(np.sum(values)),
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values))
                        })
                    
                    # If minimal
                    if significant_vars and verify_term_is_minimal(significant_vars, list(found_terms), model):
                        found_terms.add(significant_vars)
                        logger.info(f"Found term {significant_vars} at processed {processed}/{total_inputs}")
                
                signal.alarm(0)
                completion_status = "Complete"
                
            except TimeoutException:
                logger.info(f"Timeout reached after {processed}/{total_inputs} inputs.")
                completion_status = f"Timeout after {timeout_sec}s"
            
            processing_time = time.time() - start_time
            
            # Construct the DNF
            if not found_terms:
                reconstructed_dnf = "False"
            else:
                dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
                reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
            
            logger.info(f"Reconstructed DNF: {reconstructed_dnf}")
            
            # Evaluate
            term_metrics = compute_term_metrics(found_terms, known_terms_set)
            functional_equiv = evaluate_functional_equivalence(reconstructed_dnf, boolean_func)
            exact_match = are_dnfs_equivalent(reconstructed_dnf, target_dnf)
            
            # Summarize explanation stats
            if agg_stats:
                true_stats = [d for d in agg_stats if d['prediction'] >= 0.5]
                false_stats = [d for d in agg_stats if d['prediction'] < 0.5]
                true_max_mean = float(np.mean([d['max'] for d in true_stats])) if true_stats else None
                true_min_mean = float(np.mean([d['min'] for d in true_stats])) if true_stats else None
                false_max_mean = float(np.mean([d['max'] for d in false_stats])) if false_stats else None
                
                logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}:")
                logger.info(f"  True cases: count={len(true_stats)}, mean(max)={true_max_mean}, mean(min)={true_min_mean}")
                logger.info(f"  False cases: count={len(false_stats)}, mean(max)={false_max_mean}")
                
                attribution_stats = {
                    'true_max_mean': true_max_mean,
                    'true_min_mean': true_min_mean,
                    'false_max_mean': false_max_mean,
                    'attribution_magnitude': float(np.mean(np.abs(all_attributions))) if all_attributions else None,
                    'attribution_stability': float(np.std(np.abs(all_attributions))) if all_attributions else None
                }
            else:
                attribution_stats = {}
            
            # Build final key
            key = f"{model_name}-{explainer_name} {regime_label}"
            
            # Store
            explainer_metrics[key] = {
                'term_precision': term_metrics['precision'],
                'term_recall': term_metrics['recall'],
                'term_f1': term_metrics['f1'],
                'functional_equivalence': functional_equiv,
                'exact_match': 1.0 if exact_match else 0.0,
                'attribution_stats': attribution_stats,
                'training_regime': 'overfitted' if overtrained else 'normal',
                'processing': {
                    'inputs_processed': processed,
                    'total_inputs': total_inputs,
                    'completion_percentage': float(processed / total_inputs * 100),
                    'status': completion_status,
                    'processing_time': float(processing_time)
                }
            }
            
            # Use functional_equiv as main "score"
            results[key] = (functional_equiv, processed, total_inputs)
            
            logger.info(f"Exact Match: {'✓' if exact_match else '✗'}")
            logger.info(f"Term Precision: {term_metrics['precision']:.4f}")
            logger.info(f"Term Recall: {term_metrics['recall']:.4f}")
            logger.info(f"Term F1: {term_metrics['f1']:.4f}")
            logger.info(f"Functional Equivalence: {functional_equiv:.4f}")
            logger.info(f"Processing Time: {processing_time:.2f}s ({processed}/{total_inputs} inputs)")
            
            # Save artifact
            artifact_file = os.path.join(
                "artifacts",
                f"reconstruction_{func_name}_{model_name}_{explainer_name}_approach3.txt"
            )
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Original terms: {known_terms_set}\n")
                f.write(f"Exact Match: {'Yes' if exact_match else 'No'}\n")
                f.write(f"Term Precision: {term_metrics['precision']:.4f}\n")
                f.write(f"Term Recall: {term_metrics['recall']:.4f}\n")
                f.write(f"Term F1: {term_metrics['f1']:.4f}\n")
                f.write(f"Functional Equivalence: {functional_equiv:.4f}\n")
                f.write(f"Processing: {processed}/{total_inputs} inputs ({processed/total_inputs*100:.1f}%)\n")
                f.write(f"Processing Time: {processing_time:.2f}s\n")
                f.write(f"Status: {completion_status}\n")
                f.write(f"Attribution Statistics: {attribution_stats}\n")
    
    logger.info("\nRanking of explainers (Exhaustive):")
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (key, (score, proc, total)) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Processed {proc}/{total} inputs, {proc/total*100:.1f}%)")
    
    return results, explainer_metrics

##############################
# Helper to Evaluate Reconstructed DNF
##############################
def evaluate_reconstructed_dnf(dnf_expr, actual_func):
    """
    Evaluate how well the reconstructed DNF matches the actual function
    by testing all possible inputs
    """
    terms = parse_dnf_to_terms(dnf_expr)
    
    def eval_dnf(x, terms):
        for term in terms:
            if all(x[i] == 1 for i in term):
                return True
        return False
    
    all_inputs = list(product([0, 1], repeat=9))
    correct = 0
    total = len(all_inputs)
    
    # Metrics for detailed analysis
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for x in all_inputs:
        pred = eval_dnf(x, terms)
        actual = actual_func(np.array(x))
        if pred == actual:
            correct += 1
            if pred:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if pred:
                false_positives += 1
            else:
                false_negatives += 1
    
    score = correct / total
    logger.info(f"Reconstruction accuracy: {score*100:.2f}% ({correct}/{total})")
    logger.info(f"True Positives: {true_positives}, True Negatives: {true_negatives}")
    logger.info(f"False Positives: {false_positives}, False Negatives: {false_negatives}")
    
    # Return not just the score but detailed metrics
    return score

##############################
# Main entry point
##############################
def main():
    parser = argparse.ArgumentParser(
        description="Run reconstruction tests for explainers with different approaches."
    )
    parser.add_argument("--approach", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which approach to run: 1 (selective), 2 (50 random samples), 3 (all inputs), or all.")
    args = parser.parse_args()
    
    functions = [
        (dnf_simple, "Simple_DNF"),
        (dnf_example, "Example_DNF"),
        (dnf_complex, "Complex_DNF")
    ]
    
    all_results = {}
    all_explainer_metrics = {}
    
    for overtrained in [False, True]:
        regime_suffix = "normal" if not overtrained else "overfitted"
        if args.approach in ["1", "all"]:
            for func, name in functions:
                res, metrics = selective_approach_test(func, name, overtrained)
                key = f"{name}_selective_{regime_suffix}"
                all_results[key] = res
                all_explainer_metrics[key] = metrics
        
        if args.approach in ["2", "all"]:
            for func, name in functions:
                res, metrics = approach2_test(func, name, num_samples=50, overtrained=overtrained)
                key = f"{name}_approach2_{regime_suffix}"
                all_results[key] = res
                all_explainer_metrics[key] = metrics
        
        if args.approach in ["3", "all"]:
            for func, name in functions:
                res, metrics = approach3_test(func, name, timeout_sec=30, overtrained=overtrained)
                key = f"{name}_approach3_{regime_suffix}"
                all_results[key] = res
                all_explainer_metrics[key] = metrics
    
    # Save all metrics to a comprehensive JSON file
    with open("artifacts/comprehensive_metrics.json", "w") as f:
        json.dump({
            "results": all_results,
            "explainer_metrics": all_explainer_metrics,
            "training_regime": "mixed"  # now contains both normal and overfitted entries
        }, f, indent=2, default=str)
    
    # Generate a comprehensive summary report
    logger.info("\n" + "="*50)
    logger.info("COMPREHENSIVE RESULTS SUMMARY")
    logger.info("="*50)
    
    # 1. Summary by metric
    logger.info("\n1. Performance by Metric:")
    for metric in ['exact_match', 'term_precision', 'term_recall', 'term_f1', 'functional_equivalence']:
        logger.info(f"\n  {metric.replace('_', ' ').title()}:")
        for approach in ["selective", "approach2", "approach3"]:
            logger.info(f"    {approach.capitalize()} Approach:")
            metrics_list = [metrics for k, metrics in all_explainer_metrics.items() if approach in k]
            
            normal_vals = []
            overfitted_vals = []
            
            for metrics_dict in metrics_list:
                for model_explainer, metrics in metrics_dict.items():
                    if metrics.get('training_regime') == 'normal':
                        if metric in metrics:
                            normal_vals.append(metrics[metric])
                    elif metrics.get('training_regime') == 'overfitted':
                        if metric in metrics:
                            overfitted_vals.append(metrics[metric])
            
            if normal_vals:
                logger.info(f"      Normal training: {np.mean(normal_vals):.4f} (± {np.std(normal_vals):.4f})")
            if overfitted_vals:
                logger.info(f"      Overfitted: {np.mean(overfitted_vals):.4f} (± {np.std(overfitted_vals):.4f})")
    
    # 2. Summary by model architecture
    logger.info("\n2. Performance by Model Architecture:")
    for model_type in ["FCN", "CNN", "Decision Tree"]:
        logger.info(f"\n  {model_type}:")
        for approach in ["selective", "approach2", "approach3"]:
            logger.info(f"    {approach.capitalize()} Approach:")
            
            model_metrics = {}
            for metrics_dict in [m for k, m in all_explainer_metrics.items() if approach in k]:
                for model_explainer, metrics in metrics_dict.items():
                    if model_type in model_explainer:
                        explainer = model_explainer.split('-')[1]
                        if explainer not in model_metrics:
                            model_metrics[explainer] = []
                        model_metrics[explainer].append(metrics.get('functional_equivalence', 0))
            
            for explainer, values in model_metrics.items():
                if values:
                    logger.info(f"      {explainer}: {np.mean(values):.4f} (± {np.std(values):.4f})")
    
    # 3. Summary by explainer
    logger.info("\n3. Performance by Explainer:")
    for explainer in ["LIME", "KernelSHAP", "TreeSHAP", "IntegratedGradients"]:
        logger.info(f"\n  {explainer}:")
        for approach in ["selective", "approach2", "approach3"]:
            logger.info(f"    {approach.capitalize()} Approach:")
            
            normal_vals = []
            overfitted_vals = []
            
            for metrics_dict in [m for k, m in all_explainer_metrics.items() if approach in k]:
                for model_explainer, metrics in metrics_dict.items():
                    if explainer in model_explainer:
                        if metrics.get('training_regime') == 'normal':
                            normal_vals.append(metrics.get('functional_equivalence', 0))
                        elif metrics.get('training_regime') == 'overfitted':
                            overfitted_vals.append(metrics.get('functional_equivalence', 0))
            
            if normal_vals:
                logger.info(f"      Normal training: {np.mean(normal_vals):.4f} (± {np.std(normal_vals):.4f})")
            if overfitted_vals:
                logger.info(f"      Overfitted: {np.mean(overfitted_vals):.4f} (± {np.std(overfitted_vals):.4f})")
    
    # 4. Summary by Boolean function complexity
    logger.info("\n4. Performance by Boolean Function Complexity:")
    for func_name in ["Simple_DNF", "Example_DNF", "Complex_DNF"]:
        logger.info(f"\n  {func_name}:")
        for approach in ["selective", "approach2", "approach3"]:
            logger.info(f"    {approach.capitalize()} Approach:")
            
            metrics_key = f"{func_name}_{approach}"
            if metrics_key in all_explainer_metrics:
                metrics_dict = all_explainer_metrics[metrics_key]
                
                normal_vals = []
                overfitted_vals = []
                
                for model_explainer, metrics in metrics_dict.items():
                    if metrics.get('training_regime') == 'normal':
                        normal_vals.append(metrics.get('functional_equivalence', 0))
                    elif metrics.get('training_regime') == 'overfitted':
                        overfitted_vals.append(metrics.get('functional_equivalence', 0))
                
                if normal_vals:
                    logger.info(f"      Normal training: {np.mean(normal_vals):.4f} (± {np.std(normal_vals):.4f})")
                if overfitted_vals:
                    logger.info(f"      Overfitted: {np.mean(overfitted_vals):.4f} (± {np.std(overfitted_vals):.4f})")
    
    # 5. Attribution statistics
    logger.info("\n5. Attribution Statistics by Explainer:")
    for explainer in ["LIME", "KernelSHAP", "TreeSHAP", "IntegratedGradients"]:
        logger.info(f"\n  {explainer}:")
        
        all_stats = {'normal': {}, 'overfitted': {}}
        
        for metrics_dict in [m for d in all_explainer_metrics.values() for m in d.values()]:
            if explainer in metrics_dict and 'attribution_stats' in metrics_dict:
                stats = metrics_dict['attribution_stats']
                regime = metrics_dict.get('training_regime', 'normal')
                
                for stat_name, stat_value in stats.items():
                    if stat_value is not None:
                        if stat_name not in all_stats[regime]:
                            all_stats[regime][stat_name] = []
                        all_stats[regime][stat_name].append(stat_value)
        
        for regime in ['normal', 'overfitted']:
            logger.info(f"    {regime.capitalize()} Training:")
            for stat_name, values in all_stats[regime].items():
                if values:
                    logger.info(f"      {stat_name.replace('_', ' ').title()}: {np.mean(values):.4f} (± {np.std(values):.4f})")
    
    logger.info("\nDetailed metrics saved to artifacts/comprehensive_metrics.json")

if __name__ == "__main__":
    main()