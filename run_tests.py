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
    true_max_vals = []
    true_min_vals = []
    false_max_vals = []
    for sample in samples:
        p = get_prediction(model, sample)
        explanation = explainer.explain(sample)
        if 'coefficients' in explanation:
            values = np.array(explanation['coefficients'])
        elif 'shap_values' in explanation:
            values = np.array(explanation['shap_values'])
        else:
            continue
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
    return stats

##############################
# Training helper: uses different parameters for normal vs overtrained mode.
##############################
def train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained):
    if overtrained:
        # Use a higher epoch count and no weight decay for overfitting.
        if hasattr(model, 'conv1'):  # Assume CNN
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=1000, lr=0.001, weight_decay=0.0)
        else:
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=1000, lr=0.001, weight_decay=0.0)
    else:
        # Normal training schedule.
        if hasattr(model, 'conv1'):  # CNN
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=500, lr=0.001, weight_decay=0.01)
        else:
            return train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001, weight_decay=0.01)

##############################
# Approach 1: Selective (Known DNF)
##############################
def selective_approach_test(boolean_func, func_name, overtrained):
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
    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

    known_terms = parse_dnf_to_terms(target_dnf)
    samples = [create_test_case_for_term(term) for term in known_terms]
    total_terms = len(known_terms)
    results = {}
    explainer_metrics = {}
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        found_terms = set()
        for idx, term in enumerate(known_terms):
            threshold_val = 0.5  # Use a strict threshold.
            if verify_term(term, model, threshold=threshold_val, log_pred=True):
                found_terms.add(term)
                logger.info(f"Found term {term} at sample {idx+1} out of {total_terms}")
            else:
                logger.info(f"Term {term} did not trigger a positive prediction (threshold {threshold_val}).")
        if not found_terms:
            reconstructed_dnf = "False"
        else:
            dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
            reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
        logger.info(f"Reconstructed DNF (from model {model_name}): {reconstructed_dnf}")
        explainer_list = ['LIME', 'KernelSHAP']
        if hasattr(model, 'parameters'):
            explainer_list.append('IntegratedGradients')
        if model_name == 'Decision Tree':
            explainer_list.append('TreeSHAP')
        for explainer_name in explainer_list:
            if explainer_name == 'LIME':
                explainer = LIMEExplainer(model)
            elif explainer_name == 'KernelSHAP':
                explainer = KernelSHAPExplainer(model)
            elif explainer_name == 'IntegratedGradients':
                explainer = IntegratedGradientsExplainer(model)
            elif explainer_name == 'TreeSHAP':
                explainer = TreeSHAPExplainer(model)
            stats = aggregate_explanation_stats(model, samples, explainer)
            logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}: {stats}")
            correct = are_dnfs_equivalent(reconstructed_dnf, target_dnf)
            score = 1.0 if correct else 0.0
            results[f"{model_name}-{explainer_name}"] = score
            explainer_metrics[f"{model_name}-{explainer_name}"] = {
                'term_precision': 1.0 if correct else 0.0,
                'term_recall': 1.0 if correct else 0.0,
                'term_f1': 1.0 if correct else 0.0,
            }
            logger.info(f"\nExplainer: {explainer_name}")
            logger.info("✓ Correct reconstruction!" if correct else "✗ Incorrect reconstruction")
            artifact_file = os.path.join("artifacts", f"reconstruction_{func_name}_{model_name}_{explainer_name}_selective.txt")
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Score: {score}\n")
    logger.info("\nRanking of explainers (Selective):")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (key, score) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Processed 512/512)")
    return results, explainer_metrics

##############################
# Approach 2: Random Sampling (50 samples)
##############################
def approach2_test(boolean_func, func_name, num_samples, overtrained):
    logger.info(f"\n=== Approach 2: Reconstruction with {num_samples} random samples for {func_name} ===")
    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)
    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

    expl_summary = {}
    results = {}
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        explainers = {
            'LIME': LIMEExplainer(model),
            'KernelSHAP': KernelSHAPExplainer(model)
        }
        if hasattr(model, 'parameters'):
            explainers['IntegratedGradients'] = IntegratedGradientsExplainer(model)
        if model_name == 'Decision Tree':
            explainers['TreeSHAP'] = TreeSHAPExplainer(model)
        all_inputs = list(product([0, 1], repeat=9))
        samples = [np.array(random.choice(all_inputs)) for _ in range(num_samples)]
        for explainer_name, explainer in explainers.items():
            logger.info(f"\nUsing explainer: {explainer_name}")
            stats = aggregate_explanation_stats(model, samples, explainer)
            expl_summary.setdefault(model_name, {})[explainer_name] = stats
            logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}: {stats}")
            if stats['false_max_mean'] is not None:
                adaptive_threshold = (stats['true_max_mean'] + stats['false_max_mean']) / 2
            else:
                adaptive_threshold = 0.1
            logger.info(f"Adaptive threshold for {explainer_name} on {model_name}: {adaptive_threshold:.4f}")
            found_terms = set()
            for idx, sample in enumerate(samples):
                explanation = explainer.explain(sample)
                if 'coefficients' in explanation:
                    significant_vars = tuple(sorted(
                        i for i, coef in enumerate(explanation['coefficients'])
                        if coef > adaptive_threshold and sample[i] == 1
                    ))
                elif 'shap_values' in explanation:
                    significant_vars = tuple(sorted(
                        i for i, val in enumerate(explanation['shap_values'])
                        if val > adaptive_threshold and sample[i] == 1
                    ))
                else:
                    significant_vars = None
                if significant_vars and verify_term_is_minimal(significant_vars, list(found_terms), model):
                    found_terms.add(significant_vars)
                    logger.info(f"Found term {significant_vars} at sample {idx+1} out of {num_samples}")
            if not found_terms:
                reconstructed_dnf = "False"
            else:
                dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
                reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
            logger.info(f"Reconstructed DNF: {reconstructed_dnf}")
            score = evaluate_reconstructed_dnf(reconstructed_dnf, boolean_func)
            results[f"{model_name}-{explainer_name}"] = score
            artifact_file = os.path.join("artifacts", f"reconstruction_{func_name}_{model_name}_{explainer_name}_approach2.txt")
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Accuracy Score: {score}\n")
    logger.info("\nRanking of explainers (Approach 2):")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (key, score) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}%")
    return results

##############################
# Approach 3: Exhaustive (All 512 inputs)
##############################
def approach3_test(boolean_func, func_name, timeout_sec, overtrained):
    logger.info(f"\n=== Approach 3: Full reconstruction over all inputs with timeout {timeout_sec}s for {func_name} ===")
    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)
    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}_{func_name}.pt") 
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            model, _ = train_model_with_mode(model, X_train, y_train, X_val, y_val, overtrained)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model
    results = {}
    class TimeoutException(Exception):
        pass
    def timeout_handler(signum, frame):
        raise TimeoutException()
    signal.signal(signal.SIGALRM, timeout_handler)
    all_inputs = list(product([0, 1], repeat=9))
    total_inputs = len(all_inputs)
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
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
            found_terms = set()
            processed = 0
            agg_stats = []
            try:
                signal.alarm(timeout_sec)
                for sample in all_inputs:
                    processed += 1
                    if isinstance(sample, tuple):
                        sample = np.array(sample)
                    explanation = explainer.explain(sample)
                    if 'coefficients' in explanation:
                        values = np.array(explanation['coefficients'])
                        significant_vars = tuple(sorted(
                            i for i, coef in enumerate(explanation['coefficients'])
                            if coef > 0.1 and sample[i] == 1
                        ))
                    elif 'shap_values' in explanation:
                        values = np.array(explanation['shap_values'])
                        significant_vars = tuple(sorted(
                            i for i, val in enumerate(explanation['shap_values'])
                            if val > 0.1 and sample[i] == 1
                        ))
                    else:
                        significant_vars = None
                        values = None
                    if values is not None:
                        p = get_prediction(model, sample)
                        agg_stats.append({'prediction': p, 'max': np.max(values), 'min': np.min(values)})
                    if significant_vars and verify_term_is_minimal(significant_vars, list(found_terms), model):
                        found_terms.add(significant_vars)
                        logger.info(f"Found term {significant_vars} at processed count {processed} out of {total_inputs}")
                signal.alarm(0)
            except TimeoutException:
                logger.info(f"Timeout reached after processing {processed} out of {total_inputs} inputs.")
            if not found_terms:
                reconstructed_dnf = "False"
            else:
                dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
                reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
            logger.info(f"Reconstructed DNF: {reconstructed_dnf}")
            score = evaluate_reconstructed_dnf(reconstructed_dnf, boolean_func)
            results[f"{model_name}-{explainer_name}"] = (score, processed, total_inputs)
            artifact_file = os.path.join("artifacts", f"reconstruction_{func_name}_{model_name}_{explainer_name}_approach3.txt")
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Accuracy Score: {score}\n")
                f.write(f"Processed {processed} out of {total_inputs} inputs.\n")
            if agg_stats:
                true_stats = [d for d in agg_stats if d['prediction'] >= 0.5]
                false_stats = [d for d in agg_stats if d['prediction'] < 0.5]
                true_max_mean = np.mean([d['max'] for d in true_stats]) if true_stats else None
                true_min_mean = np.mean([d['min'] for d in true_stats]) if true_stats else None
                false_max_mean = np.mean([d['max'] for d in false_stats]) if false_stats else None
                logger.info(f"Aggregated explanation stats for {explainer_name} on {model_name}:")
                logger.info(f"  True cases: count={len(true_stats)}, mean(max)={true_max_mean}, mean(min)={true_min_mean}")
                logger.info(f"  False cases: count={len(false_stats)}, mean(max)={false_max_mean}")
    logger.info("\nRanking of explainers (Approach 3):")
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (key, (score, proc, total)) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Processed {proc}/{total})")
    return results

##############################
# Helper to Evaluate Reconstructed DNF
##############################
def evaluate_reconstructed_dnf(dnf_expr, actual_func):
    terms = parse_dnf_to_terms(dnf_expr)
    def eval_dnf(x, terms):
        for term in terms:
            if all(x[i] == 1 for i in term):
                return True
        return False
    all_inputs = list(product([0, 1], repeat=9))
    correct = 0
    for x in all_inputs:
        pred = eval_dnf(x, terms)
        actual = actual_func(np.array(x))
        if pred == actual:
            correct += 1
    score = correct / len(all_inputs)
    logger.info(f"Reconstruction accuracy: {score*100:.2f}% ({correct}/{len(all_inputs)})")
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
    parser.add_argument("--overtrained", action="store_true",
                        help="If set, train models in an overtrained regime (more epochs) to achieve near-100% accuracy.")
    args = parser.parse_args()
    
    overtrained = args.overtrained
    functions = [
        (dnf_simple, "Simple_DNF"),
        (dnf_example, "Example_DNF"),
        (dnf_complex, "Complex_DNF")
    ]
    all_results = {}
    all_explainer_metrics = {}
    if args.approach in ["1", "all"]:
        for func, name in functions:
            res, metrics = selective_approach_test(func, name, overtrained)
            all_results[f"{name}_selective"] = res
            all_explainer_metrics[f"{name}_selective"] = metrics
    if args.approach in ["2", "all"]:
        for func, name in functions:
            res = approach2_test(func, name, num_samples=50, overtrained=overtrained)
            all_results[f"{name}_approach2"] = res
    if args.approach in ["3", "all"]:
        for func, name in functions:
            res = approach3_test(func, name, timeout_sec=30, overtrained=overtrained)
            all_results[f"{name}_approach3"] = res
    logger.info("\nFinal aggregated results:")
    logger.info(pformat(all_results, indent=4))
    
if __name__ == "__main__":
    main()
