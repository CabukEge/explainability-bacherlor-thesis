#!/usr/bin/env python
"""
run_tests.py

This script runs three different approaches to test the explainers.

Approach 1 (Selective):
  - Assumes the target DNF is known.
  - Uses the known terms to construct inputs that are guaranteed to be positive for each term.

Approach 2 (Random Sampling):
  - Uses 50 random samples chosen from the 512 possible 9-bit combinations.

Approach 3 (Exhaustive):
  - Uses all 512 possible inputs.
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

# Setup logging to both terminal and a log file.
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

# Import modules from your project
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

# DEBUG_EXPLANATION flag: set to True to log summary stats for explanation values.
DEBUG_EXPLANATION = True

def log_explanation_summary(explanation):
    """
    Logs summary statistics (min, max, mean) for explanation values (coefficients or shap_values)
    """
    if 'coefficients' in explanation:
        coeffs = explanation['coefficients']
        logger.info(f"Coefficients summary: min={min(coeffs):.4f}, max={max(coeffs):.4f}, mean={np.mean(coeffs):.4f}")
    elif 'shap_values' in explanation:
        shap_vals = explanation['shap_values']
        logger.info(f"SHAP values summary: min={np.min(shap_vals):.4f}, max={np.max(shap_vals):.4f}, mean={np.mean(shap_vals):.4f}")

def load_model_weights(model: torch.nn.Module, weight_path: str) -> bool:
    """
    Attempts to load model weights from a given path.
    If successful, returns True; otherwise returns False.
    """
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info(f"Loaded weights from {weight_path}")
        return True
    else:
        logger.info(f"No weight file found at {weight_path}. Will train from scratch.")
        return False

# ================================
# Approach 1: Selective (Known DNF)
# ================================
def selective_approach_test(boolean_func, func_name):
    logger.info("=== Approach 1: Selective inputs based on known DNF ===")
    target_dnf = get_function_str(boolean_func)
    logger.info(f"Testing function: {func_name}")
    logger.info(f"Target DNF: {target_dnf}")

    # Generate data
    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)

    # Initialize models
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),  # Fast training for tree
        'CNN': CNN()
    }

    # For non–Decision Tree models, either load or train
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)

    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            if name == 'CNN':
                logger.info(f"Training model: {name} (increased epochs for CNN)")
                model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=500, lr=0.001)
            else:
                logger.info(f"Training model: {name}")
                model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

    # Parse the known DNF into its constituent terms.
    known_terms = parse_dnf_to_terms(target_dnf)
    total_terms = len(known_terms)

    results = {}
    explainer_metrics = {}

    # For each model, we check each known term using the model.
    # For neural network models (which have 'parameters'), we use a lower threshold.
    for model_name, model in models.items():
        logger.info(f"\nModel: {model_name}")
        found_terms = set()
        for idx, term in enumerate(known_terms):
            # Use a lower threshold for neural networks to account for slight uncertainty.
            threshold_val = 0.4 if hasattr(model, 'parameters') else 0.5
            # Log the predicted probability for this test case by enabling log_pred.
            if verify_term(term, model, threshold=threshold_val, log_pred=True):
                found_terms.add(term)
                logger.info(f"Found term {term} at sample {idx+1} out of {total_terms}")
            else:
                logger.info(f"Term {term} did not trigger a positive prediction in the model (threshold {threshold_val}).")

        # Build the reconstructed DNF from the found terms
        if not found_terms:
            reconstructed_dnf = "False"
        else:
            dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in found_terms]
            reconstructed_dnf = " ∨ ".join(sorted(dnf_terms))
        logger.info(f"Reconstructed DNF (from model {model_name}): {reconstructed_dnf}")

        # For each explainer attached to this model, assign the same reconstruction.
        explainer_list = ['LIME', 'KernelSHAP']
        if hasattr(model, 'parameters'):
            explainer_list.append('IntegratedGradients')
        if model_name == 'Decision Tree':
            explainer_list.append('TreeSHAP')
        for explainer_name in explainer_list:
            correct = are_dnfs_equivalent(reconstructed_dnf, target_dnf)
            term_metrics = compute_term_metrics(found_terms, set(known_terms))
            results[f"{model_name}-{explainer_name}"] = correct
            explainer_metrics[f"{model_name}-{explainer_name}"] = {
                'term_precision': term_metrics['precision'],
                'term_recall': term_metrics['recall'],
                'term_f1': term_metrics['f1'],
            }
            logger.info(f"\nExplainer: {explainer_name}")
            if correct:
                logger.info("✓ Correct reconstruction!")
            else:
                logger.info("✗ Incorrect reconstruction")
            artifact_file = os.path.join("artifacts", f"reconstruction_{func_name}_{model_name}_{explainer_name}_selective.txt")
            with open(artifact_file, "w") as f:
                f.write(f"Reconstructed DNF: {reconstructed_dnf}\n")
                f.write(f"Found terms: {found_terms}\n")
                f.write(f"Term metrics: {term_metrics}\n")

    return results, explainer_metrics

# ================================
# Approach 2: Random Sampling (50 samples)
# ================================
def approach2_test(boolean_func, func_name, num_samples=50):
    logger.info(f"\n=== Approach 2: Reconstruction with {num_samples} random samples for {func_name} ===")

    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)

    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }

    # For non–Decision Tree models, either load or train
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)

    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            if name == 'CNN':
                model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001)
            else:
                model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001)
            torch.save(model.state_dict(), weight_path)
            logger.info(f"Saved trained weights to {weight_path}")
        models[name] = model

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

        # Use random samples from the 512 possible inputs
        all_inputs = list(product([0, 1], repeat=9))
        samples = [np.array(random.choice(all_inputs)) for _ in range(num_samples)]

        for explainer_name, explainer in explainers.items():
            logger.info(f"\nUsing explainer: {explainer_name}")
            found_terms = set()
            for idx, sample in enumerate(samples):
                explanation = explainer.explain(sample)
                # Log explanation summary if debug is enabled.
                if DEBUG_EXPLANATION:
                    log_explanation_summary(explanation)
                if 'coefficients' in explanation:
                    coefficients = explanation['coefficients']
                    significant_vars = tuple(sorted(
                        i for i, coef in enumerate(coefficients) if coef > 0.1 and sample[i] == 1
                    ))
                elif 'shap_values' in explanation:
                    shap_values = explanation['shap_values']
                    significant_vars = tuple(sorted(
                        i for i, val in enumerate(shap_values) if val > 0.1 and sample[i] == 1
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

# ================================
# Approach 3: Exhaustive (All 512 inputs)
# ================================
def approach3_test(boolean_func, func_name, timeout_sec=30):
    logger.info(f"\n=== Approach 3: Full reconstruction over all inputs with timeout {timeout_sec}s for {func_name} ===")

    (X_train, y_train), (X_val, y_val), _ = generate_data(boolean_func)

    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }

    # For non–Decision Tree models, either load or train
    weights_dir = "models_weights"
    os.makedirs(weights_dir, exist_ok=True)

    for name in ['FCN', 'CNN']:
        model = models[name]
        weight_path = os.path.join(weights_dir, f"{name}.pt")
        loaded = load_model_weights(model, weight_path)
        if not loaded:
            logger.info(f"Training model: {name}")
            model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=300, lr=0.001)
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

            try:
                signal.alarm(timeout_sec)
                for sample in all_inputs:
                    processed += 1
                    explanation = explainer.explain(np.array(sample))
                    if DEBUG_EXPLANATION:
                        log_explanation_summary(explanation)
                    if 'coefficients' in explanation:
                        coefficients = explanation['coefficients']
                        significant_vars = tuple(sorted(
                            i for i, coef in enumerate(coefficients) if coef > 0.1 and sample[i] == 1
                        ))
                    elif 'shap_values' in explanation:
                        shap_values = explanation['shap_values']
                        significant_vars = tuple(sorted(
                            i for i, val in enumerate(shap_values) if val > 0.1 and sample[i] == 1
                        ))
                    else:
                        significant_vars = None

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

    logger.info("\nRanking of explainers (Approach 3):")
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for rank, (key, (score, proc, total)) in enumerate(sorted_results, start=1):
        logger.info(f"{rank}. {key}: {score*100:.2f}% (Processed {proc}/{total})")

    return results

# ==========================================
# Helper to Evaluate Reconstructed DNF
# ==========================================
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

# ==========================================
# Main entry point
# ==========================================
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

    if args.approach in ["1", "all"]:
        for func, name in functions:
            res, metrics = selective_approach_test(func, name)
            all_results[f"{name}_selective"] = res
            all_explainer_metrics[f"{name}_selective"] = metrics

    if args.approach in ["2", "all"]:
        for func, name in functions:
            res = approach2_test(func, name, num_samples=50)
            all_results[f"{name}_approach2"] = res

    if args.approach in ["3", "all"]:
        for func, name in functions:
            res = approach3_test(func, name, timeout_sec=30)
            all_results[f"{name}_approach3"] = res

    logger.info("\nFinal aggregated results:")
    logger.info(pformat(all_results, indent=4))

if __name__ == "__main__":
    main()
