# evaluate.py

import torch
import numpy as np
from data import generate_data
from models import FCN, CNN, train_model, train_tree
from explainers import (
    LIMEExplainer, 
    TreeSHAPExplainer,
    KernelSHAPExplainer,
    IntegratedGradientsExplainer  # <-- add this
)
import scipy.special
from boolean_functions import dnf_example, dnf_simple, dnf_complex
from itertools import combinations, product
from sklearn.tree import DecisionTreeClassifier
from sympy.logic.boolalg import Or, And, Not
from sympy import symbols
from typing import List, Dict, Any, Set, Tuple
import time

def create_test_case_for_term(term: tuple, n: int = 9) -> np.ndarray:
    """Create a test case where specified variables are 1, others are 0"""
    x = np.zeros(n)
    x[list(term)] = 1
    return x

def verify_term(term: tuple, model, threshold: float = 0.5, log_pred: bool = False) -> bool:
    """
    Verify if a term is valid by checking model prediction.
    If log_pred is True, print the predicted probability for the test case.
    """
    test_case = create_test_case_for_term(term)
    
    if isinstance(model, DecisionTreeClassifier):
        pred = model.predict_proba(test_case.reshape(1, -1))[0, 1]
        if log_pred:
            print(f"Term {term} predicted probability: {pred:.4f}")
        return pred > threshold
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            if isinstance(model, CNN):
                x_tensor = torch.FloatTensor(test_case).reshape(1, 1, 3, 3)
            else:
                x_tensor = torch.FloatTensor(test_case).reshape(1, 3, 3)
            output = model(x_tensor)
            pred = torch.softmax(output, dim=1)[0, 1].item()
            if log_pred:
                print(f"Term {term} predicted probability: {pred:.4f}")
            return pred > threshold
    
    return False

def verify_term_is_minimal(term: tuple, known_terms: list, model) -> bool:
    """Verify if a term is minimal considering both local and global minimality"""
    if not verify_term(term, model):
        return False
    
    for i in range(1, len(term)):
        for subset in combinations(term, i):
            if verify_term(subset, model):
                return False

    if is_covered_by_terms(term, known_terms, model):
        return False
    
    return True

def is_covered_by_terms(term: tuple, known_terms: list, model) -> bool:
    """Check if a term can be represented by a combination of known terms"""
    if not known_terms:
        return False

    for r in range(1, len(known_terms) + 1):
        for terms_combo in combinations(known_terms, r):
            test_cases = []
            for inputs in product([0, 1], repeat=len(terms_combo)):
                case = np.zeros(9, dtype=int)
                for term_idx, use_term in enumerate(inputs):
                    if use_term:
                        for var in terms_combo[term_idx]:
                            case[var] = 1
                test_cases.append(case)

            term_result = verify_term(term, model)
            combo_results = [verify_term(tuple(np.where(case == 1)[0]), model) for case in test_cases]
            
            if all(result == term_result for result in combo_results):
                return True
    return False

def compute_term_metrics(reconstructed_terms, actual_terms):
    """Compute precision, recall, and F1 score for term reconstruction."""
    true_positives = len(set(reconstructed_terms).intersection(set(actual_terms)))
    false_positives = len(set(reconstructed_terms) - set(actual_terms))
    false_negatives = len(set(actual_terms) - set(reconstructed_terms))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def parse_dnf_to_terms(dnf_str: str) -> list:
    """Convert DNF string to list of terms"""
    if dnf_str == "False":
        return []
    terms = []
    for term_str in dnf_str.replace("(", "").replace(")", "").split(" ∨ "):
        term = [int(var.split("_")[1]) - 1 for var in term_str.strip().split(" ∧ ")]
        terms.append(tuple(sorted(term)))
    return sorted(terms)

def are_dnfs_equivalent(dnf1: str, dnf2: str) -> bool:
    """Check if two DNF expressions are logically equivalent"""
    if dnf1 == dnf2:
        return True
    terms1 = parse_dnf_to_terms(dnf1)
    terms2 = parse_dnf_to_terms(dnf2)
    return terms1 == terms2

def get_function_str(func) -> str:
    """Get the string representation of the target function"""
    if func == dnf_example:
        return "(x_1 ∧ x_2) ∨ (x_4 ∧ x_5 ∧ x_6) ∨ (x_7 ∧ x_8 ∧ x_9)"
    elif func == dnf_simple:
        return "(x_1 ∧ x_2) ∨ (x_5 ∧ x_6)"
    elif func == dnf_complex:
        return "(x_1 ∧ x_2 ∧ x_3) ∨ (x_4 ∧ x_5) ∨ (x_7 ∧ x_8 ∧ x_9)"
    return "Unknown function"

def reconstruct_dnf_with_explainer(model: Any, explainer: Any, known_dnf_terms: List, debug: bool = False) -> str:
    """
    Reconstruct DNF using a specific explainer.
    We now also handle the 'shap_values' from IntegratedGradientsExplainer.
    If debug is True, print summary statistics for explanation values (only for the first test case).
    """
    terms = set()
    
    # Create test cases from known terms
    test_cases = []
    for term in known_dnf_terms:
        test_case = np.zeros(9)
        test_case[list(term)] = 1
        test_cases.append(test_case)

    print(f"\nAnalyzing terms with {explainer.__class__.__name__}:")
    printed_summary = False  # Only print summary once
    for test_case in test_cases:
        explanation = explainer.explain(test_case)

        # If debug is enabled, print summary statistics for explanation values only once.
        if debug and not printed_summary:
            if 'coefficients' in explanation:
                coeffs = explanation['coefficients']
                print(f"Coefficients summary: min={min(coeffs):.4f}, max={max(coeffs):.4f}, mean={np.mean(coeffs):.4f}")
            elif 'shap_values' in explanation:
                shap_vals = explanation['shap_values']
                print(f"SHAP values summary: min={np.min(shap_vals):.4f}, max={np.max(shap_vals):.4f}, mean={np.mean(shap_vals):.4f}")
            printed_summary = True
        
        # Check if explanation has 'coefficients' (LIME) or 'shap_values' (KernelSHAP/IntegratedGradients)
        if hasattr(explainer, 'num_samples'):
            if 'coefficients' in explanation:  # LIME
                coefficients = explanation['coefficients']
                significant_vars = tuple(sorted(
                    i for i, coef in enumerate(coefficients)
                    if coef > 0.1 and test_case[i] == 1
                ))
            else:
                shap_values = explanation['shap_values']
                significant_vars = tuple(sorted(
                    i for i, val in enumerate(shap_values)
                    if val > 0.1 and test_case[i] == 1
                ))
        elif hasattr(explainer, 'model') and isinstance(explainer, TreeSHAPExplainer):
            shap_values = explanation['shap_values']
            significant_vars = tuple(sorted(
                i for i, val in enumerate(shap_values)
                if val > 0.1 and test_case[i] == 1
            ))
        else:
            shap_values = explanation['shap_values']
            significant_vars = tuple(sorted(
                i for i, val in enumerate(shap_values)
                if val > 0.1 and test_case[i] == 1
            ))

        if significant_vars and verify_term_is_minimal(significant_vars, list(terms), model):
            terms.add(significant_vars)
            print(f"Found term: {significant_vars}")

    if not terms:
        return "False"
        
    dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in term])})" for term in terms]
    return " ∨ ".join(sorted(dnf_terms))
    
def evaluate(boolean_func, func_name=""):
    """
    Train models and evaluate explanations.
    """
    print(f"\nTesting function: {func_name}")
    target_dnf = get_function_str(boolean_func)
    print(f"Target DNF: {target_dnf}")

    # Generate data
    start_time = time.time()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_data(boolean_func)
    data_gen_time = time.time() - start_time

    # Initialize models
    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train),
        'CNN': CNN()
    }

    # Train or skip
    training_metrics = {}
    for name, model in models.items():
        if name != 'Decision Tree':
            start_time = time.time()
            model, val_accuracies = train_model(
                models[name],
                (X_train, y_train),
                (X_val, y_val),
                epochs=1000,
                lr=0.001  # smaller LR for CNN
            )
            training_time = time.time() - start_time
            training_metrics[name] = {
                'training_time': training_time,
                'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
                'convergence_epoch': len(val_accuracies) if val_accuracies else 0
            }
            models[name] = model
        else:
            tree_pred = model.predict(X_val.reshape(-1, 9))
            val_acc = np.mean(tree_pred == y_val.numpy())
            training_metrics[name] = {
                'training_time': 0,
                'final_val_accuracy': val_acc,
                'convergence_epoch': 1
            }

    known_dnf_terms = set(parse_dnf_to_terms(target_dnf))
    
    results = {}
    explainer_metrics = {}
    
    for name, model in models.items():
    print(f"\nModel: {name}")
    
    # Determine training regime label based on final validation accuracy.
    # (Here we assume a final validation accuracy of 1.0 indicates overfitting.)
    regime_label = "(Overfitted)" if training_metrics[name]['final_val_accuracy'] == 1.0 else "(Normal)"
    
    # 1) Build the dictionary for this model
    explainers = {
        'LIME': LIMEExplainer(model),
        'KernelSHAP': KernelSHAPExplainer(model)
    }

    # 2) Only add IntegratedGradients if this model is a PyTorch model
    if isinstance(model, torch.nn.Module):
        explainers['IntegratedGradients'] = IntegratedGradientsExplainer(model)

    # 3) Only add TreeSHAP if this is a DecisionTreeClassifier
    if isinstance(model, DecisionTreeClassifier):
        explainers['TreeSHAP'] = TreeSHAPExplainer(model)

    # 4) Now actually run the reconstruction for each explainer
    for explainer_name, explainer in explainers.items():
        print(f"\nExplainer: {explainer_name}")
        start_time = time.time()

        reconstructed_dnf = reconstruct_dnf_with_explainer(
            model, explainer, list(known_dnf_terms), debug=DEBUG
        )

        explanation_time = time.time() - start_time
        print(f"Reconstructed DNF: {reconstructed_dnf}")

        # Evaluate the reconstruction
        reconstructed_terms = set(parse_dnf_to_terms(reconstructed_dnf))
        term_metrics = compute_term_metrics(reconstructed_terms, known_dnf_terms)
        correct = are_dnfs_equivalent(reconstructed_dnf, target_dnf)

        # Append the training regime label to the key
        results[f"{name}-{explainer_name} {regime_label}"] = correct
        explainer_metrics[f"{name}-{explainer_name} {regime_label}"] = {
            'explanation_time': explanation_time,
            'term_precision': term_metrics['precision'],
            'term_recall': term_metrics['recall'],
            'term_f1': term_metrics['f1'],
        }

        print("✓ Correct reconstruction!" if correct else "✗ Incorrect reconstruction")

    
    return results, training_metrics, explainer_metrics

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
    for x in all_inputs:
        x_array = np.array(x)
        pred = eval_dnf(x_array, terms)
        actual = original_function(x_array)
        if pred == (actual > 0.5):
            correct += 1
    
    return correct / len(all_inputs)
    
def main():
    functions = [
        (dnf_simple, "Simple DNF"),
        (dnf_example, "Example DNF"),
        (dnf_complex, "Complex DNF")
    ]

    all_results = {}
    all_training_metrics = {}
    all_explainer_metrics = {}
    
    for func, name in functions:
        results, training_metrics, explainer_metrics = evaluate(func, name)
        all_results[name] = results
        all_training_metrics[name] = training_metrics
        all_explainer_metrics[name] = explainer_metrics

    print("\nFinal Results:")
    for func_name, results in all_results.items():
        print(f"\n{func_name}:")
        for model_explainer, correct in results.items():
            print(f"{model_explainer}: {'✓' if correct else '✗'}")

    total_tests = 0
    successful_tests = 0
    for results in all_results.values():
        total_tests += len(results)
        successful_tests += sum(1 for correct in results.values() if correct)

    print("\nOverall Statistics:")
    print(f"Total correct reconstructions: {successful_tests}/{total_tests}")
    print(f"Overall accuracy: {(successful_tests/total_tests)*100:.1f}%")

    print("\nModel Training Performance:")
    for func_name, metrics in all_training_metrics.items():
        print(f"\n{func_name}:")
        for model, stats in metrics.items():
            print(f"{model}:")
            print(f"  Training time: {stats['training_time']:.2f}s")
            print(f"  Final validation accuracy: {stats['final_val_accuracy']:.3f}")
            print(f"  Convergence epoch: {stats['convergence_epoch']}")

    print("\nExplainer Performance:")
    for func_name, metrics in all_explainer_metrics.items():
        print(f"\n{func_name}:")
        for model_explainer, stats in metrics.items():
            print(f"{model_explainer}:")
            print(f"  Explanation time: {stats['explanation_time']:.3f}s")
            print(f"  Term precision: {stats['term_precision']:.3f}")
            print(f"  Term recall: {stats['term_recall']:.3f}")
            print(f"  Term F1 score: {stats['term_f1']:.3f}")

if __name__ == "__main__":
    main()
