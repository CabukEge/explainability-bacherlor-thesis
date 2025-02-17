import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from data import create_test_case_for_term  # Assumes this helper is defined in data.py

def verify_term(term: tuple, model, threshold: float = 0.5, log_pred: bool = False) -> bool:
    """
    Verify if a term is valid by checking model prediction.
    If log_pred is True, log the predicted probability for the test case.
    """
    test_case = create_test_case_for_term(term)
    pred = None
    if isinstance(model, DecisionTreeClassifier):
        pred = model.predict_proba(test_case.reshape(1, -1))[0, 1]
    elif isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            # For CNN, reshape accordingly.
            if hasattr(model, 'conv1'):
                x_tensor = torch.FloatTensor(test_case).reshape(1, 1, 3, 3)
            else:
                x_tensor = torch.FloatTensor(test_case).reshape(1, 3, 3)
            output = model(x_tensor)
            pred = torch.softmax(output, dim=1)[0, 1].item()
    if log_pred:
        print(f"Term {term} predicted probability: {pred:.4f}")
    return pred is not None and pred > threshold

def parse_dnf_to_terms(dnf_str: str) -> list:
    """
    Convert DNF string to list of terms
    """
    if dnf_str == "False":
        return []
    terms = []
    for term_str in dnf_str.replace("(", "").replace(")", "").split(" ∨ "):
        term = [int(var.split("_")[1]) - 1 for var in term_str.strip().split(" ∧ ")]
        terms.append(tuple(sorted(term)))
    return sorted(terms)

def are_dnfs_equivalent(dnf1: str, dnf2: str) -> bool:
    """
    Check if two DNF expressions are logically equivalent
    """
    if dnf1 == dnf2:
        return True
    terms1 = parse_dnf_to_terms(dnf1)
    terms2 = parse_dnf_to_terms(dnf2)
    return terms1 == terms2

def compute_term_metrics(reconstructed_terms: set, actual_terms: set) -> dict:
    """
    Compute precision, recall, and F1 score for term reconstruction.
    """
    true_positives = len(reconstructed_terms.intersection(actual_terms))
    false_positives = len(reconstructed_terms - actual_terms)
    false_negatives = len(actual_terms - reconstructed_terms)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
