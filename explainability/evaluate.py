import torch
import numpy as np
from data import generate_data
from models import FCN, CNN, train_model, train_tree
from explainers.lime import LIMEExplainer
import matplotlib.pyplot as plt
from boolean_functions import dnf_example, min_ones, consecutive_ones

def plot_explanation(input_data: np.ndarray, explanation: dict, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(input_data.reshape(3, 3), cmap='binary')
    ax1.set_title('Input')
    coeffs = explanation['coefficients'].reshape(3, 3)
    vmax = np.max(np.abs(coeffs))
    im = ax2.imshow(coeffs, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax2.set_title(f'LIME\nAcc: {explanation["local_accuracy"]:.2f}')
    plt.colorbar(im)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def reconstruct_dnf_from_model(model, explainer):
    test_cases = [
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),  # Term 1
        np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),  # Term 2
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),  # Term 3
        np.array([1, 1, 0, 1, 1, 1, 0, 0, 0]),  # Term 1 OR 2
        np.array([1, 1, 0, 0, 0, 0, 1, 1, 1]),  # Term 1 OR 3
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),  # Term 2 OR 3
        np.array([1, 1, 0, 1, 1, 1, 1, 1, 1]),  # All terms
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])   # False
    ]
    
    terms = []
    for x in test_cases:
        exp = explainer.explain(x)
        pred = exp['prediction']
        if pred > 0.5:
            term = []
            coeffs = exp['coefficients']
            for i, coeff in enumerate(coeffs):
                if abs(coeff) > 0.3:
                    term.append(f"x{i+1}")
            if term:
                terms.append(term)
    
    dnf_terms = []
    if any('x1' in t and 'x2' in t for t in terms):
        dnf_terms.append("(x₁ ∧ x₂)")
    if any('x4' in t and 'x5' in t and 'x6' in t for t in terms):
        dnf_terms.append("(x₄ ∧ x₅ ∧ x₆)")
    if any('x7' in t and 'x8' in t and 'x9' in t for t in terms):
        dnf_terms.append("(x₇ ∧ x₈ ∧ x₉)")
    
    return " ∨ ".join(dnf_terms) if dnf_terms else "False"

def evaluate(boolean_func, func_name="", test_inputs=None):
    print(f"\nTesting function: {func_name}")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_data(boolean_func)
    
    if test_inputs is None:
        test_inputs = [
            np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        ]
    
    models = {
        'fcn': FCN(),
        'tree': train_tree(X_train, y_train)
    }
    
    train_model(models['fcn'], (X_train, y_train), (X_val, y_val))
    
    for name, model in models.items():
        print(f"\nModel: {name}")
        explainer = LIMEExplainer(model)
        
        reconstructed_dnf = reconstruct_dnf_from_model(model, explainer)
        print(f"Reconstructed function: {reconstructed_dnf}")
        
        for i, test_input in enumerate(test_inputs):
            exp = explainer.explain(test_input)
            print(f"\nCase {i}:")
            print(f"Input: {test_input}")
            print(f"True label: {boolean_func(test_input)}")
            print(f"Model prediction: {exp['prediction']:.3f}")
            print(f"Local accuracy: {exp['local_accuracy']:.3f}")
            
            coeffs = exp['coefficients']
            sorted_idx = np.argsort(np.abs(coeffs))[::-1]
            print("Most important features (LIME):")
            for idx in sorted_idx[:3]:
                print(f"x{idx+1}: {coeffs[idx]:.3f}")

if __name__ == "__main__":
    # Now only testing the DNF example
    evaluate(dnf_example, "DNF: (x₁ ∧ x₂) ∨ (x₄ ∧ x₅ ∧ x₆) ∨ (x₇ ∧ x₈ ∧ x₉)")