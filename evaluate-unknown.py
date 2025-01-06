import torch
import numpy as np
from data import generate_data
from models import FCN, train_model, train_tree
from explainers.lime import LIMEExplainer
from boolean_functions import dnf_example, dnf_simple, dnf_complex
from itertools import combinations, product
from sklearn.tree import DecisionTreeClassifier
from sympy.logic.boolalg import Or, And, Not
from sympy import symbols

def create_test_case_for_term(term: tuple, n: int = 9) -> np.ndarray:
    """Create a test case where specified variables are 1, others are 0"""
    x = np.zeros(n)
    x[list(term)] = 1
    return x

def verify_term(term: tuple, model, threshold: float = 0.5) -> bool:
    """Verify if a term is valid by checking model prediction"""
    test_case = create_test_case_for_term(term)
    
    if isinstance(model, DecisionTreeClassifier):
        pred = model.predict_proba(test_case.reshape(1, -1))[0, 1]
        return pred > threshold
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(test_case).reshape(1, 3, 3)
            output = model(x_tensor)
            pred = torch.softmax(output, dim=1)[0, 1].item()
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

def find_terms_with_lime(model, lime_explainer) -> list:
    """Find minimal terms using LIME explanations"""
    terms = []
    n_vars = 9

    test_cases = []
    for term_size in range(1, 4):
        for term in combinations(range(n_vars), term_size):
            test_case = np.zeros(9)
            test_case[list(term)] = 1
            test_cases.append(test_case)

    print("\nAnalyzing terms with LIME:")
    for test_case in test_cases:
        explanation = lime_explainer.explain(test_case)
        coefficients = explanation['coefficients']
        significant_vars = tuple(sorted(i for i, coef in enumerate(coefficients) 
                                         if coef > 0.1 and test_case[i] == 1))
        if significant_vars and verify_term_is_minimal(significant_vars, terms, model):
            print(f"Found minimal term: {significant_vars}")
            terms.append(significant_vars)

    return sorted(set(terms), key=len)

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

def reconstruct_dnf_from_model(model) -> str:
    """Reconstruct DNF using LIME explanations"""
    lime_explainer = LIMEExplainer(model, num_samples=1000)
    terms = find_terms_with_lime(model, lime_explainer)

    if not terms:
        return "False"

    dnf_terms = [f"({' ∧ '.join([f'x_{i+1}' for i in sorted(term)])})" for term in terms]
    return " ∨ ".join(sorted(set(dnf_terms)))

def get_function_str(func) -> str:
    """Get the string representation of the target function"""
    if func == dnf_example:
        return "(x_1 ∧ x_2) ∨ (x_4 ∧ x_5 ∧ x_6) ∨ (x_7 ∧ x_8 ∧ x_9)"
    elif func == dnf_simple:
        return "(x_1 ∧ x_2) ∨ (x_5 ∧ x_6)"
    elif func == dnf_complex:
        return "(x_1 ∧ x_2 ∧ x_3) ∨ (x_4 ∧ x_5) ∨ (x_7 ∧ x_8 ∧ x_9)"
    return "Unknown function"

def evaluate(boolean_func, func_name=""):
    """Train models and evaluate LIME explanations."""
    print(f"\nTesting function: {func_name}")
    target_dnf = get_function_str(boolean_func)
    print(f"Target DNF: {target_dnf}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_data(boolean_func)

    models = {
        'FCN': FCN(),
        'Decision Tree': train_tree(X_train, y_train)
    }

    train_model(models['FCN'], (X_train, y_train), (X_val, y_val))

    correct_predictions = 0
    for name, model in models.items():
        print(f"\nModel: {name}")
        reconstructed_dnf = reconstruct_dnf_from_model(model)
        print(f"Reconstructed DNF: {reconstructed_dnf}")
        if are_dnfs_equivalent(reconstructed_dnf, target_dnf):
            correct_predictions += 1
            print("✓ Correct reconstruction! (Logically equivalent)")
        else:
            print("✗ Incorrect reconstruction")

    return correct_predictions, len(models)

def main():
    """Main function to run the evaluation"""
    functions = [
        (dnf_simple, "Simple DNF"),
        (dnf_example, "Example DNF"),
        (dnf_complex, "Complex DNF")
    ]

    total_correct = 0
    total_models = 0

    for func, name in functions:
        correct, total = evaluate(func, name)
        total_correct += correct
        total_models += total

    print(f"\nFinal Results:")
    print(f"Total correct reconstructions: {total_correct}/{total_models}")
    print(f"Overall accuracy: {(total_correct/total_models)*100:.1f}%")

if __name__ == "__main__":
    main()
