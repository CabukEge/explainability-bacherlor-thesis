import itertools
import numpy as np
from typing import Callable, List, Tuple
import bolean_functions

def generate_disjoint_datasets(num_samples: int, 
                               label_function: Callable[[np.ndarray], bool]) -> Tuple[List[Tuple[np.ndarray, bool]], List[Tuple[np.ndarray, bool]]]:
    """
    Generiert zwei disjunkte Datensätze von 9-Element-Vektoren für Klassifikationszwecke.
    Jeder Datensatz erhält binäre Labels basierend auf der gegebenen boolschen Funktion.

    Parameters:
        num_samples (int): Anzahl der Proben in jedem Datensatz.
        label_function (Callable): boolsche Funktion für das Labeln von dataset1.

    Returns:
        tuple: Zwei Listen, die Tupel aus Vektoren und Labels enthalten.
    """
    # Alle möglichen 9-dimensionalen binären Vektoren als NumPy Arrays
    all_vectors = [np.array(vector) for vector in itertools.product([0, 1], repeat=9)]
    
    # Listen für die disjunkten Datensätze
    dataset1, dataset2 = [], []
    
    # Sicherstellen, dass wir genug Vektoren haben
    if len(all_vectors) < num_samples * 2:
        raise ValueError("Nicht genügend Vektoren für die gewünschte Anzahl an Samples in beiden Datensätzen.")
    
    # Iteriere über alle Vektoren und ordne sie den Datensätzen zu
    for vector in all_vectors:
        if label_function(vector):
            dataset1.append((vector, True))  # Label für dataset1
        else:
            dataset2.append((vector, True))  # Label für dataset2

    return dataset1, dataset2
