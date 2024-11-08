import numpy as np


def first_example (vector: np.ndarray) -> bool:
    return (vector[0] and vector[1]) or (vector[3] and vector[4] and vector[5]) or (vector[6] and vector[7] and vector[8])

# Beispiel für benutzerdefinierte boolsche Funktionen
def min_x_amount_equals_one(vector: np.ndarray, x: int) -> bool:
    # Beispiel-boolsche Funktion: mindestens 5 Elemente sind 1
    return np.sum(vector) >= x

def x_many_consecutively_one(vector: np.ndarray, x: int) -> bool:
    # Beispiel-boolsche Funktion: mindestens x aufeinanderfolgende 1er
    for i in range(9):
        if np.all(vector[i:i+x] == 1):
            return True
    return False

