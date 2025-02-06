import os
import numpy as np
from boolean_functions import dnf_example
from data import generate_data
from models import FCN, train_model
from explainers.lime_explainer import LIMEExplainer

def test_dnf_example():
    # Check that a simple input yields the expected output.
    x = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert dnf_example(x) == True

def test_fcn_training():
    (X_train, y_train), (X_val, y_val), _ = generate_data(dnf_example)
    model = FCN()
    model, val_acc = train_model(model, (X_train, y_train), (X_val, y_val), epochs=10, lr=0.01)
    assert val_acc[-1] > 0.5

def test_lime_explainer():
    (X_train, y_train), (X_val, y_val), _ = generate_data(dnf_example)
    model = FCN()
    model, _ = train_model(model, (X_train, y_train), (X_val, y_val), epochs=10, lr=0.01)
    explainer = LIMEExplainer(model)
    explanation = explainer.explain(X_train[0].view(-1).numpy())
    assert 'coefficients' in explanation