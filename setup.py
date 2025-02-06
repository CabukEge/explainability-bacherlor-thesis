from setuptools import setup, find_packages

setup(
    name="explainability",
    version="0.1",
    packages=['explainers'],  # Just include the explainers package
    py_modules=[             # Include individual modules
        'boolean_functions',
        'data',
        'models',
        'evaluate',
        'run_tests'
    ],
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn',
        'matplotlib',
        'sympy',
        'scipy'
    ],
    python_requires='>=3.9',
)