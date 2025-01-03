from setuptools import setup, find_packages

setup(
    name="explainability_toolkit",
    version="0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.7',
)