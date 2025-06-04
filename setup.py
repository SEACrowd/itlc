# setup.py

from setuptools import find_packages, setup

setup(
    name="itlc",
    version="0.1.0",
    description="ITLC:Inference-Time Language Control,
    author="SEACrowd,
    author_email="seacrowd.research@gmail.com",
    packages=find_packages(),  # Will include the itlc/ folder
    install_requires=[
        "torch>=1.8.0",
        "transformers==4.43.0",
        "joblib",
        "numpy==1.26.1",
        "tqdm",
        "pandas",
        "datasets",
        "scikit-learn==1.6.0",  # Includes train_test_split, LDA, StandardScaler, f1_score
    ],
    python_requires=">=3.7",
)