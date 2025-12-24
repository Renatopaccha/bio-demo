from setuptools import setup, find_packages

setup(
    name="biostat-easy",
    version="1.0.0",
    description="Bioestadística para investigación médica",
    author="BioStat Team",
    packages=find_packages(),
    python_requires=">=3.8",
 install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.13.0",
        "lifelines>=0.27.0",
        "streamlit>=1.0.0",
        "pingouin>=0.5.0",  # <--- AGREGA ESTA LÍNEA
        "typing-extensions>=4.0.0; python_version < '3.10'",
    ],
    extras_require={
        "dev": [
            "mypy",
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
    },
)
