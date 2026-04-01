"""
ThermoTorch Setup

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="thermotorch",
    version="0.1.0",
    description="PyTorch framework for Potential-Flow Embeddings and Thermodynamic Computing",
    author="Extropic Development Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "ode": ["torchdiffeq>=0.2.3"],
        "vae": ["diffusers>=0.21.0", "transformers>=4.30.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)