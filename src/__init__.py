"""
MENA Causal Analysis Package

This package contains utilities for causal inference analysis
of education and employment in MENA countries.
"""

__version__ = "1.0.0"
__author__ = "Zeyad Salah"

# expose submodules at package level
from . import data_processing
from . import causal_methods
from . import visualization

__all__ = ['data_processing', 'causal_methods', 'visualization']

