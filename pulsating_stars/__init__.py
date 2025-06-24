"""Pulsating Star Lightcurve Analysis Toolkit.

A comprehensive Python package for simulating, analyzing, and detecting
anomalies in variable star lightcurves using modern machine learning techniques.

Modules:
    simulation: Generate realistic pulsating star lightcurves
    visualization: Interactive plotting of lightcurves and spectra
    ml: Machine learning models for anomaly detection
    utils: Utility functions for I/O and physics calculations
"""

__version__ = "0.1.0"
__author__ = "ligne-de-fuite-0"
__email__ = "your.email@example.com"

from .simulation import LightcurveSimulator, nonlinear_transform
from .visualization import plot_lightcurve, plot_spectrum, plot_combined

__all__ = [
    "LightcurveSimulator",
    "nonlinear_transform", 
    "plot_lightcurve",
    "plot_spectrum",
    "plot_combined",
]