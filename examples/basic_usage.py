#!/usr/bin/env python
"""
Basic usage example for pulsating star analysis toolkit.

This script demonstrates how to:
1. Generate simulated lightcurves
2. Visualize them
3. Train an autoencoder for anomaly detection
"""

import numpy as np
from pulsating_stars import LightcurveSimulator
from pulsating_stars.visualization import plot_comparison, plot_combined
from pulsating_stars.ml import GRUAutoencoder

def main():
    print("Pulsating Star Analysis - Basic Usage Example")
    print("=" * 50)
    
    # 1. Generate sample lightcurves
    print("\n1. Generating sample lightcurves...")
    simulator = LightcurveSimulator(observation_days=1.0)
    
    # Generate a few normal lightcurves for training
    normal_lcs = [simulator.generate_normal_lightcurve() for _ in range(10)]
    
    # Generate test samples
    test_normal = simulator.generate_normal_lightcurve()
    test_anomaly = simulator.generate_anomaly_lightcurve()
    
    print(f"Generated {len(normal_lcs)} training lightcurves")
    print("Generated 1 normal and 1 anomaly test lightcurve")
    
    # 2. Visualize comparison
    print("\n2. Visualizing lightcurves...")
    fig = plot_comparison(test_normal, test_anomaly, show=False)
    print("Comparison plot created (set show=True to display)")
    
    # 3. Train autoencoder
    print("\n3. Training GRU autoencoder...")
    model = GRUAutoencoder(encoding_dim=16)
    
    # Train on normal lightcurves
    history = model.fit(normal_lcs, epochs=50, verbose=1)
    
    # 4. Test anomaly detection
    print("\n4. Testing anomaly detection...")
    
    # Predict reconstruction errors
    normal_error = model.predict([test_normal])[0]
    anomaly_error = model.predict([test_anomaly])[0]
    
    print(f"Normal lightcurve reconstruction error: {normal_error:.6f}")
    print(f"Anomaly lightcurve reconstruction error