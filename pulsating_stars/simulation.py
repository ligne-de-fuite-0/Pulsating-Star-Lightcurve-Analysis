# -*- coding: utf-8 -*-
"""
Lightcurve simulation module for pulsating variable stars.

This module provides functions to generate realistic lightcurves for different
types of pulsating stars, including physics-based non-linear transformations.
"""

import numpy as np
from lightkurve import LightCurve
import datetime
import os
from typing import Tuple, Optional, List, Union


def nonlinear_transform(signal: np.ndarray, freq: float, method: str = 'cepheid', 
                       strength: Optional[float] = None) -> np.ndarray:
    """
    Apply non-linear transformation to sinusoidal signal based on stellar physics.

    Parameters:
    -----------
    signal : np.ndarray
        Input sinusoidal signal array
    freq : float
        Signal frequency in cycles/day
    method : str
        Transformation method: 'cepheid', 'rr_lyrae', 'delta_scuti', 'beta_cephei'
    strength : float, optional
        Manual transformation strength. If None, calculated from physics

    Returns:
    --------
    np.ndarray
        Transformed non-sinusoidal signal
    """
    if strength is None:
        # Physics-based strength calculation
        if freq < 1:  # Low frequency - Classical Cepheids
            base_strength = 0.5
        elif freq < 5:  # Medium frequency - RR Lyrae
            base_strength = 0.3
        elif freq < 15:  # Higher frequency - some δ Scuti
            base_strength = 0.2
        else:  # High frequency - δ Scuti, β Cephei
            base_strength = 0.1

        # Add random variation within physical limits
        strength = base_strength * (1 + 0.2 * np.random.randn())
        strength = max(0.05, min(0.6, strength))

    if method == 'cepheid':
        # Asymmetric sawtooth with rapid rise, slow decline
        phase = np.linspace(0, 2*np.pi, len(signal))
        asymmetry = 0.3 + 0.2 * strength
        new_phase = phase + asymmetry * np.sin(phase)
        return np.sin(new_phase) * (1 + strength * np.sin(phase + np.pi/4))

    elif method == 'rr_lyrae':
        # RR Lyrae with potential Blazhko effect
        t = np.linspace(0, 2*np.pi, len(signal))
        blazhko_period = 20 + 10 * np.random.random()
        blazhko_phase = 2*np.pi * np.random.random()

        # Amplitude modulation
        amp_mod = 1 + strength * np.sin(2*np.pi * t/blazhko_period + blazhko_phase)
        phase = t + 0.2 * strength * np.sin(t)
        return amp_mod * np.sin(phase)

    elif method == 'delta_scuti':
        # Multi-mode with harmonic coupling
        num_components = 2 + int(2 * strength)
        result = signal.copy()

        for i in range(2, num_components + 1):
            harm_amp = strength * 0.3 / (i**1.5)
            harm_phase = 2*np.pi * np.random.random()
            result += harm_amp * np.sin(i * 2*np.pi * freq * 
                                      np.linspace(0, 1, len(signal)) + harm_phase)

        # Weak non-linear coupling
        mod_strength = strength * 0.15
        t = np.linspace(0, 2*np.pi, len(signal))
        return result + mod_strength * signal * np.sin(3*t)

    elif method == 'beta_cephei':
        # β Cephei with subtle harmonics and asymmetry
        harmonics = signal + strength * 0.2 * np.sin(4*np.pi * freq * 
                                                    np.linspace(0, 1, len(signal)))
        t = np.linspace(0, 2*np.pi, len(signal))
        asymmetry = strength * 0.1 * np.sin(2*t + np.pi/3)
        return harmonics + asymmetry

    else:
        return signal


class LightcurveSimulator:
    """Class for generating simulated pulsating star lightcurves."""
    
    def __init__(self, cadence: float = 2/(24*60), observation_days: float = 0.5):
        """
        Initialize the lightcurve simulator.
        
        Parameters:
        -----------
        cadence : float
            Sampling rate in days (default: 2 minutes)
        observation_days : float
            Observation duration in days (default: 0.5 days)
        """
        self.cadence = cadence
        self.observation_days = observation_days
        self.time = np.arange(0, observation_days, cadence)
    
    def generate_normal_lightcurve(self, freq1: Optional[float] = None, 
                                 freq2: Optional[float] = None) -> LightCurve:
        """Generate a normal (sinusoidal) lightcurve."""
        if freq1 is None:
            freq1 = np.random.uniform(10, 30)
        if freq2 is None:
            freq2 = np.random.uniform(10, 30)
            
        amp1 = np.random.uniform(0.5, 1.5)
        amp2 = np.random.uniform(0.2, 0.8)
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        
        signal1 = amp1 * np.sin(2*np.pi * freq1 * self.time + phase1)
        signal2 = amp2 * np.sin(2*np.pi * freq2 * self.time + phase2)
        
        flux = signal1 + signal2 + 1e4  # Add baseline
        flux_err = np.full_like(flux, 0.01)
        
        return LightCurve(time=self.time, flux=flux, flux_err=flux_err)
    
    def generate_anomaly_lightcurve(self, freq1: Optional[float] = None,
                                  freq2: Optional[float] = None) -> LightCurve:
        """Generate an anomaly (non-sinusoidal) lightcurve."""
        if freq1 is None:
            freq1 = np.random.uniform(10, 30)
        if freq2 is None:
            freq2 = np.random.uniform(10, 30)
            
        amp1 = np.random.uniform(0.5, 1.5)
        amp2 = np.random.uniform(0.2, 0.8)
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        
        # Generate pure signals
        pure_signal1 = amp1 * np.sin(2*np.pi * freq1 * self.time + phase1)
        pure_signal2 = amp2 * np.sin(2*np.pi * freq2 * self.time + phase2)
        
        # Apply physics-based transformations
        method1 = self._select_method(freq1)
        method2 = self._select_method(freq2)
        
        signal1 = nonlinear_transform(pure_signal1, freq1, method=method1)
        signal2 = nonlinear_transform(pure_signal2, freq2, method=method2)
        
        flux = signal1 + signal2 + 1e4  # Add baseline
        flux_err = np.full_like(flux, 0.01)
        
        lc = LightCurve(time=self.time, flux=flux, flux_err=flux_err)
        lc.meta['TRANSFORM_METHOD1'] = method1
        lc.meta['TRANSFORM_METHOD2'] = method2
        
        return lc
    
    def _select_method(self, freq: float) -> str:
        """Select appropriate transformation method based on frequency."""
        if freq < 5:
            return np.random.choice(['cepheid', 'rr_lyrae'])
        elif freq < 15:
            return np.random.choice(['rr_lyrae', 'beta_cephei'])
        else:
            return np.random.choice(['delta_scuti', 'beta_cephei'])
    
    def generate_pair(self) -> Tuple[LightCurve, LightCurve]:
        """Generate a matched pair of normal and anomaly lightcurves."""
        normal = self.generate_normal_lightcurve()
        anomaly = self.generate_anomaly_lightcurve()
        return normal, anomaly
    
    def generate_dataset(self, n_pairs: int, output_dir: str = "data") -> None:
        """
        Generate a complete dataset of lightcurves.
        
        Parameters:
        -----------
        n_pairs : int
            Number of normal/anomaly pairs to generate
        output_dir : str
            Output directory for saved files
        """
        normal_dir = os.path.join(output_dir, "normal")
        anomaly_dir = os.path.join(output_dir, "anomaly")
        
        for directory in [normal_dir, anomaly_dir]:
            os.makedirs(directory, exist_ok=True)
        
        for i in range(n_pairs):
            normal, anomaly = self.generate_pair()
            
            # Save with timestamps
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            
            normal_file = os.path.join(normal_dir, f'normal_lightcurve_{timestamp}_{i:03d}.fits')
            anomaly_file = os.path.join(anomaly_dir, f'anomaly_lightcurve_{timestamp}_{i:03d}.fits')
            
            normal.to_fits(normal_file, overwrite=True)
            anomaly.to_fits(anomaly_file, overwrite=True)
        
        print(f"Generated {n_pairs} pairs of lightcurves in {output_dir}/")