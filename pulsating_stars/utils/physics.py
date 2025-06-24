# -*- coding: utf-8 -*-
"""
Physics utilities for pulsating star analysis.

This module contains physical constants, stellar parameters,
and pulsation theory calculations.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# Physical constants
SOLAR_MASS = 1.9891e30  # kg
SOLAR_RADIUS = 6.96e8   # m
SOLAR_LUMINOSITY = 3.828e26  # W
G = 6.67430e-11  # m^3 kg^-1 s^-2
C = 2.99792458e8  # m/s
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4


class StellarParameters:
    """Container for stellar parameters."""
    
    def __init__(self, mass: float, radius: float, temperature: float,
                 luminosity: Optional[float] = None):
        """
        Initialize stellar parameters.
        
        Parameters:
        -----------
        mass : float
            Stellar mass in solar masses
        radius : float
            Stellar radius in solar radii  
        temperature : float
            Effective temperature in K
        luminosity : float, optional
            Luminosity in solar luminosities (calculated if not provided)
        """
        self.mass = mass
        self.radius = radius
        self.temperature = temperature
        
        if luminosity is None:
            # Calculate luminosity from Stefan-Boltzmann law
            self.luminosity = 4 * np.pi * (radius * SOLAR_RADIUS)**2 * \
                            STEFAN_BOLTZMANN * temperature**4 / SOLAR_LUMINOSITY
        else:
            self.luminosity = luminosity
            
    @property
    def surface_gravity(self) -> float:
        """Surface gravity in m/s^2."""
        return G * self.mass * SOLAR_MASS / (self.radius * SOLAR_RADIUS)**2
        
    @property
    def log_g(self) -> float:
        """Log surface gravity (cgs)."""
        g_cgs = self.surface_gravity * 100  # Convert to cm/s^2
        return np.log10(g_cgs)
        
    @property
    def mean_density(self) -> float:
        """Mean density in kg/m^3."""
        volume = (4/3) * np.pi * (self.radius * SOLAR_RADIUS)**3
        return (self.mass * SOLAR_MASS) / volume
        
    @property
    def dynamical_timescale(self) -> float:
        """Dynamical timescale in seconds."""
        return np.sqrt((self.radius * SOLAR_RADIUS)**3 / (G * self.mass * SOLAR_MASS))


def get_typical_parameters(star_type: str) -> StellarParameters:
    """
    Get typical stellar parameters for different variable star types.
    
    Parameters:
    -----------
    star_type : str
        Type of variable star
        
    Returns:
    --------
    StellarParameters
        Typical parameters for the star type
    """
    parameters = {
        'cepheid': StellarParameters(mass=5.0, radius=50.0, temperature=5500),
        'rr_lyrae': StellarParameters(mass=0.65, radius=5.5, temperature=6200),
        'delta_scuti': StellarParameters(mass=1.8, radius=2.2, temperature=7500),
        'beta_cephei': StellarParameters(mass=12.0, radius=6.0, temperature=22000),
        'main_sequence': StellarParameters(mass=1.0, radius=1.0, temperature=5778)
    }
    
    return parameters.get(star_type.lower(), parameters['main_sequence'])


def pulsation_constant(period_days: float, mass: float, radius: float,
                      temperature: float) -> float:
    """
    Calculate the pulsation constant Q.
    
    Parameters:
    -----------
    period_days : float
        Pulsation period in days
    mass : float
        Stellar mass in solar masses
    radius : float
        Stellar radius in solar radii
    temperature : float
        Effective temperature in K
        
    Returns:
    --------
    float
        Pulsation constant Q
    """
    period_seconds = period_days * 24 * 3600
    stellar_params = StellarParameters(mass, radius, temperature)
    
    # Q = P * sqrt(rho / rho_sun)
    rho_sun = SOLAR_MASS / ((4/3) * np.pi * SOLAR_RADIUS**3)
    Q = period_seconds * np.sqrt(stellar_params.mean_density / rho_sun)
    
    return Q


def period_luminosity_relation(luminosity: float, star_type: str = 'cepheid') -> float:
    """
    Calculate expected period from period-luminosity relation.
    
    Parameters:
    -----------
    luminosity : float
        Luminosity in solar luminosities
    star_type : str
        Type of variable star
        
    Returns:
    --------
    float
        Expected period in days
    """
    # Period-Luminosity relations (approximate)
    if star_type.lower() == 'cepheid':
        # Classical Cepheid P-L relation
        log_P = 0.75 * (np.log10(luminosity) - 1.5) + 1.0
        return 10**log_P
    elif star_type.lower() == 'rr_lyrae':
        # RR Lyrae have roughly constant luminosity and period
        return 0.5  # ~0.5 day typical period
    else:
        # Generic relation based on stellar structure
        return 1.0  # Default 1 day


def instability_strip_check(temperature: float, luminosity: float) -> Dict[str, bool]:
    """
    Check if a star lies in various instability strips.
    
    Parameters:
    -----------
    temperature : float
        Effective temperature in K
    luminosity : float
        Luminosity in solar luminosities
        
    Returns:
    --------
    dict
        Boolean flags for each instability strip
    """
    log_L = np.log10(luminosity)
    
    result = {
        'cepheid': 5000 < temperature < 6500 and log_L > 2.5,
        'rr_lyrae': 6000 < temperature < 7500 and 1.5 < log_L < 2.0,
        'delta_scuti': 6500 < temperature < 8500 and 0.8 < log_L < 1.8,
        'beta_cephei': temperature > 20000 and log_L > 3.0,
        'gamma_doradus': 6800 < temperature < 7400 and 0.3 < log_L < 1.2
    }
    
    return result


def adiabatic_frequency(mode_degree: int, radial_order: int,
                       stellar_params: StellarParameters) -> float:
    """
    Estimate adiabatic pulsation frequency.
    
    Parameters:
    -----------
    mode_degree : int
        Spherical harmonic degree (0 for radial)
    radial_order : int
        Radial order (number of nodes)
    stellar_params : StellarParameters
        Stellar parameters
        
    Returns:
    --------
    float
        Frequency in microHz
    """
    # Simplified asymptotic formula
    # For more accurate calculations, stellar oscillation codes are needed
    
    dynamical_freq = 1.0 / stellar_params.dynamical_timescale  # Hz
    
    if mode_degree == 0:  # Radial modes
        # Radial modes scale roughly with dynamical frequency
        freq_Hz = dynamical_freq * (radial_order + 0.5) / 10
    else:  # Non-radial modes
        # p-modes: higher frequency
        # g-modes: lower frequency (not implemented here)
        freq_Hz = dynamical_freq * mode_degree * (radial_order + 0.5) / 20
        
    return freq_Hz * 1e6  # Convert to microHz


def frequency_to_period(frequency_day: float) -> float:
    """Convert frequency in cycles/day to period in days."""
    return 1.0 / frequency_day


def period_to_frequency(period_day: float) -> float:
    """Convert period in days to frequency in cycles/day."""
    return 1.0 / period_day


def get_amplitude_scaling(star_type: str) -> Tuple[float, float]:
    """
    Get typical amplitude ranges for different star types.
    
    Parameters:
    -----------
    star_type : str
        Type of variable star
        
    Returns:
    --------
    tuple
        (min_amplitude, max_amplitude) in magnitudes
    """
    amplitudes = {
        'cepheid': (0.1, 2.0),
        'rr_lyrae': (0.3, 1.5),
        'delta_scuti': (0.001, 0.3),
        'beta_cephei': (0.01, 0.3),
        'gamma_doradus': (0.001, 0.1)
    }
    
    return amplitudes.get(star_type.lower(), (0.01, 0.5))