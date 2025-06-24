# -*- coding: utf-8 -*-
"""
Visualization module for pulsating star lightcurves.

This module provides interactive plotting functions using Plotly for
lightcurves, frequency spectra, and combined displays.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightkurve import LightCurve
from scipy.signal import find_peaks
from typing import Optional, Tuple, List
import os


def to_native(array: np.ndarray) -> np.ndarray:
    """Convert array to native byte order float64."""
    return np.array(array, dtype=np.float64)


def plot_lightcurve(lc: LightCurve, title: Optional[str] = None, 
                   show: bool = True) -> go.Figure:
    """
    Plot a lightcurve using Plotly.
    
    Parameters:
    -----------
    lc : LightCurve
        Input lightcurve object
    title : str, optional
        Plot title
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if title is None:
        title = "Light Curve"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc.time.value,
        y=lc.flux.value,
        error_y=dict(type='data', array=lc.flux_err.value, visible=False),
        mode='lines',
        name='Flux',
        line=dict(width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time [day]",
        yaxis_title="Flux [e⁻/s]",
        template="plotly_white",
        showlegend=False
    )
    
    if show:
        fig.show()
    
    return fig


def compute_spectrum(lc: LightCurve) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT spectrum of a lightcurve.
    
    Parameters:
    -----------
    lc : LightCurve
        Input lightcurve
        
    Returns:
    --------
    tuple
        (frequencies, power) arrays
    """
    flux_detrended = lc.flux.value - np.mean(lc.flux.value)
    n = len(flux_detrended)
    dt = lc.time.value[1] - lc.time.value[0]
    
    fft_vals = np.fft.rfft(flux_detrended)
    frequencies = np.fft.rfftfreq(n, d=dt)
    power = np.abs(fft_vals)
    
    return frequencies, power


def find_peaks_in_spectrum(frequencies: np.ndarray, power: np.ndarray,
                          n_peaks: int = 3, min_height_ratio: float = 0.1
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find dominant peaks in frequency spectrum.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency array
    power : np.ndarray
        Power array
    n_peaks : int
        Number of top peaks to return
    min_height_ratio : float
        Minimum peak height as fraction of maximum
        
    Returns:
    --------
    tuple
        (peak_frequencies, peak_powers)
    """
    peaks, props = find_peaks(power, height=np.max(power) * min_height_ratio)
    
    if len(peaks) > 0:
        heights = props.get('peak_heights', np.array([]))
        top_indices = np.argsort(heights)[-n_peaks:][::-1]
        peak_freqs = frequencies[peaks[top_indices]]
        peak_powers = heights[top_indices]
    else:
        peak_freqs = np.array([])
        peak_powers = np.array([])
    
    return peak_freqs, peak_powers


def plot_spectrum(lc: LightCurve, title: Optional[str] = None,
                 mark_peaks: bool = True, n_peaks: int = 3,
                 show: bool = True) -> go.Figure:
    """
    Plot frequency spectrum of a lightcurve.
    
    Parameters:
    -----------
    lc : LightCurve
        Input lightcurve
    title : str, optional
        Plot title
    mark_peaks : bool
        Whether to mark dominant peaks
    n_peaks : int
        Number of peaks to mark
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if title is None:
        title = "Frequency Spectrum"
    
    frequencies, power = compute_spectrum(lc)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=power,
        mode='lines',
        name='Power Spectrum',
        line=dict(width=1)
    ))
    
    if mark_peaks:
        peak_freqs, peak_powers = find_peaks_in_spectrum(
            frequencies, power, n_peaks=n_peaks
        )
        
        if len(peak_freqs) > 0:
            fig.add_trace(go.Scatter(
                x=peak_freqs,
                y=peak_powers,
                mode='markers+text',
                text=[f"{f:.2f} d⁻¹" for f in peak_freqs],
                textposition="top center",
                marker=dict(color='red', size=8),
                name='Peaks'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency [cycles/day]",
        yaxis_title="Amplitude",
        template="plotly_white"
    )
    
    if show:
        fig.show()
    
    return fig


def plot_combined(lc: LightCurve, title: Optional[str] = None,
                 show: bool = True) -> go.Figure:
    """
    Plot lightcurve and spectrum in a combined figure.
    
    Parameters:
    -----------
    lc : LightCurve
        Input lightcurve
    title : str, optional
        Overall title
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    go.Figure
        Plotly figure object with subplots
    """
    if title is None:
        title = "Lightcurve Analysis"
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Light Curve', 'Frequency Spectrum'),
        vertical_spacing=0.12
    )
    
    # Add lightcurve
    fig.add_trace(
        go.Scatter(
            x=lc.time.value,
            y=lc.flux.value,
            mode='lines',
            name='Flux',
            line=dict(width=1)
        ),
        row=1, col=1
    )
    
    # Add spectrum
    frequencies, power = compute_spectrum(lc)
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power,
            mode='lines',
            name='Power',
            line=dict(width=1)
        ),
        row=2, col=1
    )
    
    # Add peaks to spectrum
    peak_freqs, peak_powers = find_peaks_in_spectrum(frequencies, power)
    if len(peak_freqs) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_freqs,
                y=peak_powers,
                mode='markers+text',
                text=[f"{f:.2f}" for f in peak_freqs],
                textposition="top center",
                marker=dict(color='red', size=6),
                name='Peaks',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700
    )
    
    fig.update_xaxes(title_text="Time [day]", row=1, col=1)
    fig.update_yaxes(title_text="Flux [e⁻/s]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [cycles/day]", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    
    if show:
        fig.show()
    
    return fig


def plot_comparison(normal_lc: LightCurve, anomaly_lc: LightCurve,
                   show: bool = True) -> go.Figure:
    """
    Plot comparison between normal and anomaly lightcurves.
    
    Parameters:
    -----------
    normal_lc : LightCurve
        Normal lightcurve
    anomaly_lc : LightCurve
        Anomaly lightcurve
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    go.Figure
        Plotly comparison figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Normal Light Curve', 'Normal Spectrum',
                       'Anomaly Light Curve', 'Anomaly Spectrum'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Normal lightcurve
    fig.add_trace(
        go.Scatter(
            x=normal_lc.time.value,
            y=normal_lc.flux.value,
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Normal spectrum
    norm_freq, norm_power = compute_spectrum(normal_lc)
    fig.add_trace(
        go.Scatter(
            x=norm_freq,
            y=norm_power,
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Anomaly lightcurve
    fig.add_trace(
        go.Scatter(
            x=anomaly_lc.time.value,
            y=anomaly_lc.flux.value,
            mode='lines',
            name='Anomaly',
            line=dict(color='red', width=1)
        ),
        row=2, col=1
    )
    
    # Anomaly spectrum
    anom_freq, anom_power = compute_spectrum(anomaly_lc)
    fig.add_trace(
        go.Scatter(
            x=anom_freq,
            y=anom_power,
            mode='lines',
            name='Anomaly',
            line=dict(color='red', width=1),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Normal vs Anomaly Comparison",
        template="plotly_white",
        height=800
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time [day]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [cycles/day]", row=1, col=2)
    fig.update_xaxes(title_text="Time [day]", row=2, col=1)
    fig.update_xaxes(title_text="Frequency [cycles/day]", row=2, col=2)
    
    fig.update_yaxes(title_text="Flux [e⁻/s]", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    fig.update_yaxes(title_text="Flux [e⁻/s]", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=2)
    
    if show:
        fig.show()
    
    return fig