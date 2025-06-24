# -*- coding: utf-8 -*-
"""
Input/Output utilities for lightcurve data.

This module provides functions for loading and saving lightcurve data
in various formats, with special handling for FITS files.
"""

import os
import glob
import numpy as np
from astropy.io import fits
from lightkurve import LightCurve
from typing import List, Optional, Union, Tuple
import warnings

# Suppress astropy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")


def to_native(array: np.ndarray) -> np.ndarray:
    """Convert array to native byte order float64."""
    return np.array(array, dtype=np.float64)


def load_fits_lightcurve(filepath: str) -> LightCurve:
    """
    Load a single FITS lightcurve file.
    
    Parameters:
    -----------
    filepath : str
        Path to FITS file
        
    Returns:
    --------
    LightCurve
        Loaded lightcurve object
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        time = to_native(data['TIME'])
        flux = to_native(data['FLUX'])
        
        if 'FLUX_ERR' in data.names:
            flux_err = to_native(data['FLUX_ERR'])
        elif 'FLUXERROR' in data.names:
            flux_err = to_native(data['FLUXERROR'])
        else:
            flux_err = np.full_like(flux, 0.01)
            
        # Extract metadata
        meta = {}
        if len(hdul) > 0:
            header = hdul[0].header
            for key in header:
                meta[key] = header[key]
                
    return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)


def load_lightcurves(directory: str, 
                    pattern: str = "*.fits",
                    max_files: Optional[int] = None) -> List[LightCurve]:
    """
    Load multiple lightcurve files from a directory.
    
    Parameters:
    -----------
    directory : str
        Directory containing lightcurve files
    pattern : str
        File pattern to match
    max_files : int, optional
        Maximum number of files to load
        
    Returns:
    --------
    list of LightCurve
        List of loaded lightcurves
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    file_pattern = os.path.join(directory, pattern)
    files = sorted(glob.glob(file_pattern))
    
    if max_files is not None:
        files = files[:max_files]
        
    lightcurves = []
    for filepath in files:
        try:
            lc = load_fits_lightcurve(filepath)
            lc.meta['FILENAME'] = os.path.basename(filepath)
            lightcurves.append(lc)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            
    return lightcurves


def save_lightcurve(lc: LightCurve, filepath: str, 
                   overwrite: bool = True) -> None:
    """
    Save a lightcurve to FITS format.
    
    Parameters:
    -----------
    lc : LightCurve
        Lightcurve to save
    filepath : str
        Output file path
    overwrite : bool
        Whether to overwrite existing files
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to FITS
    lc.to_fits(filepath, overwrite=overwrite)


def load_dataset(data_dir: str = "data") -> Tuple[List[LightCurve], List[LightCurve]]:
    """
    Load complete dataset with normal and anomaly lightcurves.
    
    Parameters:
    -----------
    data_dir : str
        Base data directory
        
    Returns:
    --------
    tuple
        (normal_lightcurves, anomaly_lightcurves)
    """
    normal_dir = os.path.join(data_dir, "normal")
    anomaly_dir = os.path.join(data_dir, "anomaly")
    
    normal_lcs = load_lightcurves(normal_dir)
    anomaly_lcs = load_lightcurves(anomaly_dir)
    
    print(f"Loaded {len(normal_lcs)} normal and {len(anomaly_lcs)} anomaly lightcurves")
    
    return normal_lcs, anomaly_lcs


def export_to_csv(lc: LightCurve, filepath: str) -> None:
    """
    Export lightcurve to CSV format.
    
    Parameters:
    -----------
    lc : LightCurve
        Lightcurve to export
    filepath : str
        Output CSV file path
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'time': lc.time.value,
        'flux': lc.flux.value,
        'flux_err': lc.flux_err.value
    })
    
    df.to_csv(filepath, index=False)


def get_file_info(directory: str) -> List[dict]:
    """
    Get information about files in a directory.
    
    Parameters:
    -----------
    directory : str
        Directory to scan
        
    Returns:
    --------
    list of dict
        File information dictionaries
    """
    files = glob.glob(os.path.join(directory, "*.fits"))
    file_info = []
    
    for filepath in files:
        try:
            lc = load_fits_lightcurve(filepath)
            info = {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'n_points': len(lc.time),
                'duration': float(lc.time.max() - lc.time.min()),
                'mean_flux': float(np.mean(lc.flux.value)),
                'std_flux': float(np.std(lc.flux.value))
            }
            
            # Add metadata if available
            for key in ['TELESCOP', 'INSTRUME', 'OBJECT']:
                if key in lc.meta:
                    info[key.lower()] = lc.meta[key]
                    
            file_info.append(info)
            
        except Exception as e:
            print(f"Could not process {filepath}: {e}")
            
    return file_info