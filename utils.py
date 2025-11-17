"""
Utility functions for Parcellated REACT analysis.

Shared functions for data loading, validation, and preprocessing.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_data_file(filepath):
    """
    Load CSV or TXT file as numpy array.
    
    Parameters
    ----------
    filepath : str
        Path to data file
        
    Returns
    -------
    np.ndarray
        Loaded data
    """
    try:
        # Try CSV first
        data = pd.read_csv(filepath, header=None).values
    except:
        # Try space/tab delimited
        data = np.loadtxt(filepath)
    
    return data


def validate_parcel_data(data, name, check_axis=1):
    """
    Validate parcellated data for common issues.
    
    Parameters
    ----------
    data : np.ndarray
        Data to validate
    name : str
        Name for error messages
    check_axis : int
        Axis to check for zeros/NaNs (0 for rows, 1 for columns)
        
    Returns
    -------
    None
        Raises ValueError if issues found
    """
    # Check for NaNs
    if np.any(np.isnan(data)):
        n_nan = np.sum(np.isnan(data))
        raise ValueError(f'{name} contains {n_nan} NaN values')
    
    # Check for infinite values
    if np.any(np.isinf(data)):
        n_inf = np.sum(np.isinf(data))
        raise ValueError(f'{name} contains {n_inf} infinite values')
    
    # Check for all-zero rows/columns
    all_zero = np.all(data == 0, axis=check_axis)
    if np.any(all_zero):
        idx = np.where(all_zero)[0]
        direction = 'rows' if check_axis == 1 else 'columns'
        raise ValueError(
            f'{name} contains all-zero {direction} at indices: {idx.tolist()}'
        )


def scale_pet_maps(pet_maps):
    """
    Scale PET maps to 0-1 range (recommended in REACT paper).
    
    Each PET map (row) is independently scaled to [0, 1].
    
    Parameters
    ----------
    pet_maps : np.ndarray
        PET receptor maps (maps x parcels)
        
    Returns
    -------
    np.ndarray
        Scaled PET maps (maps x parcels)
    """
    scaled_maps = np.zeros_like(pet_maps, dtype=float)
    
    for i in range(pet_maps.shape[0]):
        pet_map = pet_maps[i, :]
        min_val = np.min(pet_map)
        max_val = np.max(pet_map)
        
        if max_val - min_val > 0:
            scaled_maps[i, :] = (pet_map - min_val) / (max_val - min_val)
        else:
            # Constant map, set to 0.5
            scaled_maps[i, :] = 0.5
    
    return scaled_maps


def load_fmri_list(fmri_list_arg):
    """
    Load list of fMRI file paths.
    
    Parameters
    ----------
    fmri_list_arg : str
        Either path to text file with file paths, or comma-separated paths
        
    Returns
    -------
    list of str
        List of fMRI file paths
    """
    import os
    
    if os.path.isfile(fmri_list_arg):
        # Read from file
        with open(fmri_list_arg, 'r') as f:
            fmri_files = [line.strip() for line in f if line.strip()]
    else:
        # Parse as comma-separated list
        fmri_files = [f.strip() for f in fmri_list_arg.split(',') if f.strip()]
    
    # Check all files exist
    missing = [f for f in fmri_files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f'fMRI files not found: {missing}')
    
    return fmri_files


def load_pet_names(pet_names_file, n_maps, pet_maps_file):
    """
    Load or generate PET map names.
    
    Parameters
    ----------
    pet_names_file : str or None
        Path to file with custom names
    n_maps : int
        Number of PET maps
    pet_maps_file : str
        Path to PET maps file (used for default naming)
        
    Returns
    -------
    list of str
        PET map names
    """
    if pet_names_file is not None:
        with open(pet_names_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        
        if len(names) != n_maps:
            raise ValueError(
                f'Number of PET names ({len(names)}) does not match '
                f'number of PET maps ({n_maps})'
            )
    else:
        # Use filename stem as base
        base_name = Path(pet_maps_file).stem
        if n_maps == 1:
            names = [base_name]
        else:
            names = [f'{base_name}_map{i+1}' for i in range(n_maps)]
    
    return names


def get_subject_name(fmri_filepath):
    """Extract subject name from fMRI filepath."""
    return Path(fmri_filepath).stem


def load_react_results(results_dir):
    """
    Load REACT analysis results from output directory.
    
    Parameters
    ----------
    results_dir : str or Path
        Path to REACT results directory containing subject subdirectories
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'subjects': list of subject names
        - 'pet_names': list of PET map names
        - 'stage1_timeseries': dict mapping subject -> timeseries array
        - 'spatial_maps': dict mapping subject -> spatial maps array
        - 'all_spatial_maps': 3D array (subjects x maps x parcels)
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')
    
    # Find all subject directories (exclude qc_plots and other non-subject dirs)
    subject_dirs = [d for d in results_dir.iterdir() 
                   if d.is_dir() and (d / 'all_petmaps.csv').exists()]
    
    if len(subject_dirs) == 0:
        raise ValueError(f'No subject directories found in {results_dir}')
    
    subjects = [d.name for d in subject_dirs]
    
    # Load first subject to get dimensions and PET names
    first_subj_dir = subject_dirs[0]
    all_maps_file = first_subj_dir / 'all_petmaps.csv'
    
    if not all_maps_file.exists():
        raise FileNotFoundError(
            f'Missing all_petmaps.csv in {first_subj_dir}. '
            f'Has REACT analysis been run?'
        )
    
    # Load all_petmaps to get PET names
    all_maps_df = pd.read_csv(all_maps_file, header=None, index_col=0)
    pet_names = all_maps_df.index.tolist()
    n_parcels = all_maps_df.shape[1]
    n_maps = len(pet_names)
    
    # Initialize storage
    stage1_timeseries = {}
    spatial_maps = {}
    all_spatial_maps = np.zeros((len(subjects), n_maps, n_parcels))
    
    # Load data for each subject
    for i, subj_dir in enumerate(subject_dirs):
        subj_name = subj_dir.name
        
        # Load stage 1 timeseries
        ts_file = subj_dir / 'stage1_timeseries.csv'
        if ts_file.exists():
            stage1_timeseries[subj_name] = pd.read_csv(ts_file).values
        
        # Load spatial maps
        maps_file = subj_dir / 'all_petmaps.csv'
        if maps_file.exists():
            maps_df = pd.read_csv(maps_file, header=None, index_col=0)
            spatial_maps[subj_name] = maps_df.values
            all_spatial_maps[i, :, :] = maps_df.values
    
    return {
        'subjects': subjects,
        'pet_names': pet_names,
        'stage1_timeseries': stage1_timeseries,
        'spatial_maps': spatial_maps,
        'all_spatial_maps': all_spatial_maps
    }