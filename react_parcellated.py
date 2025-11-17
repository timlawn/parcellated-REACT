#!/usr/bin/env python
"""
Parcellated REACT (Receptor-Enriched Analysis of Functional Connectivity by Targets)

A parcellation-based implementation of REACT for analyzing fMRI data
using PET receptor maps as spatial priors.

Reference: Dipasquale et al. (2019) NeuroImage 195:252-260
https://doi.org/10.1016/j.neuroimage.2019.04.007
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_file,
    validate_parcel_data,
    scale_pet_maps,
    load_fmri_list,
    load_pet_names,
    get_subject_name
)


__version__ = '1.0.0'


def setup_logging(verbose):
    """Configure logging based on verbosity level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example usage:\n'
               '  %(prog)s --fmri_list subjects.txt --pet_maps receptors.csv '
               '--out_dir ./results\n'
    )
    
    parser.add_argument(
        '--fmri_list',
        type=str,
        required=True,
        help='Path to text file containing fMRI file paths (one per line), '
             'or comma-separated list of file paths. Each file should be '
             'CSV/TXT with rows=timepoints, columns=parcels.'
    )
    
    parser.add_argument(
        '--pet_maps',
        type=str,
        required=True,
        help='Path to PET receptor map file (CSV/TXT format). '
             'Rows=receptor maps, columns=parcels. Must have same number '
             'of parcels as fMRI data. Will be automatically scaled to 0-1 range.'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Output directory where subject subdirectories will be created.'
    )
    
    parser.add_argument(
        '--pet_names',
        type=str,
        default=None,
        help='Optional: Path to text file with custom PET map names '
             '(one per line). Must match number of rows in PET file. '
             'If not provided, uses PET filename.'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output files.'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    return parser.parse_args()


def check_output_exists(out_subj_dir, force):
    """
    Check if output directory exists and handle accordingly.
    
    Parameters
    ----------
    out_subj_dir : Path
        Subject output directory
    force : bool
        Whether to overwrite existing files
        
    Returns
    -------
    bool
        True if should proceed, False if should skip
    """
    if out_subj_dir.exists():
        if force:
            logging.warning(f'Overwriting existing outputs in {out_subj_dir}')
            return True
        else:
            logging.warning(
                f'Output directory {out_subj_dir} already exists. '
                f'Use --force to overwrite. Skipping subject.'
            )
            return False
    return True


def run_stage1(fmri_data, pet_maps):
    """
    Stage 1: Use PET maps as spatial regressors to estimate time series.
    
    Parameters
    ----------
    fmri_data : np.ndarray
        fMRI data (timepoints x parcels)
    pet_maps : np.ndarray
        PET receptor maps (maps x parcels)
        
    Returns
    -------
    np.ndarray
        Estimated time series (timepoints x maps)
    """
    logging.info('Running Stage 1: PET maps as spatial regressors')
    
    # For each timepoint, regress fMRI against PET maps across parcels
    
    n_timepoints, n_parcels = fmri_data.shape
    n_maps = pet_maps.shape[0]
    
    # Demean PET maps across parcels
    scaler_pet = StandardScaler(with_mean=True, with_std=False)
    pet_centered = scaler_pet.fit_transform(pet_maps.T).T  # (maps x parcels)
    
    # Initialize output
    timeseries = np.zeros((n_timepoints, n_maps))
    
    # For each timepoint, fit: fMRI[t, :] = PET.T @ beta[t, :]
    for t in range(n_timepoints):
        # Get BOLD at this timepoint across all parcels
        y = fmri_data[t, :]  # (parcels,)
        
        # Demean this timepoint
        y_centered = y - np.mean(y)
        
        # Fit regression: y = PET.T @ beta
        # X is (parcels x maps), y is (parcels,)
        model = LinearRegression(fit_intercept=False)
        model.fit(pet_centered.T, y_centered)
        
        # Store coefficients (maps,)
        timeseries[t, :] = model.coef_
    
    logging.info(f'Stage 1 complete: Estimated {n_maps} receptor time series')
    
    return timeseries


def run_stage2(stage1_timeseries, fmri_data):
    """
    Stage 2: Use stage 1 time series as temporal regressors.
    
    Parameters
    ----------
    stage1_timeseries : np.ndarray
        Time series from stage 1 (timepoints x maps)
    fmri_data : np.ndarray
        fMRI data (timepoints x parcels)
        
    Returns
    -------
    np.ndarray
        Spatial maps (maps x parcels)
    """
    logging.info('Running Stage 2: Time series as temporal regressors')
    
    # Now we use time as the dimension for regression
    x = stage1_timeseries  # timepoints x maps
    y = fmri_data          # timepoints x parcels
    
    # Demean y (center only, no normalization) and normalize x to unit std
    scaler_y = StandardScaler(with_mean=True, with_std=False)
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    
    y_normalized = scaler_y.fit_transform(y)
    x_normalized = scaler_x.fit_transform(x)
    
    # Fit regression: y = x * beta + error
    # x is (timepoints x maps), y is (timepoints x parcels)
    model = LinearRegression(fit_intercept=False)  # Already centered
    model.fit(x_normalized, y_normalized)
    
    # model.coef_ has shape (parcels, maps) 
    beta = model.coef_.T  # Transpose to get (maps x parcels)
    
    logging.info(f'Stage 2 complete: Estimated spatial maps for {beta.shape[0]} PET maps')
    
    return beta


def save_outputs(out_subj_dir, stage1_timeseries, stage2_maps, pet_names):
    """
    Save all outputs for a subject.
    
    Parameters
    ----------
    out_subj_dir : Path
        Subject output directory
    stage1_timeseries : np.ndarray
        Time series from stage 1 (timepoints x maps)
    stage2_maps : np.ndarray
        Spatial maps from stage 2 (maps x parcels)
    pet_names : list of str
        Names of PET maps
    """
    out_subj_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stage 1 time series
    stage1_file = out_subj_dir / 'stage1_timeseries.csv'
    pd.DataFrame(
        stage1_timeseries,
        columns=pet_names
    ).to_csv(stage1_file, index=False)
    logging.info(f'Saved stage 1 time series: {stage1_file}')
    
    # Save combined stage 2 maps
    all_maps_file = out_subj_dir / 'all_petmaps.csv'
    pd.DataFrame(
        stage2_maps,
        index=pet_names
    ).to_csv(all_maps_file, index=True, header=False)
    logging.info(f'Saved combined PET maps: {all_maps_file}')
    
    # Save individual PET map files
    for i, pet_name in enumerate(pet_names):
        pet_file = out_subj_dir / f'{pet_name}.csv'
        pd.DataFrame(
            stage2_maps[i:i+1, :]
        ).to_csv(pet_file, index=False, header=False)
        logging.info(f'Saved individual map: {pet_file}')


def process_subject(fmri_file, pet_maps, pet_names, out_dir, force):
    """
    Process a single subject through REACT pipeline.
    
    Parameters
    ----------
    fmri_file : str
        Path to subject's fMRI data file
    pet_maps : np.ndarray
        PET receptor maps (maps x parcels)
    pet_names : list of str
        Names of PET maps
    out_dir : Path
        Base output directory
    force : bool
        Whether to overwrite existing outputs
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    subj_name = get_subject_name(fmri_file)
    logging.info(f'\n{"="*60}')
    logging.info(f'Processing subject: {subj_name}')
    logging.info(f'{"="*60}')
    
    # Check output
    out_subj_dir = out_dir / subj_name
    if not check_output_exists(out_subj_dir, force):
        return False
    
    try:
        # Load fMRI data
        logging.info(f'Loading fMRI data from: {fmri_file}')
        fmri_data = load_data_file(fmri_file)
        logging.info(f'fMRI data shape: {fmri_data.shape} (timepoints x parcels)')
        
        # Validate dimensions match
        n_timepoints, n_parcels_fmri = fmri_data.shape
        n_maps, n_parcels_pet = pet_maps.shape
        
        if n_parcels_fmri != n_parcels_pet:
            raise ValueError(
                f'Number of parcels mismatch: fMRI has {n_parcels_fmri}, '
                f'PET has {n_parcels_pet}'
            )
        
        # Validate fMRI data
        validate_parcel_data(fmri_data, f'fMRI data ({subj_name})', check_axis=0)
        
        # Run Stage 1
        stage1_timeseries = run_stage1(fmri_data, pet_maps)
        
        # Run Stage 2
        stage2_maps = run_stage2(stage1_timeseries, fmri_data)
        
        # Save outputs
        save_outputs(out_subj_dir, stage1_timeseries, stage2_maps, pet_names)
        
        logging.info(f'âœ“ Successfully completed processing for {subj_name}')
        return True
        
    except Exception as e:
        logging.error(f'âœ— Error processing {subj_name}: {str(e)}')
        return False


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info(f'Parcellated REACT v{__version__}')
    logging.info(f'{"="*60}')
    
    try:
        # Load inputs
        logging.info('Loading inputs...')
        fmri_files = load_fmri_list(args.fmri_list)
        
        logging.info(f'Loading PET maps from: {args.pet_maps}')
        pet_maps_raw = load_data_file(args.pet_maps)
        logging.info(f'PET maps shape (raw): {pet_maps_raw.shape} (maps x parcels)')
        
        # Scale PET maps to 0-1 (recommended in REACT paper)
        logging.info('Scaling PET maps to 0-1 range...')
        pet_maps = scale_pet_maps(pet_maps_raw)
        logging.info('PET maps scaled successfully')
        
        n_maps, n_parcels = pet_maps.shape
        
        # Validate PET data
        validate_parcel_data(pet_maps, 'PET maps', check_axis=1)
        
        # Load/generate PET names
        pet_names = load_pet_names(args.pet_names, n_maps, args.pet_maps)
        logging.info(f'PET map names: {pet_names}')
        
        # Create output directory
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Output directory: {out_dir}')
        
        # Process each subject
        n_subjects = len(fmri_files)
        n_success = 0
        
        for i, fmri_file in enumerate(fmri_files, 1):
            logging.info(f'\nSubject {i}/{n_subjects}')
            success = process_subject(fmri_file, pet_maps, pet_names, out_dir, args.force)
            if success:
                n_success += 1
        
        # Summary
        logging.info(f'\n{"="*60}')
        logging.info(f'SUMMARY')
        logging.info(f'{"="*60}')
        logging.info(f'Total subjects: {n_subjects}')
        logging.info(f'Successfully processed: {n_success}')
        logging.info(f'Failed: {n_subjects - n_success}')
        
        if n_success == n_subjects:
            logging.info('âœ“ All subjects processed successfully')
            return 0
        elif n_success == 0:
            logging.error('âœ— All subjects failed')
            return 1
        else:
            logging.warning('âš  Some subjects failed')
            return 1
            
    except Exception as e:
        logging.error(f'Fatal error: {str(e)}')
        return 1


if __name__ == '__main__':
    sys.exit(main())