#!/usr/bin/env python
"""
Test script for Parcellated REACT toolbox.

This script will:
1. Generate synthetic test data
2. Run the REACT analysis
3. Generate the QC report
4. Verify all outputs exist and are valid

Run from the toolbox directory:
    python test_toolbox.py
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


class Colors:
    """Terminal colors for pretty output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required_core = ['numpy', 'pandas', 'sklearn']
    required_qc = ['matplotlib', 'seaborn']
    
    all_good = True
    
    # Check core dependencies
    print_info("Checking core dependencies...")
    for package in required_core:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            all_good = False
    
    # Check QC dependencies
    print_info("\nChecking QC dependencies...")
    for package in required_qc:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_warning(f"{package} is NOT installed (needed for QC reports)")
    
    return all_good


def check_scripts_exist():
    """Check if all required scripts exist."""
    print_header("Checking Script Files")
    
    required_scripts = [
        'utils.py',
        'react_parcellated.py',
        'react_qc_report.py',
        'generate_example_data.py'
    ]
    
    all_exist = True
    
    for script in required_scripts:
        if Path(script).exists():
            print_success(f"{script} found")
        else:
            print_error(f"{script} NOT found")
            all_exist = False
    
    return all_exist


def run_command(cmd, description):
    """
    Run a command and report success/failure.
    
    Parameters
    ----------
    cmd : list
        Command and arguments as list
    description : str
        Description of what the command does
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print_info(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print_success(f"{description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} - FAILED")
        print(f"  Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print_error(f"{description} - Script not found")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        print_success(f"{description} exists ({size:,} bytes)")
        return True
    else:
        print_error(f"{description} NOT found at {filepath}")
        return False


def verify_outputs():
    """Verify all expected output files exist."""
    print_header("Verifying Outputs")
    
    all_good = True
    
    # Check example_results directory
    results_dir = Path('example_results')
    if not results_dir.exists():
        print_error("example_results directory not found")
        return False
    
    # Check for subject directories
    print_info("Checking subject directories...")
    subject_dirs = list(results_dir.glob('subject_*_fmri'))
    if len(subject_dirs) == 0:
        print_error("No subject directories found")
        all_good = False
    else:
        print_success(f"Found {len(subject_dirs)} subject directories")
        
        # Check first subject's outputs
        first_subj = subject_dirs[0]
        print_info(f"Checking outputs in {first_subj.name}...")
        
        required_files = [
            'stage1_timeseries.csv',
            'all_petmaps.csv'
        ]
        
        for fname in required_files:
            fpath = first_subj / fname
            if not check_file_exists(fpath, f"  {fname}"):
                all_good = False
    
    # Check QC outputs
    print_info("\nChecking QC outputs...")
    qc_files = [
        'react_qc_report.html',
        'summary_statistics.csv',
        'vif_values.csv'
    ]
    
    for fname in qc_files:
        fpath = results_dir / fname
        if not check_file_exists(fpath, f"  {fname}"):
            all_good = False
    
    # Check QC plots directory
    qc_plots_dir = results_dir / 'qc_plots'
    if qc_plots_dir.exists():
        n_plots = len(list(qc_plots_dir.glob('*.png')))
        print_success(f"  qc_plots/ directory exists with {n_plots} plots")
    else:
        print_error("  qc_plots/ directory not found")
        all_good = False
    
    return all_good


def validate_outputs():
    """Validate content of output files."""
    print_header("Validating Output Content")
    
    import pandas as pd
    import numpy as np
    
    all_good = True
    
    # Load and validate stage1 timeseries
    print_info("Validating stage1_timeseries.csv...")
    try:
        ts_file = Path('example_results') / 'subject_001_fmri' / 'stage1_timeseries.csv'
        ts_df = pd.read_csv(ts_file)
        
        if ts_df.shape[0] == 0:
            print_error("  Timeseries is empty")
            all_good = False
        elif ts_df.shape[1] == 0:
            print_error("  No receptors in timeseries")
            all_good = False
        elif np.any(np.isnan(ts_df.values)):
            print_error("  Timeseries contains NaN values")
            all_good = False
        else:
            print_success(f"  Shape: {ts_df.shape} (timepoints × receptors)")
            print_success(f"  Receptors: {list(ts_df.columns)}")
    except Exception as e:
        print_error(f"  Failed to load/validate: {e}")
        all_good = False
    
    # Load and validate spatial maps
    print_info("\nValidating all_petmaps.csv...")
    try:
        maps_file = Path('example_results') / 'subject_001_fmri' / 'all_petmaps.csv'
        maps_df = pd.read_csv(maps_file, header=None, index_col=0)
        
        if maps_df.shape[0] == 0:
            print_error("  Spatial maps empty")
            all_good = False
        elif maps_df.shape[1] == 0:
            print_error("  No parcels in spatial maps")
            all_good = False
        elif np.any(np.isnan(maps_df.values)):
            print_error("  Spatial maps contain NaN values")
            all_good = False
        else:
            print_success(f"  Shape: {maps_df.shape} (receptors × parcels)")
            print_success(f"  Receptors: {list(maps_df.index)}")
    except Exception as e:
        print_error(f"  Failed to load/validate: {e}")
        all_good = False
    
    # Load and validate VIF values
    print_info("\nValidating vif_values.csv...")
    try:
        vif_file = Path('example_results') / 'vif_values.csv'
        vif_df = pd.read_csv(vif_file)
        
        if 'receptor' not in vif_df.columns or 'vif' not in vif_df.columns:
            print_error("  VIF file missing required columns")
            all_good = False
        elif np.any(vif_df['vif'] < 1):
            print_error("  VIF values should be >= 1")
            all_good = False
        else:
            print_success(f"  Found {len(vif_df)} VIF values")
            for _, row in vif_df.iterrows():
                status = "LOW" if row['vif'] < 5 else "MODERATE" if row['vif'] < 10 else "HIGH"
                print_success(f"    {row['receptor']}: VIF = {row['vif']:.2f} ({status})")
    except Exception as e:
        print_error(f"  Failed to load/validate: {e}")
        all_good = False
    
    return all_good



def main():
    """Main test function."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║          Parcellated REACT Toolbox - Test Suite                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # Track overall success
    all_tests_passed = True
    
    # 1. Check dependencies
    if not check_dependencies():
        print_error("\nMissing core dependencies! Install with: pip install -r requirements.txt")
        return 1
    
    # 2. Check scripts exist
    if not check_scripts_exist():
        print_error("\nMissing required scripts! Are you in the toolbox directory?")
        return 1
    
    # 3. Generate example data
    print_header("Step 1: Generate Example Data")
    if not run_command(
        [sys.executable, 'generate_example_data.py'],
        "Generating synthetic data"
    ):
        all_tests_passed = False
        print_error("\nFailed to generate example data")
        return 1
    
    # 4. Run REACT analysis
    print_header("Step 2: Run REACT Analysis")
    if not run_command(
        [
            sys.executable, 'react_parcellated.py',
            '--fmri_list', 'example_data/subject_list.txt',
            '--pet_maps', 'example_data/pet_receptors.csv',
            '--pet_names', 'example_data/receptor_names.txt',
            '--out_dir', 'example_results',
            '--verbose'
        ],
        "Running REACT analysis"
    ):
        all_tests_passed = False
        print_error("\nFailed to run REACT analysis")
        return 1
    
    # 5. Generate QC report
    print_header("Step 3: Generate QC Report")
    if not run_command(
        [
            sys.executable, 'react_qc_report.py',
            '--results_dir', 'example_results',
            '--pet_maps', 'example_data/pet_receptors.csv',
            '--verbose'
        ],
        "Generating QC report"
    ):
        all_tests_passed = False
        print_error("\nFailed to generate QC report")
        return 1
    
    # 6. Verify outputs exist
    if not verify_outputs():
        all_tests_passed = False
        print_error("\nSome expected outputs are missing")
    
    # 7. Validate output content
    if not validate_outputs():
        all_tests_passed = False
        print_error("\nSome outputs have invalid content")
    
    # 8. Final summary
    print_header("Test Summary")
    
    if all_tests_passed:
        print(f"{Colors.OKGREEN}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║                  ALL TESTS PASSED! ✓                              ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}\n")
        
        print_info("You can view the QC report:")
        print(f"  open example_results/react_qc_report.html\n")
        

        
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║                  SOME TESTS FAILED ✗                              ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}\n")
        
        print_error("Please check the error messages above and fix the issues.")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())