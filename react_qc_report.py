#!/usr/bin/env python
"""
Quality Control Report for Parcellated REACT Analysis

Generates comprehensive QC report including:
- Variance Inflation Factor (VIF) for PET maps
- Correlation matrices
- Time series visualizations
- Spatial network plots
- HTML report with embedded figures

Run this after completing REACT analysis with react_parcellated.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import load_data_file, scale_pet_maps, load_react_results


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
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing REACT analysis results (output from react_parcellated.py)'
    )
    
    parser.add_argument(
        '--pet_maps',
        type=str,
        required=True,
        help='Path to original PET receptor map file used in analysis'
    )
    
    parser.add_argument(
        '--out_file',
        type=str,
        default=None,
        help='Output HTML filename (default: react_qc_report.html in results_dir)'
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


def compute_vif(pet_maps):
    """
    Compute Variance Inflation Factor for PET maps.
    
    VIF measures multicollinearity between PET maps. Values > 10 indicate
    extremely high collinearity. Values > 5 warrant careful consideration.
    
    Parameters
    ----------
    pet_maps : np.ndarray
        PET receptor maps (maps x parcels)
        
    Returns
    -------
    np.ndarray
        VIF values for each PET map
    """
    n_maps = pet_maps.shape[0]
    vif_values = np.zeros(n_maps)
    
    # Transpose for sklearn (parcels x maps)
    X = pet_maps.T
    
    for i in range(n_maps):
        # Use other maps to predict current map
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        
        if X_other.shape[1] > 0:
            model = LinearRegression()
            model.fit(X_other, y)
            y_pred = model.predict(X_other)
            r2 = r2_score(y, y_pred)
            # VIF = 1 / (1 - RÂ²)
            vif_values[i] = 1 / (1 - r2) if r2 > 0.9999 else np.inf
        else:
            vif_values[i] = 1.0
    
    return vif_values


def compute_summary_statistics(results):
    """
    Compute summary statistics across subjects.
    
    Parameters
    ----------
    results : dict
        REACT results from load_react_results()
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per receptor per subject
    """
    summary_data = []
    
    for subj in results['subjects']:
        spatial_map = results['spatial_maps'][subj]  # (maps x parcels)
        timeseries = results['stage1_timeseries'][subj]  # (timepoints x maps)
        
        for i, pet_name in enumerate(results['pet_names']):
            summary_data.append({
                'subject': subj,
                'receptor': pet_name,
                'spatial_mean': np.mean(spatial_map[i, :]),
                'spatial_std': np.std(spatial_map[i, :]),
                'spatial_max': np.max(spatial_map[i, :]),
                'spatial_min': np.min(spatial_map[i, :]),
                'timeseries_mean': np.mean(timeseries[:, i]),
                'timeseries_std': np.std(timeseries[:, i]),
                'timeseries_max': np.max(timeseries[:, i]),
                'timeseries_min': np.min(timeseries[:, i])
            })
    
    return pd.DataFrame(summary_data)


def plot_correlation_matrix(data, labels, title, output_file):
    """
    Create correlation matrix heatmap.
    
    Parameters
    ----------
    data : np.ndarray
        Data to correlate (features x samples)
    labels : list of str
        Feature labels
    title : str
        Plot title
    output_file : Path
        Output file path
    """
    corr_matrix = np.corrcoef(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add correlation values as text
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=9,
                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_timeseries(results, output_file):
    """
    Plot receptor time series for all subjects (or random sample of 10 if >10 subjects).
    
    Parameters
    ----------
    results : dict
        REACT results
    output_file : Path
        Output file path
    """
    n_receptors = len(results['pet_names'])
    
    # If more than 10 subjects, randomly sample 10 to reduce clutter
    all_subjects = results['subjects']
    if len(all_subjects) > 10:
        np.random.seed(42)  # For reproducibility
        subjects_to_plot = list(np.random.choice(all_subjects, size=10, replace=False))
    else:
        subjects_to_plot = all_subjects
    
    fig, axes = plt.subplots(n_receptors, 1, figsize=(14, 3 * n_receptors))
    
    if n_receptors == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects_to_plot)))
    
    for i, pet_name in enumerate(results['pet_names']):
        ax = axes[i]
        
        for j, subj in enumerate(subjects_to_plot):
            timeseries = results['stage1_timeseries'][subj][:, i]
            ax.plot(timeseries, alpha=0.7, linewidth=1.5, 
                   color=colors[j], label=subj if i == 0 else '')
        
        ax.set_title(f'{pet_name} Time Series', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Point', fontsize=10)
        ax.set_ylabel('Signal Intensity', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0 and len(subjects_to_plot) <= 10:
            ax.legend(fontsize=8, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_spatial_networks(results, output_dir):
    """
    Plot spatial networks for each receptor using boxplots to show variability across subjects.
    
    Parameters
    ----------
    results : dict
        REACT results
    output_dir : Path
        Output directory for plots
        
    Returns
    -------
    list of Path
        List of output file paths
    """
    output_files = []
    n_parcels = results['all_spatial_maps'].shape[2]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results['pet_names'])))
    
    for i, pet_name in enumerate(results['pet_names']):
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Get data for all subjects for this receptor: (subjects x parcels)
        receptor_data = results['all_spatial_maps'][:, i, :]
        
        # Prepare data for boxplot - each parcel gets a boxplot
        bp = ax.boxplot([receptor_data[:, p] for p in range(n_parcels)],
                        positions=range(n_parcels),
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,  # Don't show outliers to reduce clutter
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5),
                        whiskerprops=dict(color='black', linewidth=0.5),
                        capprops=dict(color='black', linewidth=0.5))
        
        ax.set_title(f'{pet_name} Spatial Network (Variability Across Space and Subjects)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Parcel', fontsize=11)
        ax.set_ylabel('Connectivity Strength', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set x-axis ticks to show only min, Q1, Q2, Q3, max
        tick_positions = [0, n_parcels // 4, n_parcels // 2, 3 * n_parcels // 4, n_parcels - 1]
        tick_labels = [str(pos) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        # Add statistics box
        mean_across_all = np.mean(receptor_data)
        std_across_all = np.std(receptor_data)
        stats_text = (f'Overall Mean: {mean_across_all:.3f}\n'
                     f'Overall Std: {std_across_all:.3f}\n'
                     f'N subjects: {receptor_data.shape[0]}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, 
                        edgecolor='black'))
        
        plt.tight_layout()
        output_file = output_dir / f'spatial_network_{pet_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        output_files.append(output_file)
    
    return output_files



def plot_vif_bar(vif_values, pet_names, output_file):
    """
    Create bar plot of VIF values.
    
    Parameters
    ----------
    vif_values : np.ndarray
        VIF values
    pet_names : list of str
        PET map names
    output_file : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' 
              for v in vif_values]
    
    bars = ax.bar(range(len(pet_names)), vif_values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(pet_names)))
    ax.set_xticklabels(pet_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('VIF Value', fontsize=12)
    ax.set_title('Variance Inflation Factor (VIF) for PET Maps', 
                fontsize=14, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, 
              label='Moderate threshold (VIF=5)')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, 
              label='High threshold (VIF=10)')
    
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, vif_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_html_report(results_dir, pet_names, vif_values, summary_stats, 
                       plot_files, output_file, n_parcels):
    """
    Create comprehensive HTML report.
    
    Parameters
    ----------
    results_dir : Path
        Results directory
    pet_names : list of str
        PET map names
    vif_values : np.ndarray
        VIF values
    summary_stats : pd.DataFrame
        Summary statistics
    plot_files : dict
        Dictionary of plot file paths
    output_file : Path
        Output HTML file path
    n_parcels : int
        Number of parcels
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Parcellated REACT QC Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 32px;
            }}
            .header p {{
                margin: 5px 0;
                font-size: 16px;
                opacity: 0.9;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background-color: #fafafa;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .section h2 {{
                color: #667eea;
                margin-top: 0;
                font-size: 24px;
            }}
            .metric {{
                background-color: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 3px solid #764ba2;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .grid-3 {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .image-container {{
                text-align: center;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }}
            .image-container h3 {{
                margin-top: 0;
                color: #667eea;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .vif-low {{ color: #27ae60; font-weight: bold; }}
            .vif-moderate {{ color: #f39c12; font-weight: bold; }}
            .vif-high {{ color: #e74c3c; font-weight: bold; }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                text-align: center;
                color: #999;
                font-size: 14px;
            }}
            .warning-box {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}
            .info-box {{
                background-color: #d1ecf1;
                border: 1px solid #17a2b8;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Parcellated REACT Quality Control Report</h1>
                <p>Receptor-Enriched Analysis of functional Connectivity by Targets</p>
                <p>Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Results directory: {results_dir}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Overview</h2>
                <div class="metric">
                    <strong>Number of subjects:</strong> {len(set(summary_stats['subject']))}
                </div>
                <div class="metric">
                    <strong>Number of PET maps:</strong> {len(pet_names)}
                </div>
                <div class="metric">
                    <strong>PET maps analyzed:</strong> {', '.join(pet_names)}
                </div>
                <div class="metric">
                    <strong>Number of parcels:</strong> {n_parcels}
                </div>
            </div>
            
            <div class="section">
                <h2>Multicollinearity Analysis (VIF)</h2>
                <div class="info-box">
                    <strong>Variance Inflation Factor (VIF)</strong> measures multicollinearity between PET maps.
                    <ul>
                        <li><span class="vif-low">VIF &lt; 5</span>: Low multicollinearity (good!)</li>
                        <li><span class="vif-moderate">VIF 5-10</span>: Problematic multicollinearity (caution!)</li>
                        <li><span class="vif-high">VIF &gt; 10</span>: Extreme multicollinearity (abort!)</li>
                    </ul>
                </div>
                
                <table>
                    <tr>
                        <th>PET Map</th>
                        <th>VIF Value</th>
                        <th>Interpretation</th>
                    </tr>
    """
    
    for pet_name, vif in zip(pet_names, vif_values):
        if vif < 5:
            vif_class = 'vif-low'
            interpretation = 'Low - Good!'
        elif vif < 10:
            vif_class = 'vif-moderate'
            interpretation = 'Problematic - Caution!'
        else:
            vif_class = 'vif-high'
            interpretation = 'Extreme - Abort!'
        
        html_content += f"""
                    <tr>
                        <td>{pet_name}</td>
                        <td class="{vif_class}">{vif:.2f}</td>
                        <td class="{vif_class}">{interpretation}</td>
                    </tr>
        """
    
    html_content += f"""
                </table>
                
                <div class="image-container">
                    <img src="qc_plots/{plot_files['vif'].name}">
                </div>
            </div>
            
            <div class="section">
                <h2>Correlation Analyses</h2>
                <div class="grid-3">
                    <div class="image-container">
                        <h3>PET Maps</h3>
                        <img src="qc_plots/{plot_files['pet_corr'].name}">
                    </div>
                    <div class="image-container">
                        <h3>REACT Network</h3>
                        <img src="qc_plots/{plot_files['spatial_corr'].name}">
                    </div>
                    <div class="image-container">
                        <h3>Time Series</h3>
                        <img src="qc_plots/{plot_files['timeseries_corr'].name}">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Receptor Time Series</h2>
                <div class="image-container">
                    <img src="qc_plots/{plot_files['timeseries'].name}">
                </div>
            </div>
            
            <div class="section">
                <h2>Spatial Networks</h2>
                <p>Boxplots showing variability across subjects for each parcel:</p>
    """
    
    for i, pet_name in enumerate(pet_names):
        html_content += f"""
                    <div class="image-container">
                        <h3>{pet_name}</h3>
                        <img src="qc_plots/{plot_files['spatial_networks'][i].name}">
                    </div>
        """
    
    html_content += f"""
            </div>
            
            <div class="footer">
                <p>Generated by Parcellated REACT QC Report v{__version__}</p>
                <p>Reference: Dipasquale et al. (2019) NeuroImage 195:252-260</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info(f'Parcellated REACT QC Report v{__version__}')
    logging.info('='*60)
    
    try:
        results_dir = Path(args.results_dir)
        
        # Set output file
        if args.out_file is None:
            output_file = results_dir / 'react_qc_report.html'
        else:
            output_file = Path(args.out_file)
        
        # Create QC subdirectory for plots
        qc_dir = results_dir / 'qc_plots'
        qc_dir.mkdir(exist_ok=True)
        logging.info(f'QC plots directory: {qc_dir}')
        
        # Load PET maps
        logging.info('Loading PET maps...')
        pet_maps_raw = load_data_file(args.pet_maps)
        pet_maps = scale_pet_maps(pet_maps_raw)
        logging.info(f'PET maps shape: {pet_maps.shape} (maps x parcels)')
        
        # Load REACT results
        logging.info('Loading REACT results...')
        results = load_react_results(results_dir)
        logging.info(f"Found {len(results['subjects'])} subjects")
        logging.info(f"PET maps: {results['pet_names']}")
        
        # Compute VIF
        logging.info('Computing Variance Inflation Factors...')
        vif_values = compute_vif(pet_maps)
        for name, vif in zip(results['pet_names'], vif_values):
            status = 'LOW' if vif < 5 else 'MODERATE' if vif < 10 else 'HIGH'
            logging.info(f'  {name}: VIF = {vif:.2f} ({status})')
        
        # Save VIF values
        vif_df = pd.DataFrame({
            'receptor': results['pet_names'],
            'vif': vif_values
        })
        vif_df.to_csv(results_dir / 'vif_values.csv', index=False)
        logging.info(f"Saved VIF values to {results_dir / 'vif_values.csv'}")
        
        # Compute summary statistics
        logging.info('Computing summary statistics...')
        summary_stats = compute_summary_statistics(results)
        summary_stats.to_csv(results_dir / 'summary_statistics.csv', index=False)
        logging.info(f"Saved summary statistics to {results_dir / 'summary_statistics.csv'}")
        
        # Create visualizations
        logging.info('Creating visualizations...')
        
        plot_files = {}
        
        # VIF bar plot
        logging.info('  - VIF bar plot')
        plot_files['vif'] = qc_dir / 'vif_values.png'
        plot_vif_bar(vif_values, results['pet_names'], plot_files['vif'])
        
        # Correlation matrices
        logging.info('  - PET map correlations')
        plot_files['pet_corr'] = qc_dir / 'pet_correlations.png'
        plot_correlation_matrix(pet_maps, results['pet_names'], 
                              'PET Map Correlations', plot_files['pet_corr'])
        
        logging.info('  - Spatial map correlations')
        plot_files['spatial_corr'] = qc_dir / 'spatial_correlations.png'
        # Average spatial maps across subjects
        mean_spatial = np.mean(results['all_spatial_maps'], axis=0)
        plot_correlation_matrix(mean_spatial, results['pet_names'],
                              'Spatial Network Correlations (Mean)', 
                              plot_files['spatial_corr'])
        
        logging.info('  - Time series correlations')
        plot_files['timeseries_corr'] = qc_dir / 'timeseries_correlations.png'
        # Compute correlations within each subject, then average the correlation matrices
        all_corr_matrices = []
        for subj in results['subjects']:
            ts = results['stage1_timeseries'][subj].T  # (maps x timepoints)
            subj_corr = np.corrcoef(ts)
            all_corr_matrices.append(subj_corr)
        mean_corr = np.mean(all_corr_matrices, axis=0)
        
        # Create custom plot for mean correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(results['pet_names'])))
        ax.set_yticks(range(len(results['pet_names'])))
        ax.set_xticklabels(results['pet_names'], rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(results['pet_names'], fontsize=10)
        # Add correlation values as text
        for i in range(len(results['pet_names'])):
            for j in range(len(results['pet_names'])):
                text = ax.text(j, i, f'{mean_corr[i, j]:.2f}',
                              ha='center', va='center', fontsize=9,
                              color='white' if abs(mean_corr[i, j]) > 0.5 else 'black')
        ax.set_title('Time Series Correlations (Mean Across Subjects)', fontsize=14, fontweight='bold', pad=20)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=11)
        plt.tight_layout()
        plt.savefig(plot_files['timeseries_corr'], dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Time series plots
        logging.info('  - Time series plots')
        plot_files['timeseries'] = qc_dir / 'timeseries.png'
        plot_timeseries(results, plot_files['timeseries'])
        
        # Spatial network plots
        logging.info('  - Spatial network plots')
        plot_files['spatial_networks'] = plot_spatial_networks(results, qc_dir)
        
        # Create HTML report
        logging.info('Creating HTML report...')
        n_parcels = pet_maps.shape[1]
        create_html_report(results_dir, results['pet_names'], vif_values,
                         summary_stats, plot_files, output_file, n_parcels)
        
        logging.info('='*60)
        logging.info(f'âœ“ QC report complete!')
        logging.info(f'HTML report: {output_file}')
        logging.info(f'Open in browser to view comprehensive QC report')
        logging.info('='*60)
        
        return 0
        
    except Exception as e:
        logging.error(f'Fatal error: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())