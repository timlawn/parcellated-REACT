# Parcellated REACT

A parcellation-based implementation of **REACT** (Receptor-Enriched Analysis of functional Connectivity by Targets) for analyzing fMRI data using PET receptor maps as spatial priors.

## Overview

This toolbox implements the dual regression approach from [Dipasquale et al. (2019)](https://doi.org/10.1016/j.neuroimage.2019.04.007) for parcellated data. 

This provides the two main benefits of (1) fitting more seamlessly into parcellated data analysis pipeline and (2) utilising PET data from multiple sources with varying spatial resolutions at a coarser granularity where these differences matter less. 

### Key Features

-  **Parcellation-based**: Works with pre-extracted time series from brain atlases
-  **Automatic PET scaling**: Scales PET maps to 0-1 range 
-  **Modular design**: Separate tools for analysis and quality control
-  **Comprehensive QC**: HTML reports with VIF, correlations, and visualizations

## Installation

**Requirements**: Python 3.7 or higher

### Dependencies

**Core analysis** (`react_parcellated.py`):
```bash
pip install numpy pandas scikit-learn
```

**QC reports** (`react_qc_report.py`):
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Or install all dependencies:**
```bash
pip install -r requirements.txt
```

### Setup

Clone or download this repository:
```bash
git clone https://github.com/yourusername/parcellated-react.git
cd parcellated-react
```

Make scripts executable:
```bash
chmod +x react_parcellated.py react_qc_report.py
```

## Quick Start

### 1. Prepare Your Data

**fMRI data**: CSV/TXT files with rows=timepoints, columns=parcels (one file per subject)
```
timepoint1: 0.23, 0.45, 0.12, ...
timepoint2: 0.34, 0.56, 0.23, ...
...
```

**PET data**: Single CSV/TXT file with rows=receptor maps, columns=parcels
```
5HT1a:  0.42, 0.38, 0.45, ...
5HT2a:  0.35, 0.40, 0.33, ...
DAT:    0.28, 0.31, 0.29, ...
```

### 2. Run REACT Analysis

Create a text file listing your fMRI files:
```bash
# subjects.txt
/path/to/subject001_fmri.csv
/path/to/subject002_fmri.csv
/path/to/subject003_fmri.csv
```

Run the analysis:
```bash
python react_parcellated.py \
  --fmri_list subjects.txt \
  --pet_maps receptors.csv \
  --out_dir ./results \
  --verbose
```

### 3. Generate QC Report

```bash
python react_qc_report.py \
  --results_dir ./results \
  --pet_maps receptors.csv \
  --verbose
```

Open `./results/react_qc_report.html` in your browser!

## Detailed Usage

### react_parcellated.py

Core REACT analysis tool.

#### Arguments

- `--fmri_list` (required): Path to text file with fMRI file paths (one per line), OR comma-separated file paths
- `--pet_maps` (required): Path to PET receptor map file (rows=maps, columns=parcels)
- `--out_dir` (required): Output directory for results
- `--pet_names` (optional): Text file with custom PET map names (one per line)
- `--force`: Overwrite existing outputs
- `-v, --verbose`: Enable detailed logging

#### Output Structure

```
results/
├── subject001/
│   ├── stage1_timeseries.csv      # Receptor-specific time series
│   ├── all_petmaps.csv            # Combined spatial maps (all receptors)
│   ├── 5HT1a_map.csv              # Individual receptor map
│   ├── 5HT2a_map.csv
│   └── ...
├── subject002/
│   └── ...
└── ...
```

**File formats:**
- `stage1_timeseries.csv`: timepoints Ã— receptors
- `all_petmaps.csv`: receptors Ã— parcels (with receptor names as row indices)
- `{receptor}_map.csv`: 1 Ã— parcels (spatial map for single receptor)

#### Examples

**Basic usage:**
```bash
python react_parcellated.py \
  --fmri_list subjects.txt \
  --pet_maps receptors.csv \
  --out_dir ./results
```

**With custom PET names:**
```bash
# Create receptor_names.txt:
echo -e "Serotonin_1A\nSerotonin_2A\nDopamine_Transporter" > receptor_names.txt

python react_parcellated.py \
  --fmri_list subjects.txt \
  --pet_maps receptors.csv \
  --pet_names receptor_names.txt \
  --out_dir ./results
```

**Using comma-separated file list:**
```bash
python react_parcellated.py \
  --fmri_list "sub1.csv,sub2.csv,sub3.csv" \
  --pet_maps receptors.csv \
  --out_dir ./results
```

### react_qc_report.py

Generate comprehensive quality control report.

#### Arguments

- `--results_dir` (required): Directory with REACT results from `react_parcellated.py`
- `--pet_maps` (required): Original PET maps file used in analysis
- `--out_file` (optional): Output HTML filename (default: `react_qc_report.html`)
- `-v, --verbose`: Enable detailed logging

#### Output Files

```
results/
├── react_qc_report.html           # Main HTML report (open in browser!)
├── summary_statistics.csv         # Per-subject per-receptor statistics
├── vif_values.csv                 # Variance Inflation Factors
└── qc_plots/                      # All plots embedded in HTML
    ├── vif_values.png
    ├── pet_correlations.png
    ├── spatial_correlations.png
    ├── timeseries_correlations.png
    ├── timeseries.png
    └── spatial_network_{receptor}.png
```

#### QC Report Contents

1. **Analysis Overview**: Basic info about subjects, receptors, parcels
2. **VIF Analysis**: Multicollinearity assessment for PET maps
3. **Correlation Analyses**: PET maps, spatial maps, and time series
4. **Time Series Plots**: Receptor-specific time series across subjects
5. **Spatial Networks**: Mean spatial maps per receptor
6. **Summary Statistics**: Group-level statistics across subjects

#### Example

```bash
python react_qc_report.py \
  --results_dir ./results \
  --pet_maps receptors.csv \
  --verbose
```

## Using Hansen PET Data

You can use pre-parcellated PET maps from the [Hansen Receptors dataset](https://github.com/netneurolab/hansen_receptors):

**Single receptor analysis:**
```bash
# Download a PET map (e.g., Schaefer 400 parcellation)
wget https://raw.githubusercontent.com/netneurolab/hansen_receptors/main/data/PET_parcellated/scale400/5HT1a_cumi_hc8_beliveau.csv

# Use directly in analysis
python react_parcellated.py \
  --fmri_list subjects.txt \
  --pet_maps 5HT1a_cumi_hc8_beliveau.csv \
  --out_dir ./results
```

**Multiple receptor analysis:**
For multiple receptors, download individual files and combine them into a single CSV where each row is a receptor map and each column is a parcel. For example, if you have `5HT1a.csv`, `5HT2a.csv`, and `DAT.csv`, stack them as rows in `receptors.csv`.

**Available parcellations**: scale033, scale060, scale100, scale125, scale200, scale400

## Methodology


### Dual Regression Overview

REACT uses a two-stage dual regression approach:

![REACT Stages](https://timlawn.github.io/images/react-stages.png)

**Stage 1**: PET maps as spatial regressors
- Input: PET maps (receptors × parcels) and fMRI data (timepoints × parcels)
- Output: Receptor-specific time series (timepoints × receptors)
- For each timepoint, regress BOLD activity across parcels against PET receptor densities

**Stage 2**: Time series as temporal regressors
- Input: Stage 1 time series and fMRI data
- Output: Receptor-enriched spatial maps (receptors × parcels)
- For each parcel, regress BOLD time series against receptor-specific time series

### Preprocessing

1. **PET scaling**: All PET maps scaled to [0,1] range
2. **Centering**: Both stages demean data (standard dual regression)
3. **Normalization**: Stage 2 normalizes design matrix to unit variance

### Interpretation

The output spatial maps represent how strongly each parcel's activity relates to the receptor-enriched time series. Positive values indicate positive coupling, negative values indicate anti-correlation. For more details see this [blog post](https://timlawn.github.io/posts/2025/01/react-guide/) and [review paper](https://pubmed.ncbi.nlm.nih.gov/37086932/).

## Variance Inflation Factor (VIF)

VIF measures multicollinearity between PET maps. High collinearity can affect REACT results.

**Interpreting VIF:**
- **VIF < 5**: Low multicollinearity (good)
- **VIF 5-10**: Moderate multicollinearity (problematic)
- **VIF > 10**: High multicollinearity (abort!)

**What to do if VIF is high:**
- Consider using fewer, less correlated PET maps
- Use PCA to create orthogonal components (at risk of making interpretation even more complex...)

## Tips and Best Practices

### Data Preparation

1. **Parcellation consistency**: Ensure fMRI and PET use the same parcellation
2. **Check for NaNs/zeros**: Script validates but good to check beforehand
4. **Multiple runs**: Consider averaging resulting networks across runs

### PET Maps

1. **Source quality**: Use high quality group average receptor maps 
2. **Number of maps**: Start with fewer maps; more maps = more multicollinearity
3. **Biological relevance**: Choose receptors relevant to your hypothesis (REACT is not well suited to purely data driven analyses)

### Downstream Analyses

The output spatial maps can be used for:
- Group comparisons (patients vs. controls)
- Drug effects (pre/post administration)
- Correlation with behavioral/clinical measures
- Prediction modeling or classification
- Endless other applications!
  
## Troubleshooting

### "Dimension mismatch" error
- Check that fMRI and PET have same number of parcels
- Verify data orientation (rows=timepoints for fMRI, rows=receptors for PET)

### "All-zero parcels" error
- Some parcels may be outside brain or have no data
- Deal with these before analysis

### High VIF values
- Consider using subset of less-correlated PET maps
- Try PCA on PET maps to create orthogonal components

### NaN in outputs
- Check input data for NaNs

## Citation

If you use this toolbox, please cite the original REACT paper:

> Dipasquale, O., Selvaggi, P., Veronese, M., Gabay, A. S., Turkheimer, F., & Mehta, M. A. (2019). Receptor-Enriched Analysis of functional connectivity by targets (REACT): A novel, multimodal analytical approach informed by PET to study the pharmacodynamic response of the brain under MDMA. *NeuroImage*, https://doi.org/10.1016/j.neuroimage.2019.04.007

If using Hansen PET data, also cite the main paper alongside the individual original PET map papers:

> Hansen, J. Y., et al. (2022). Mapping neurotransmitter systems to the structural and functional organization of the human neocortex. *Nature Neuroscience*, https://www.nature.com/articles/s41593-022-01186-3

For more information about these sorts of analyses, including their applications, limitations, and interpretation: 

> Lawn, T., et al. (2023). From neurotransmitters to networks: Transcending organizational hierarchies with molecular-informed functional imaging. *Neuroscience and Biobehavioral Reviews*, https://pubmed.ncbi.nlm.nih.gov/37086932/

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contact

tlawn1@mgh.harvard.edu

## Acknowledgments

- Original REACT toolbox: https://github.com/ottaviadipasquale/react-fmri
- Hansen receptor data: https://github.com/netneurolab/hansen_receptors
