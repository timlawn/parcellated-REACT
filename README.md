# Parcellated REACT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17643163.svg)](https://doi.org/10.5281/zenodo.17643163)

REACT (Receptor-Enriched Analysis of functional Connectivity by Targets;
[Dipasquale et al., 2019](https://doi.org/10.1016/j.neuroimage.2019.04.007))
for parcellated data: region-wise fMRI timeseries and PET maps in the same
parcellation.

The toolbox does not provide PET maps or parcellation code. It assumes you
supply timeseries and PET maps that are already parcellated with the same atlas,
in the same region order.

## Install

```bash
pip install git+https://github.com/timlawn/parcellated-REACT
```

or, from a local copy, `pip install /path/to/parcellated-REACT`. The only
dependencies are numpy and pandas.

## Inputs

| Input | Format |
|---|---|
| Timeseries | One file per scan, rows = timepoints, columns = regions. The scan ID is the filename stem. |
| PET maps | One file, rows = regions, columns = maps, **header = map names**. An optional `region` column is used as region labels (otherwise regions are numbered 1..N). |

Files ending in `.tsv` are read as tab-separated; anything else as
comma-separated.

Timeseries files may have a header row. If the first row contains any text,
it is treated as a header and **ignored**. A header of numeric region labels
(whole numbers in increasing order) is rejected, because it can't be told
apart from data; use text labels or no header. Empty cells, `n/a` and `NaN`
are read as missing values, so the region is excluded for that scan.

In the PET table, map names become output filenames, so they must be unique
(ignoring case). An unnamed index column, as written by pandas `to_csv()`
without `index=False`, is rejected.

**Regions are matched to PET maps by position only.** The only check is that
the number of regions is the same. Region labels in a timeseries header are
not compared with the PET table, so make sure both use the same atlas and the
same region order.


## Usage

```bash
parcellated-react --timeseries data/sub-*_timeseries.csv --pet pet.csv \
    --out results/univariate
```

| Options | |
|---|---|
| `--timeseries` / `--timeseries-list` | Timeseries files, or a text file listing them |
| `--pet` | PET maps file |
| `--mode` | `univariate` (default): an independent model per map. `multivariate`: all maps fitted jointly in both stages. |
| `--out` | Output directory |
| `--scale-pet` | Min-max scale each PET map to [0, 1] (default: off) |
| `--data-norm` | Normalise each region's timeseries to unit SD in stage 2 (default: off; see Method) |
| `--force` | Overwrite existing outputs (stale map files are removed) |
| `-q` | Warnings only |

From Python:

```python
from parcellated_react import run_react

res = run_react(files, 'pet.csv', out_dir='results/univariate')  # mode='univariate' by default
res.stage2['5-HT2A']    # DataFrame: regions x scans
res.stage1['5-HT2A']    # DataFrame: timepoints x scans
res.info                # same content as react_info.json
```

`out_dir` is optional. Omit it to keep the results in memory only. `pet` can
also be a DataFrame.

## Outputs

```
out/
├── stage1/<map>.csv   # rows = timepoints, columns = scans
├── stage2/<map>.csv   # rows = regions,    columns = scans
└── react_info.json    # version, mode, settings, inputs, excluded regions,
                       # PET map correlations and VIFs
```

Read with `pd.read_csv(path, index_col=0)`. If scans differ in length, the
stage 1 columns of shorter scans are NaN-padded.

# Collinearity 

Collinearity among PET maps can make multivariate estimates unstable. PET map
correlations and VIFs are logged and saved in `react_info.json`. See 
https://pmc.ncbi.nlm.nih.gov/articles/PMC13590927/ for further details.


## Citations

If you use this toolbox, please cite it alongside the original REACT paper:

Lawn, T. Parcellated REACT: A toolbox for receptor-enriched analysis of
parcellated fMRI data. Zenodo. https://doi.org/10.5281/zenodo.17643163

Dipasquale, O., et al. (2019). Receptor-Enriched Analysis of functional
connectivity by targets (REACT). *NeuroImage*.
https://doi.org/10.1016/j.neuroimage.2019.04.007

Lawn, T., et al. (2023). From neurotransmitters to networks: Transcending
organizational hierarchies with molecular-informed functional imaging.
*Neuroscience and Biobehavioral Reviews*. https://pubmed.ncbi.nlm.nih.gov/37086932/

Lawn, T., et al. (2023). Spatial Collinearity Constrains Multivariate Molecular‐Enriched
Network Estimation. *Human Brain Mapping*. https://pmc.ncbi.nlm.nih.gov/articles/PMC13590927/

## Contact

Questions and bug reports: please open an
[issue](https://github.com/timlawn/parcellated-REACT/issues), or email Tim Lawn
(tim.lawn@psy.ox.ac.uk).

## License

MIT
