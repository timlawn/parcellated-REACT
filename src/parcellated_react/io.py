"""Loading inputs and writing outputs."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# Cell values treated as missing (NaN); 'n/a' is the BIDS convention
NA_TOKENS = {'', 'n/a', 'na', 'nan'}


def _separator(path):
    """Tab for .tsv files, comma otherwise."""
    return '\t' if Path(path).suffix.lower() == '.tsv' else ','


def _is_number(token):
    token = token.strip()
    if token.lower() in NA_TOKENS:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_timeseries(path):
    """
    Load one scan: rows = timepoints, cols = regions.

    Comma-separated, or tab-separated if the file ends in .tsv. If the first
    row contains any text it is treated as a header and ignored (regions are
    matched to PET maps by position only). Empty cells and 'n/a' are NaN.
    """
    # utf-8-sig strips the byte-order mark Excel adds to CSVs
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.reader(f, delimiter=_separator(path)) if r]
    if not rows:
        raise ValueError(f'{path}: file is empty')
    if len({len(r) for r in rows}) > 1:
        raise ValueError(f'{path}: rows have unequal numbers of fields')
    if not all(_is_number(t) for t in rows[0]):
        rows = rows[1:]
    if not rows:
        raise ValueError(f'{path}: no data rows')

    try:
        data = np.array([[np.nan if t.strip().lower() in NA_TOKENS else float(t)
                          for t in r] for r in rows])
    except ValueError:
        bad = next(t for r in rows for t in r if not _is_number(t))
        raise ValueError(f'{path}: non-numeric value {bad!r} in data rows') from None

    # A header of numeric region labels (e.g. 0, 1, 2 from pandas, or atlas
    # label values) would be read as data. Labels are whole numbers in
    # increasing order; a row of BOLD values practically never is.
    first = data[0]
    if (len(first) >= 3 and np.all(np.isfinite(first))
            and np.all(first == np.round(first)) and np.all(np.diff(first) > 0)):
        raise ValueError(
            f'{path}: first row looks like numeric region labels (whole numbers '
            f'in increasing order); use a text header or none')
    return data


def load_pet(path):
    """
    Load and validate PET maps: CSV/TSV (or DataFrame) with header,
    rows = regions, cols = maps.

    An optional column (or index) named 'region' is used as region labels;
    otherwise regions are numbered from 1. Returns a DataFrame indexed by region.
    """
    if isinstance(path, pd.DataFrame):
        df, path = path.copy(), '<DataFrame>'
    else:
        df = pd.read_csv(path, sep=_separator(path))
    if 'region' in df.columns:
        df = df.set_index('region')
    elif df.index.name != 'region':
        df.index = pd.RangeIndex(1, len(df) + 1, name='region')
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if unnamed:
        raise ValueError(
            f'{path}: unnamed column(s) {unnamed}, probably an index written by '
            f'pandas; save with index=False or name the column \'region\'')
    if df.shape[1] == 0:
        raise ValueError(f'{path}: no PET map columns found')
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f'{path}: non-numeric PET map columns {non_numeric}')
    if df.index.has_duplicates:
        raise ValueError(f'{path}: duplicate region labels')
    for name in df.columns:
        if not isinstance(name, str) or not name.strip() or any(c in name for c in '/\\'):
            raise ValueError(f'{path}: invalid map name {name!r} (used as filename)')
    # Map names become filenames, and Mac/Windows filenames ignore case
    lowered = [c.lower() for c in df.columns]
    clashes = sorted({c for c in df.columns if lowered.count(c.lower()) > 1})
    if clashes:
        raise ValueError(f'{path}: map names must differ by more than case: {clashes}')
    df = df.astype(float)
    flat = [c for c in df.columns if df[c].nunique(dropna=True) < 2]
    if flat:
        raise ValueError(f'{path}: PET map(s) {flat} are constant or empty')
    return df


def read_path_list(path):
    """Read a text file of paths, one per line (blank lines ignored)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def scan_ids(paths):
    """Scan IDs are filename stems; they must be unique."""
    ids = [Path(p).stem for p in paths]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f'Duplicate scan IDs (filename stems): {dupes}')
    return ids


OUTPUT_NAMES = ('stage1', 'stage2', 'react_info.json')


def check_out_dir(out_dir, force):
    """Refuse to overwrite previous outputs unless force (checked before running)."""
    out_dir = Path(out_dir)
    if not force and any((out_dir / n).exists() for n in OUTPUT_NAMES):
        raise FileExistsError(
            f'Outputs already exist in {out_dir}; use force to overwrite')
    return out_dir


def write_outputs(out_dir, stage1, stage2, info):
    """
    Write one CSV per map for each stage, plus react_info.json, replacing any
    previous outputs (called only after all scans have run successfully).

    stage1/<map>.csv : rows = timepoints, cols = scans
    stage2/<map>.csv : rows = regions,    cols = scans
    """
    out_dir = Path(out_dir)
    for n in OUTPUT_NAMES:
        p = out_dir / n
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage, frames in (('stage1', stage1), ('stage2', stage2)):
        (out_dir / stage).mkdir()
        for name, df in frames.items():
            df.to_csv(out_dir / stage / f'{name}.csv')
    with open(out_dir / 'react_info.json', 'w') as f:
        json.dump(info, f, indent=2)
