"""Run REACT across scans and assemble per-map outputs."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import collinearity, minmax_scale, react
from .io import check_out_dir, load_pet, load_timeseries, scan_ids, write_outputs

log = logging.getLogger(__name__)


@dataclass
class ReactResult:
    """
    stage1 : {map: DataFrame (timepoints x scans)}
    stage2 : {map: DataFrame (regions x scans)}
    info   : run metadata (also written to react_info.json)
    """
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def run_react(timeseries, pet, mode='univariate', out_dir=None, scale_pet=False,
              data_norm=False, force=False):
    """
    Run parcellated REACT on a set of scans.

    Parameters
    ----------
    timeseries : list of paths
        One CSV/TSV per scan (rows = timepoints, cols = regions), with an
        optional header row that is ignored. Scan IDs are the filename stems.
    pet : path or DataFrame
        PET maps (rows = regions, cols = maps, header = map names), aligned
        to the same parcellation and region order as the timeseries.
    mode : 'univariate' (default) or 'multivariate'
    out_dir : path, optional
        If given, outputs are written here.
    scale_pet : bool
        Min-max scale each PET map to [0, 1]. Affects stage 1 amplitudes
        only; stage 2 maps are invariant to it.
    data_norm : bool
        Normalise each region's timeseries to unit SD in stage 2 (as
        react-fmri --data_norm). Stage 2 betas become standardised.
    force : bool
        Overwrite existing outputs in out_dir.

    Returns
    -------
    ReactResult
    """
    paths = [str(p) for p in timeseries]
    if not paths:
        raise ValueError('No timeseries files given')
    ids = scan_ids(paths)

    pet_source = '<DataFrame>' if isinstance(pet, pd.DataFrame) else str(Path(pet).resolve())
    pet_df = load_pet(pet)
    maps = list(pet_df.columns)
    pet_values = pet_df.to_numpy()
    if scale_pet:
        pet_values = minmax_scale(pet_values)

    corr, vif = collinearity(pet_values)
    log.info('Mode: %s | %d scans | %d regions | maps: %s',
             mode, len(paths), len(pet_df), ', '.join(maps))
    if len(maps) > 1:
        log.info('PET VIF: %s', ', '.join(f'{m}={v:.2f}' for m, v in zip(maps, vif)))

    if out_dir is not None:
        out_dir = check_out_dir(out_dir, force)

    s1 = {m: {} for m in maps}
    s2 = {m: {} for m in maps}
    excluded = {}
    for i, (sid, path) in enumerate(zip(ids, paths), 1):
        log.info('[%d/%d] %s', i, len(paths), sid)
        ts = load_timeseries(path)
        try:
            stage1_ts, stage2_maps, mask = react(ts, pet_values, mode, data_norm)
        except ValueError as e:
            raise ValueError(f'{path}: {e}') from None
        if not mask.all():
            excluded[sid] = [_label(r) for r in pet_df.index[~mask]]
            log.warning('%s: %d region(s) excluded (non-finite or zero variance)',
                        sid, (~mask).sum())
        for j, m in enumerate(maps):
            s1[m][sid] = pd.Series(stage1_ts[:, j],
                                   index=pd.RangeIndex(1, len(stage1_ts) + 1))
            s2[m][sid] = stage2_maps[j]

    result = ReactResult()
    for m in maps:
        # Scans of unequal length are NaN-padded
        df1 = pd.DataFrame(s1[m])
        df1.index.name = 'timepoint'
        result.stage1[m] = df1
        result.stage2[m] = pd.DataFrame(s2[m], index=pet_df.index)

    result.info = {
        'toolbox': 'parcellated_react',
        'version': __version__,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'mode': mode,
        'scale_pet': scale_pet,
        'data_norm': data_norm,
        'pet_file': pet_source,
        'maps': maps,
        'n_regions': len(pet_df),
        'scans': dict(zip(ids, (str(Path(p).resolve()) for p in paths))),
        'excluded_regions': excluded,
        'pet_collinearity': {
            'vif': dict(zip(maps, _json_safe(vif))),
            'correlation': {a: dict(zip(maps, _json_safe(row)))
                            for a, row in zip(maps, corr)},
        },
    }

    if out_dir is not None:
        write_outputs(out_dir, result.stage1, result.stage2, result.info)
        log.info('Outputs written to %s', out_dir)
    return result


def _json_safe(values):
    """Round to 4 dp; non-finite values (e.g. infinite VIF) become null."""
    return [round(float(v), 4) if np.isfinite(v) else None for v in values]


def _label(x):
    """Region label as a JSON-safe value."""
    return x.item() if hasattr(x, 'item') else x
