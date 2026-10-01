"""
Core REACT computations on parcellated data (pure numpy, no I/O).

Conventions
-----------
timeseries : (n_timepoints, n_regions)
pet        : (n_regions, n_maps)
stage1     : (n_timepoints, n_maps)
stage2     : (n_maps, n_regions)
"""

import numpy as np

MODES = ('univariate', 'multivariate')


def minmax_scale(pet):
    """Scale each PET map (column) to [0, 1], ignoring NaNs."""
    pet = np.asarray(pet, dtype=float)
    lo = np.nanmin(pet, axis=0)
    hi = np.nanmax(pet, axis=0)
    if np.any(hi == lo):
        raise ValueError('Cannot min-max scale a constant PET map')
    return (pet - lo) / (hi - lo)


def valid_regions(timeseries, pet):
    """
    Boolean mask of regions usable for REACT.

    A region is excluded if any PET value or any timepoint is non-finite,
    or if its timeseries has zero temporal variance (e.g. all zeros).
    """
    timeseries = np.asarray(timeseries, dtype=float)
    pet = np.asarray(pet, dtype=float)
    finite = np.all(np.isfinite(timeseries), axis=0) & np.all(np.isfinite(pet), axis=1)
    # np.ptp on non-finite columns is meaningless, but those are already excluded
    with np.errstate(invalid='ignore'):
        varying = np.ptp(timeseries, axis=0) > 0
    return finite & varying


def stage1(timeseries, pet):
    """
    Stage 1: PET maps as spatial regressors.

    For each timepoint, regress the spatially demeaned BOLD pattern on the
    spatially demeaned PET maps. Timeseries should already be restricted to
    valid regions.

    Returns (n_timepoints, n_maps).
    """
    x = pet - pet.mean(axis=0)
    y = timeseries - timeseries.mean(axis=1, keepdims=True)
    beta, *_ = np.linalg.lstsq(x, y.T, rcond=None)
    return beta.T


def stage2(timeseries, stage1_ts, data_norm=False):
    """
    Stage 2: stage 1 timeseries as temporal regressors.

    The design is demeaned and normalised to unit standard deviation
    (equivalent to fsl_glm --demean --des_norm); the data are demeaned and,
    if data_norm, also normalised to unit standard deviation (as
    react-fmri --data_norm).

    Returns (n_maps, n_regions).
    """
    sd = stage1_ts.std(axis=0)
    if np.any(sd == 0):
        raise ValueError('A stage 1 timeseries is constant; cannot run stage 2')
    x = (stage1_ts - stage1_ts.mean(axis=0)) / sd
    y = timeseries - timeseries.mean(axis=0)
    if data_norm:
        y = y / y.std(axis=0)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def react(timeseries, pet, mode='univariate', data_norm=False):
    """
    Run REACT for one scan.

    Parameters
    ----------
    timeseries : array (n_timepoints, n_regions)
    pet : array (n_regions, n_maps)
    mode : 'univariate' (default) fits each map in an independent model;
           'multivariate' fits all maps jointly in both stages.
    data_norm : normalise each region's timeseries to unit SD in stage 2.

    Returns
    -------
    stage1 : array (n_timepoints, n_maps)
        Temporally demeaned stage 1 timeseries.
    stage2 : array (n_maps, n_regions)
        Stage 2 maps; excluded regions are NaN.
    mask : bool array (n_regions,)
        Regions included in the model.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    timeseries = np.asarray(timeseries, dtype=float)
    pet = np.asarray(pet, dtype=float)
    if timeseries.ndim != 2 or pet.ndim != 2:
        raise ValueError('timeseries and pet must both be 2D')
    n_timepoints, n_regions = timeseries.shape
    n_maps = pet.shape[1]
    if pet.shape[0] != n_regions:
        raise ValueError(
            f'Region count mismatch: timeseries has {n_regions}, '
            f'PET has {pet.shape[0]}')

    mask = valid_regions(timeseries, pet)
    if mask.sum() <= n_maps:
        raise ValueError(
            f'Only {mask.sum()} valid regions for {n_maps} PET maps')

    # Temporal demeaning per region. Only shifts stage 1 by a constant per
    # map, so stage 2 is unaffected, but stage 1 output is centred on zero.
    ts = timeseries[:, mask]
    ts = ts - ts.mean(axis=0)
    pt = pet[mask]

    if mode == 'multivariate':
        s1 = stage1(ts, pt)
        s2_valid = stage2(ts, s1, data_norm)
    else:
        s1 = np.empty((n_timepoints, n_maps))
        s2_valid = np.empty((n_maps, mask.sum()))
        for m in range(n_maps):
            s1[:, [m]] = stage1(ts, pt[:, [m]])
            s2_valid[[m]] = stage2(ts, s1[:, [m]], data_norm)

    s2 = np.full((n_maps, n_regions), np.nan)
    s2[:, mask] = s2_valid
    return s1, s2, mask


def collinearity(pet):
    """
    Spatial correlations and variance inflation factors of PET maps,
    computed over regions finite in all maps.

    Returns (corr (n_maps, n_maps), vif (n_maps,)).
    """
    pet = np.asarray(pet, dtype=float)
    pet = pet[np.all(np.isfinite(pet), axis=1)]
    n_maps = pet.shape[1]
    if n_maps == 1:
        return np.ones((1, 1)), np.ones(1)
    corr = np.corrcoef(pet, rowvar=False)
    try:
        vif = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(n_maps, np.inf)
    return corr, vif
