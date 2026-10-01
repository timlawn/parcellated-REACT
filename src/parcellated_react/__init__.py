"""
Parcellated REACT (Receptor-Enriched Analysis of functional Connectivity by
Targets) for region-wise fMRI timeseries and PET maps in the same parcellation.

Reference: Dipasquale et al. (2019) NeuroImage 195:252-260
"""

__version__ = '2.0.0'

from .core import react  # noqa: E402
from .run import ReactResult, run_react  # noqa: E402

__all__ = ['react', 'run_react', 'ReactResult', '__version__']
