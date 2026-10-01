"""Command-line interface: parcellated-react."""

import argparse
import logging
import sys

from . import __version__
from .io import read_path_list
from .run import run_react

EPILOG = """\
inputs:
  timeseries  one CSV (or .tsv) per scan; rows = timepoints, cols = regions;
              an optional header row is ignored (regions matched by position)
  pet         one CSV (or .tsv); rows = regions, cols = maps, header = map
              names (optional 'region' column used as region labels)

outputs (in --out):
  stage1/<map>.csv   rows = timepoints, cols = scans
  stage2/<map>.csv   rows = regions,    cols = scans
  react_info.json    settings, excluded regions, PET collinearity

example:
  parcellated-react --timeseries sub-*_timeseries.csv --pet pet.csv \\
      --out results/univariate
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='parcellated-react',
        description='Parcellated REACT: receptor-enriched analysis of '
                    'functional connectivity on region-wise timeseries.',
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--timeseries', nargs='+', metavar='CSV',
                     help='timeseries CSV/TSV files, one per scan')
    src.add_argument('--timeseries-list', metavar='TXT',
                     help='text file listing timeseries files, one per line')
    p.add_argument('--pet', required=True, metavar='CSV', help='PET maps CSV/TSV')
    p.add_argument('--mode', default='univariate', choices=['univariate', 'multivariate'],
                   help='univariate (default): an independent model per map; '
                        'multivariate: all maps in one model')
    p.add_argument('--out', required=True, metavar='DIR', help='output directory')
    p.add_argument('--scale-pet', action='store_true',
                   help='min-max scale each PET map to [0, 1] (changes stage 1 '
                        'amplitudes only; stage 2 is invariant)')
    p.add_argument('--data-norm', action='store_true',
                   help='normalise each region timeseries to unit SD in stage 2 '
                        '(as react-fmri --data_norm); off by default, as in the '
                        'original REACT method')
    p.add_argument('--force', action='store_true',
                   help='overwrite existing outputs in --out')
    p.add_argument('-q', '--quiet', action='store_true', help='only show warnings')
    p.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format='%(levelname)s: %(message)s')
    try:
        files = args.timeseries or read_path_list(args.timeseries_list)
        run_react(files, args.pet, args.mode, out_dir=args.out,
                  scale_pet=args.scale_pet, data_norm=args.data_norm,
                  force=args.force)
    except (ValueError, OSError) as e:
        logging.error('%s', e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
