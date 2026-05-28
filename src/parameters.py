#!/usr/bin/env python
"""
parameters
----------
Parameter selection / trimming for the new Q2MM (q2mm-amber-main).

This is a port of q2mm-master/q2mm/parameters.py adapted to the new
data_structs.* module. The PARM command in loop.in is dispatched here.

Format of the parameter file (same as the old codebase):

    ff_row ff_col [neg|pos|both | min_value max_value]

    ff_row  Integer line number of the parameter in the .frcmod / .fld.
    ff_col  Column within that line (1, 2, 3) depending on parameter type.
            For an AMBER bond row, col 1 is the force constant, col 2 is
            the equilibrium length; for an angle row similarly; for a
            torsion the cols are V1, V2, V3.
    neg     Forces the parameter into negative-only values.
    pos     Forces the parameter into positive-only values.
    both    Allows both positive and negative.
    a b     Explicit numeric bounds.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import logging.config
import sys

import constants as co
from data_structs import AmberFF

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

ALL_PARM_TYPES = (
    "ae", "af", "be", "bf", "df",
    "imp1", "imp2", "sb", "q", "vdwe", "vdwr",
)


def return_params_parser(add_help=True):
    """argparse parser for the standalone CLI."""
    if add_help:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
    else:
        parser = argparse.ArgumentParser(add_help=False)
    g = parser.add_argument_group("parameters")
    g.add_argument("--all", "-a", action="store_true",
                   help="Select all available parameters.")
    g.add_argument("--ffpath", "-f", metavar="frcmod", default="frcmod",
                   help="Path to the force field.")
    g.add_argument("--nozero", action="store_true",
                   help="Exclude any parameters whose value is zero.")
    g.add_argument("--pfile", "-pf", type=str, metavar="filename",
                   help="Use a file to select parameters.")
    g.add_argument("--ptypes", "-pt", nargs="+", default=[],
                   help="Select these parameter types.")
    g.add_argument("--printparams", "-pp", action="store_true",
                   help="Print information about the selected parameters.")
    return parser


def trim_params_by_type(params, ptypes):
    """Keep only the parameters with a ptype in ptypes."""
    chosen = [p for p in params if p.ptype in ptypes]
    logger.log(20, "  -- Trimmed number of parameters down to {}.".format(len(chosen)))
    return chosen


def trim_params_by_file(params, filename):
    """
    Trim the parameter list against a parameter selection file.

    Returns a list of Param objects taken from `params` whose
    (ff_row, ff_col) appear in the file. The file may also set
    _allowed_range for each chosen parameter (neg / pos / both /
    explicit min max).
    """
    chosen = []
    file_specs = read_param_file(filename)
    for param in params:
        for ff_row, ff_col, allowed_range in file_specs:
            if param.ff_row == ff_row and param.ff_col == ff_col:
                if allowed_range is not None:
                    param._allowed_range = tuple(allowed_range)
                    param.value_in_range(param.value)
                chosen.append(param)
    logger.log(20, "  -- Trimmed number of parameters down to {}.".format(len(chosen)))
    return chosen


def read_param_file(filename):
    """
    Read a parameter selection file. Returns a list of (ff_row, ff_col,
    allowed_range) tuples where allowed_range is a 2-list or None.

    Comments after # or ! are ignored.
    """
    specs = []
    with open(filename, "r") as f:
        for line in f:
            line = line.partition("#")[0]
            line = line.partition("!")[0]
            cols = line.split()
            if not cols:
                continue
            ff_row = int(cols[0])
            ff_col = int(cols[1])
            rest = cols[2:]
            if "neg" in rest:
                allowed = [-float("inf"), 0.0]
            elif "pos" in rest:
                allowed = [0.0, float("inf")]
            elif "both" in rest:
                allowed = [-float("inf"), float("inf")]
            elif rest:
                allowed = [float(x) for x in rest[:2]]
            else:
                allowed = None
            specs.append((ff_row, ff_col, allowed))
    return specs


def main(args):
    if sys.version_info > (3, 0):
        if isinstance(args, str):
            args = args.split()
    parser = return_params_parser()
    opts = parser.parse_args(args)

    ff = AmberFF(opts.ffpath)
    ff.import_ff()

    if opts.all:
        opts.ptypes = list(ALL_PARM_TYPES)
    logger.log(20, "Selected parameter types: {}".format(" ".join(opts.ptypes)))

    params = []
    if opts.ptypes:
        params.extend(trim_params_by_type(ff.params, opts.ptypes))
    if opts.pfile:
        params.extend(trim_params_by_file(ff.params, opts.pfile))
    if opts.nozero:
        params = [p for p in params if p.value != 0.0]
    logger.log(20, "  -- Total chosen parameters: {}".format(len(params)))

    if opts.printparams:
        for p in params:
            if p.allowed_range:
                print("{} {} {} {}".format(
                    p.ff_row, p.ff_col, p.allowed_range[0], p.allowed_range[1]))
            else:
                print("{} {}".format(p.ff_row, p.ff_col))

    ff.params = params
    return ff


if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
