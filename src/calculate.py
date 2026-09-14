#!/usr/bin/env python
"""
calculate
---------
Reference / FF data extraction for q2mm-amber.

This module keeps the CLI signature of the old q2mm-master calculate.py, so
loop.in RDAT / CDAT lines continue to work. It:

  1. parses a list of -flag/filename pairs (eg "-gh foo.log -i 1"),
  2. turns the Amber flags into calculators.AmberCalculator objects (one
     per leap input), which run the Amber pipeline and read the results
     back as Datum objects,
  3. reads the Gaussian reference files here, with utilities.GaussLog,
  4. returns a flat list of data_structs.Datum objects.

The optimizers do not go through main(): loop.py builds one Calculator from
the CDAT arguments with build_calculator() and hands it to them, and they
evaluate a trial force field with calculator.evaluate(ff).

Currently implemented data types
--------------------------------
Amber   : -ah (Hessian), -ageig (Hessian projected on the QM normal modes,
          "somename.in,somename.log"), -ae, -ae1, -aeo, -ae1o (energies),
          -abo, -aao, -ato, -ab, -aa, -at (geometry)
Gaussian: -gh (Hessian), -geigz (eigenmatrix: the QM eigenvalues on the
          diagonal, zero elsewhere), -ge, -ge1, -geo, -ge1o (energies),
          -gabo, -gaao, -gato (the mol2 geometry measured through Amber)

Hessian fitting pairs -gh with -ah element by element; eigenmode fitting
pairs -geigz with -ageig, comparing the force field's curvature along each
QM normal mode (and the coupling between modes) with the QM eigenvalues.
Both need the Gaussian job and the mol2 in the same Cartesian frame with the
same atom order (run Gaussian with nosymm); eigenmode fitting also wants
freq=hpmodes, since the modes are read from the printed normal coordinates.

Anything else from the old code (MacroModel / Jaguar / Tinker) is
parsed but ignored.

main(args) -> list[Datum]
build_calculator(args, ff=None) -> calculators.Calculator
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import logging.config
import os
import sys
from collections import OrderedDict

import numpy as np

import calculators
import constants as co
import math_util
import score
from data_structs import Datum, datum_from_energy, datums_from_eigenmatrix, datums_from_hessian
import utilities

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


# ---------------------------------------------------------------------------
# Fixed atoms (FXATM). Hessian elements that couple one of these atoms are
# given weight 0 -- excluded from the fit. Mirrors q2mm-master's fixedatoms.txt
# feature, but the file path is supplied by the loop.in `FXATM <file>` command
# instead of a hardcoded "fixedatoms.txt". The set lives in constants (co) so
# both calculate and score can read it without a circular import, and the
# exclusion is applied at SCORE time (score.compare_data) -- that makes it
# placement-proof: FXATM only has to precede the COMP/HYBR that scores, not
# the CDAT that built the data. co is a module global, so Linux-fork swarm
# workers inherit it like the rest of the swarm's worker state.
# ---------------------------------------------------------------------------


def load_fixed_atoms(path):
    """Read a fixed-atoms file (one 1-based atom index per line) and store the
    exclusion set in co.FIXED_ATOMS. Hessian elements coupling one of these
    atoms are dropped from the objective at score time (score.compare_data).
    Returns the set; a missing/empty path clears the exclusion."""
    atoms = set()
    if path and os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                s = line.split()
                if len(s) == 1:
                    atoms.add(int(s[0]))
        logger.log(20, "FXATM: excluding {} fixed atom(s) from the Hessian "
                   "fit: {}".format(len(atoms), sorted(atoms)))
    else:
        logger.warning("FXATM: file not found, no atoms excluded: {}".format(path))
    co.FIXED_ATOMS = set(atoms)
    return atoms


# ---------------------------------------------------------------------------
# CLI argument parser (kept compatible with the old codebase)
# ---------------------------------------------------------------------------

def return_calculate_parser(add_help=True, parents=None):
    if parents is None:
        parents = []
    if add_help:
        parser = argparse.ArgumentParser(description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(add_help=False, parents=parents)

    g = parser.add_argument_group("general")
    g.add_argument("--directory", "-d", type=str, default=os.getcwd(),
                   help="Working directory.")
    g.add_argument("--doprint", "-p", action="store_true")
    g.add_argument("--ffpath", "-f", type=str, default=None)
    g.add_argument("--invert", "-i", type=float, default=None,
                   help="Invert smallest Hessian eigenvalue to this value.")
    g.add_argument("--norun", "-n", action="store_true",
                   help="Don't actually run AMBER / leap; just read.")
    g.add_argument("--fake", action="store_true",
                   help="Generate placeholder zero-value data.")
    g.add_argument("--weight", "-w", action="store_true",
                   help="Apply weights from constants.WEIGHTS to data.")
    g.add_argument("--subnames", "-s", type=str, nargs="+",
                   default=["OPT"])
    g.add_argument("--check", "-c", action="store_true")
    g.add_argument("--nocheck", "-nc", action="store_false", dest="check")
    g.add_argument("--append", "-a", type=str, default=None)

    # Gaussian / Amber flags — every command takes a list of filenames.
    for flag in ("gta", "gtb", "gtt",
                 "gaa", "gab", "gat", "gaao", "gabo", "gato",
                 "ge", "ge1", "gea", "geo", "ge1o", "geao",
                 "gh", "geigz"):
        parser.add_argument("-" + flag, type=str, nargs="+",
                            action="append", default=[])
    parser.add_argument("-r", type=str, nargs="+", action="append", default=[])
    for flag in ("ae", "ae1", "aeo", "ae1o",
                 "abo", "aao", "ato", "ab", "aa", "at", "ah", "aha"):
        parser.add_argument("-" + flag, type=str, nargs="+",
                            action="append", default=[])
    parser.add_argument("-ageig", type=str, nargs="+", action="append", default=[],
                        metavar="somename.in,somename.log",
                        help="Amber Hessian projected on the Gaussian normal modes.")
    return parser


def _split_args(args):
    if isinstance(args, str):
        args = args.split()
    return list(args)


def _full_path(opts, filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(opts.directory, filename)


def _flag_files(opts, flag):
    """Filenames given to one flag; argparse's append/nargs gives a list of lists."""
    for filenames in getattr(opts, flag, []) or []:
        for filename in filenames:
            yield filename


def _split_pair(token):
    """'somename.in,somename.log' -> (leap input, gaussian log)."""
    parts = token.split(",")
    if len(parts) != 2:
        raise ValueError("-ageig takes 'somename.in,somename.log', got '{}'".format(token))
    return parts[0], parts[1]


def _sibling(path, extension):
    return os.path.splitext(path)[0] + extension


# Reference and calculated Hessians (and normal modes) are compared element
# by element, which only means something if the Gaussian job and the mol2 the
# Amber topology is built from share one Cartesian frame and one atom order.
# A rotated or reordered mol2 gives a finite, wrong fit with no other symptom.
FRAME_TOLERANCE = 0.01   # Angstrom, after removing the centroid
# max |V V^T - I| above which the printed normal modes are too coarse to use
MODE_ORTHONORMALITY_TOLERANCE = 0.01


def _warn_if_frames_differ(log_atoms, mol2_path, label):
    """Warn when the geometry in a Gaussian log and the mol2 do not coincide."""
    if not log_atoms or not mol2_path or not os.path.isfile(mol2_path):
        return
    try:
        mol2_atoms = utilities.Mol2(mol2_path).structures[0].atoms
        a = np.array([[x.x, x.y, x.z] for x in log_atoms], dtype=float)
        b = np.array([[x.x, x.y, x.z] for x in mol2_atoms], dtype=float)
    except Exception as e:
        logger.debug("Frame check skipped for {}: {}".format(label, e))
        return
    if a.shape != b.shape:
        logger.warning("{}: {} atoms in the Gaussian log but {} in {}; the data "
                       "cannot be compared element by element.".format(
                           label, len(a), len(b), mol2_path))
        return
    deviation = float(np.abs((a - a.mean(axis=0)) - (b - b.mean(axis=0))).max())
    if deviation > FRAME_TOLERANCE:
        logger.warning("{}: the geometry in the Gaussian log and {} differ by up to "
                       "{:.3f} A after centering, so the Hessian / normal-mode frames "
                       "do not match (run Gaussian with nosymm and build the mol2 "
                       "from the same coordinates).".format(label, mol2_path, deviation))


# ---------------------------------------------------------------------------
# Gaussian reference files
# ---------------------------------------------------------------------------

def _gauss_log_hessian(path, invert=None):
    """
    Read the raw 3N x 3N Hessian from a Gaussian frequency archive,
    mass-weight it, and emit its lower-triangular elements as
    Datum (typ='h'). Matches q2mm-master's 'gh' convention so the
    Gaussian and Amber sides always produce identical shapes
    (3N(3N+1)/2 elements) for any molecule size, with no projection
    or eigenvalue-counting needed.

    If `invert` is given, the most-negative Hessian eigenvalue (the TS
    reaction coordinate) is replaced by that value, flipping the
    imaginary frequency to a real one before the matrix is reassembled.
    """
    log = utilities.GaussLog(path)
    # The archive block at the end of the .log carries the raw lower-tri
    # Hessian; read_archive() parses it into log.structures[0].hess.
    try:
        log.read_archive()
    except Exception as e:
        logger.warning("Gaussian archive parse failed for {}: {}".format(path, e))
        return []
    if not log.structures or log.structures[0].hess is None:
        logger.warning("No Hessian in Gaussian archive: {}".format(path))
        return []
    struct = log.structures[0]
    # Copy so the in-place mass weighting does not pollute the cached struct.
    H = struct.hess.copy()
    # Gaussian's archive Hessian is NOT mass-weighted; multiply each row/col
    # by 1/sqrt(m_i) to convert to mass-weighted units that match Amber's
    # nab/nmode output.
    utilities.mass_weight_hessian(H, struct.atoms)
    if invert is not None:
        H = math_util.invert_lowest_eigenvalue(H, invert, label=path)
    _warn_if_frames_differ(struct.atoms, _sibling(path, ".mol2"), "-gh " + os.path.basename(path))
    # One Datum per lower-tri element, tagged typ='h' so it picks up the
    # uniform Hessian weight (WEIGHTS['h']) at score time.
    return datums_from_hessian(H, os.path.basename(path))


def _gauss_log_eigenmatrix(path, invert=None):
    """
    The reference eigenmatrix: the eigenvalues of the mass-weighted QM
    Hessian, read from the frequency section of the Gaussian log (force
    constant over reduced mass, per mode) in kJ/mol/A^2/amu, on the diagonal
    of an otherwise zero matrix, emitted as Datum (typ='eig'). Mode 1 is the
    lowest, the transition-state mode; with `invert` its (negative)
    eigenvalue is replaced by that value. Matches upstream q2mm's -geigz.
    """
    log = utilities.GaussLog(path)
    evals = np.asarray(log.evals, dtype=float)
    if evals.size == 0:
        logger.warning("No normal modes in the Gaussian log: {}".format(path))
        return []
    evals = evals * co.HESSIAN_CONVERSION
    if invert is not None:
        evals = math_util.replace_lowest_eigenvalue(evals, invert, label=path)
    return datums_from_eigenmatrix(np.diag(evals), os.path.basename(path))


def reference_modes(log_path, mol2_path=None):
    """
    The normalized, mass-weighted QM eigenvectors from the frequency section
    of a Gaussian log (n_modes x 3N), for projecting the calculated Hessian
    (-ageig). Warns when the log's geometry and the mol2 are not in the same
    frame. Use freq=hpmodes: the low-precision normal coordinates are only
    orthonormal to a few percent.
    """
    log = utilities.GaussLog(log_path)
    evecs = np.asarray(log.evecs, dtype=float)
    if evecs.size == 0:
        raise ValueError("No normal modes in the Gaussian log: {}".format(log_path))
    label = "-ageig " + os.path.basename(log_path)
    # Low-precision normal coordinates (no freq=hpmodes: two decimals) give
    # eigenvectors that are only roughly orthonormal, and that error goes
    # straight into every element of the projected Hessian.
    error = float(np.abs(evecs.dot(evecs.T) - np.eye(len(evecs))).max())
    if error > MODE_ORTHONORMALITY_TOLERANCE:
        logger.warning("{}: the normal modes are orthonormal only to {:.1%}; rerun the "
                       "frequency job with freq=hpmodes so the eigenvectors are printed "
                       "at full precision.".format(label, error))
    try:
        log_atoms = log.last_orientation()
    except Exception as e:   # the check must never take a run down
        logger.debug("Frame check skipped for {}: {}".format(label, e))
        log_atoms = []
    _warn_if_frames_differ(log_atoms, mol2_path, label)
    return evecs


def _gauss_log_energy(path, typ="e", group_idx=1):
    """Pull a single SCF energy from a Gaussian log via utilities.GaussLog."""
    log = utilities.GaussLog(path)
    energy = None
    # The new GaussLog doesn't currently expose energy directly; scan archive.
    for line in log.lines:
        if "HF=" in line:
            try:
                seg = line.strip().split("HF=")[1]
                val = float(seg.split("\\")[0])
                energy = val
                break
            except (IndexError, ValueError):
                continue
    if energy is None:
        logger.warning("Could not extract Gaussian energy from {}".format(path))
        return []
    return [datum_from_energy(energy * co.HARTREE_TO_KJMOL, group_idx,
                              os.path.basename(path), typ=typ)]


# (command flag, datum type, collector function)
_GAUSSIAN_DISPATCH = [
    ("gh",    "h",   lambda p, opts: _gauss_log_hessian(p, invert=opts.invert)),
    ("geigz", "eig", lambda p, opts: _gauss_log_eigenmatrix(p, invert=opts.invert)),
    ("ge",   "e",   lambda p, opts: _gauss_log_energy(p, typ="e")),
    ("ge1",  "e1",  lambda p, opts: _gauss_log_energy(p, typ="e1")),
    ("geo",  "eo",  lambda p, opts: _gauss_log_energy(p, typ="eo")),
    ("ge1o", "e1o", lambda p, opts: _gauss_log_energy(p, typ="e1o")),
]

# Every flag that produces data, with its datum type (for --fake).
_FLAG_TYPES = OrderedDict(
    [(flag, spec[0]) for flag, spec in calculators.AMBER_COMMANDS.items()]
    + [(flag, calculators.AMBER_COMMANDS[cmd][0])
       for flag, cmd in calculators.REFERENCE_GEOMETRY_COMMANDS.items()]
    + [(flag, typ) for flag, typ, _ in _GAUSSIAN_DISPATCH]
)


# ---------------------------------------------------------------------------
# Amber calculators
# ---------------------------------------------------------------------------

def build_calculators(opts, ff=None, runner=None):
    """
    The AmberCalculators the Amber flags in `opts` ask for: one per leap
    input (and per minimized/single-point kind), in order of first
    appearance, holding every command given for that file. After them, one
    reference-geometry calculator (prefix "gaus", single point) per Gaussian
    log named by -gabo/-gaao/-gato; its leap input is <log stem>.in next to
    the log, so the reference geometry comes from that mol2.
    """
    calculated = OrderedDict()   # (leap input path, minimize) -> [commands]
    mode_logs = {}               # (leap input path, minimize) -> gaussian log for -ageig
    for command, (_, minimize, _, _) in calculators.AMBER_COMMANDS.items():
        for filename in _flag_files(opts, command):
            log_name = None
            if command == "ageig":
                filename, log_name = _split_pair(filename)
            key = (_full_path(opts, filename), minimize)
            calculated.setdefault(key, []).append(command)
            if log_name is not None:
                log_path = _full_path(opts, log_name)
                if mode_logs.get(key, log_path) != log_path:
                    raise ValueError("-ageig: two different logs for {}".format(key[0]))
                mode_logs[key] = log_path
    reference = OrderedDict()    # gaussian log path -> [commands]
    for flag, command in calculators.REFERENCE_GEOMETRY_COMMANDS.items():
        for filename in _flag_files(opts, flag):
            reference.setdefault(_full_path(opts, filename), []).append(command)

    calcs = []
    for key, commands in calculated.items():
        in_path = key[0]
        eigenvectors = None
        if key in mode_logs:
            eigenvectors = reference_modes(mode_logs[key], _mol2_of(in_path))
        calcs.append(calculators.AmberCalculator(
            os.path.dirname(in_path), os.path.basename(in_path), commands,
            ff=ff, invert=opts.invert, runner=runner, eigenvectors=eigenvectors))
    for log_path, commands in reference.items():
        # NB: the reference geometry comes from <name>.mol2 (via <name>.in),
        # NOT from the Gaussian .log -- the .log path only supplies <name>.
        # Put the QM reference geometry in the mol2 accordingly.
        stem = os.path.splitext(os.path.basename(log_path))[0]
        calcs.append(calculators.AmberCalculator(
            os.path.dirname(log_path), stem + ".in", commands,
            prefix="gaus", runner=runner, src_name=os.path.basename(log_path)))
    return calcs


def _mol2_of(in_path):
    """The mol2 a leap input loads (first .mol2 it names), else <stem>.mol2."""
    if os.path.isfile(in_path):
        for rel in utilities.AmberLeapInput(in_path).referenced_files():
            if rel.endswith(".mol2"):
                return os.path.join(os.path.dirname(in_path), rel)
    return _sibling(in_path, ".mol2")


def build_calculator(args, ff=None, runner=None):
    """
    The Calculator for the FF side of a fit, from CDAT-style arguments:
    the AmberCalculator for the one leap input named, or a CalculatorGroup
    when several are. `ff` is the force field being fitted; trial force
    fields are written to ff.path.
    """
    opts = return_calculate_parser().parse_args(_split_args(args))
    calcs = [c for c in build_calculators(opts, ff=ff, runner=runner)
             if c.prefix == "amber"]
    if not calcs:
        raise ValueError("No Amber commands to build a calculator from: {}".format(args))
    if len(calcs) == 1:
        return calcs[0]
    return calculators.CalculatorGroup(calcs)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def collect_data(opts, ff=None, runner=None):
    """
    Datum objects for every enabled flag in opts: the Amber calculators
    first (calculated data, then Amber-measured reference geometry), then
    the Gaussian log collectors.
    """
    data = []
    if opts.fake:
        # produce one zeroed datum per file so optimizers don't crash
        for flag, typ in _FLAG_TYPES.items():
            for filename in _flag_files(opts, flag):
                data.append(Datum(val=0.0, typ=typ, src_1=os.path.basename(filename)))
        return data
    for calc in build_calculators(opts, ff=ff, runner=runner):
        logger.log(20, "  -- {} {}".format(" ".join(calc.commands), calc.leap_input.path))
        if opts.norun:
            data.extend(calc.gather_results())
        else:
            data.extend(calc.evaluate(ff if calc.prefix == "amber" else None))
    for flag, _typ, collector in _GAUSSIAN_DISPATCH:
        for filename in _flag_files(opts, flag):
            full = _full_path(opts, filename)
            logger.log(20, "  -- {} {}".format(flag, full))
            data.extend(collector(full, opts))
    return data


def main(args, ff=None):
    """
    Args may be a single string or list of strings. Returns a flat list
    of Datum objects. With `ff`, the force field is written to disk before
    the Amber calculations run; without it the frcmod on disk is used.
    """
    parser = return_calculate_parser()
    opts = parser.parse_args(_split_args(args))
    data = collect_data(opts, ff=ff)
    if opts.weight:
        score.import_weights(data)
    if opts.doprint:
        for d in data:
            print("{:30s} {:>11.4f}".format(d.lbl, d.val))
    return data


if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
