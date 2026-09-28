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
same atom order (run Gaussian with nosymm).

--modes chooses where -geigz / -ageig take the QM normal modes from:
"printed" (default) reads the frequency section of the log, which wants
freq=hpmodes and, without nosymm, is in Gaussian's rotated standard
orientation; "archive" diagonalizes the archive Hessian (the one -gh reads),
at full precision and in its input frame. Give the same --modes on the RDAT
and CDAT lines. -ageig stops when the modes and the mol2 are in different
frames, unless --allow-frame-mismatch is given.

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
    g.add_argument("--modes", choices=MODE_SOURCES, default="printed",
                   help="Where -geigz / -ageig take the QM normal modes from: "
                        "'printed' (the log's frequency section; default) or "
                        "'archive' (diagonalize the archive Hessian). Use the "
                        "same value on the RDAT and CDAT lines.")
    g.add_argument("--allow-frame-mismatch", action="store_true",
                   help="Let -ageig run, with a warning, when the normal modes "
                        "and the mol2 are in different frames (default: stop).")
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
# where -geigz / -ageig take the QM normal modes from (--modes)
MODE_SOURCES = ("printed", "archive")


def _coordinates(atoms):
    """N x 3 array of the x, y, z of Atom objects."""
    return np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)


def _centered_deviation(a, b):
    """Largest coordinate difference between two geometries after centering."""
    return float(np.abs((a - a.mean(axis=0)) - (b - b.mean(axis=0))).max())


def _frame_deviation(log_atoms, mol2_path, label):
    """Largest coordinate difference (A) between a Gaussian geometry and the
    mol2 after centering both, or None when it cannot be measured: no geometry,
    no or unreadable mol2, or different atom counts (the last one warned)."""
    if not log_atoms or not mol2_path or not os.path.isfile(mol2_path):
        return None
    try:
        a = _coordinates(log_atoms)
        b = _coordinates(utilities.Mol2(mol2_path).structures[0].atoms)
    except Exception as e:
        logger.debug("Frame check skipped for {}: {}".format(label, e))
        return None
    if a.shape != b.shape:
        logger.warning("{}: {} atoms in the Gaussian log but {} in {}; the data "
                       "cannot be compared element by element.".format(
                           label, len(a), len(b), mol2_path))
        return None
    return _centered_deviation(a, b)


def _warn_if_frames_differ(log_atoms, mol2_path, label):
    """Warn when the geometry in a Gaussian log and the mol2 do not coincide."""
    deviation = _frame_deviation(log_atoms, mol2_path, label)
    if deviation is not None and deviation > FRAME_TOLERANCE:
        logger.warning("{}: the geometry in the Gaussian log and {} differ by up to "
                       "{:.3f} A after centering, so the Hessian / normal-mode frames "
                       "do not match (run Gaussian with nosymm and build the mol2 "
                       "from the same coordinates).".format(label, mol2_path, deviation))


def _check_mode_frame(log_atoms, mol2_path, label, geometry, hint, allow_mismatch=False):
    """The -ageig frame check: the normal modes (whose frame is that of
    `geometry`, a description of log_atoms) and the force-field Hessian built
    from the mol2 must share one frame, or V H V^T tests every mode with the
    wrong motion. Stops the run unless allow_mismatch, then only warns.
    Returns the deviation (A), or None when it could not be measured."""
    deviation = _frame_deviation(log_atoms, mol2_path, label)
    if deviation is None or deviation <= FRAME_TOLERANCE:
        return deviation
    message = ("{}: {} and {} differ by up to {:.3f} A after centering, so the normal-mode "
               "and Hessian frames do not match and the eigenmode fit would be meaningless. "
               "{}".format(label, geometry, mol2_path, deviation, hint))
    if allow_mismatch:
        logger.warning(message + " Continuing because of --allow-frame-mismatch.")
        return deviation
    raise ValueError(message + " (--allow-frame-mismatch runs it anyway.)")


# The -geigz eigenvalues and the -ageig modes of one log must come from the
# same --modes source, or the fit compares the eigenvalues of one set of modes
# with the curvature along another. loop.py runs RDAT (-geigz) and CDAT
# (-ageig) as separate calls, so -geigz records its source for each log here
# and -ageig checks it -- a module global, like co.FIXED_ATOMS.
_REFERENCE_MODE_SOURCE = {}


def _record_mode_source(log_path, modes):
    _REFERENCE_MODE_SOURCE[os.path.abspath(log_path)] = modes


def _check_mode_source(log_path, modes):
    recorded = _REFERENCE_MODE_SOURCE.get(os.path.abspath(log_path))
    if recorded is not None and recorded != modes:
        raise ValueError(
            "-ageig {0} uses --modes {1}, but -geigz read {0} with --modes {2}; give the "
            "same --modes on the RDAT and CDAT lines.".format(
                os.path.basename(log_path), modes, recorded))


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


def _wavenumber(eigenvalues):
    """Mass-weighted Hessian eigenvalues (kJ/mol/A^2/amu) as signed cm^-1."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return np.sign(eigenvalues) * co.EIGENVALUE_CONVERSION * np.sqrt(np.abs(eigenvalues))


def _archive_hessian(path):
    """The mass-weighted archive Hessian (kJ/mol/A^2/amu) of a Gaussian
    frequency log and the archive's atoms, whose geometry is its input
    orientation. Raises ValueError when the log has no archive Hessian."""
    log = utilities.GaussLog(path)
    try:
        log.read_archive()
    except Exception as e:
        raise ValueError("cannot read the archive of {}: {}".format(path, e))
    if not log.structures or log.structures[0].hess is None:
        raise ValueError("no Hessian in the archive of {}".format(path))
    struct = log.structures[0]
    hessian = struct.hess.copy()
    utilities.mass_weight_hessian(hessian, struct.atoms)
    return hessian, [a for a in struct.atoms if not a.is_dummy]


def _archive_modes(path):
    """Eigenvalues (kJ/mol/A^2/amu, ascending) and normal modes (one per row)
    of the archive Hessian of a Gaussian frequency log -- rigid-body motion
    projected out and the rest diagonalized, as Gaussian does for the modes
    it prints -- and the archive's atoms, whose geometry fixes their frame."""
    hessian, atoms = _archive_hessian(path)
    masses = [co.MASSES[a.element] for a in atoms]
    evals, evecs = math_util.vibrational_modes(hessian, _coordinates(atoms), masses)
    return evals, evecs, atoms


def _gauss_log_eigenmatrix(path, invert=None, modes="printed"):
    """
    The reference eigenmatrix: the eigenvalues of the mass-weighted QM
    Hessian in kJ/mol/A^2/amu on the diagonal of an otherwise zero matrix,
    emitted as Datum (typ='eig'). With modes="printed" they are read from the
    frequency section of the Gaussian log (force constant over reduced mass,
    per mode), as upstream q2mm's -geigz does; with modes="archive" they come
    from diagonalizing the archive Hessian. Mode 1 is the lowest, the
    transition-state mode; with `invert` its (negative) eigenvalue is
    replaced by that value. The source is recorded for the -ageig check.
    """
    _record_mode_source(path, modes)
    if modes == "archive":
        evals = _archive_modes(path)[0]
    else:
        evals = np.asarray(utilities.GaussLog(path).evals, dtype=float)
        if evals.size == 0:
            logger.warning("No normal modes in the Gaussian log: {}".format(path))
            return []
        evals = evals * co.HESSIAN_CONVERSION
    if invert is not None:
        evals = math_util.replace_lowest_eigenvalue(evals, invert, label=path)
    return datums_from_eigenmatrix(np.diag(evals), os.path.basename(path))


def reference_modes(log_path, mol2_path=None, modes="printed", allow_frame_mismatch=False):
    """
    The normalized, mass-weighted QM normal modes (n_modes x 3N) for
    projecting the calculated Hessian (-ageig), from the source --modes names:

    * "printed": the frequency section of the log. Without nosymm Gaussian
      prints them in its rotated standard orientation, and without
      freq=hpmodes to two decimals; both are warned about.
    * "archive": the archive Hessian diagonalized, at full precision and in
      the archive's (input) frame.

    The mol2 must be in the modes' frame; a mismatch stops the run, or only
    warns with allow_frame_mismatch. A self-check logs how well the chosen
    modes diagonalize the QM Hessian itself.
    """
    label = "-ageig " + os.path.basename(log_path)
    _check_mode_source(log_path, modes)
    if modes == "archive":
        evals, evecs, atoms = _archive_modes(log_path)
        _check_mode_frame(atoms, mol2_path, label, "the archive geometry of the log",
                          "Build the mol2 from the archive geometry, the frame of the "
                          "-gh Hessian.", allow_frame_mismatch)
        _compare_with_printed_modes(log_path, evals, label)
        return evecs
    return _printed_modes(log_path, mol2_path, label, allow_frame_mismatch)


def _printed_modes(log_path, mol2_path, label, allow_frame_mismatch):
    """reference_modes for --modes printed: the modes, the precision and
    nosymm warnings, the frame check and the self-check."""
    log = utilities.GaussLog(log_path)
    evecs = np.asarray(log.evecs, dtype=float)
    if evecs.size == 0:
        raise ValueError("No normal modes in the Gaussian log: {}".format(log_path))
    # Low-precision normal coordinates (no freq=hpmodes: two decimals) give
    # eigenvectors that are only roughly orthonormal, and that error goes
    # straight into every element of the projected Hessian.
    error = float(np.abs(evecs.dot(evecs.T) - np.eye(len(evecs))).max())
    if error > MODE_ORTHONORMALITY_TOLERANCE:
        logger.warning("{}: the normal modes were printed {}and are orthonormal only to "
                       "{:.1%}; rerun the frequency job with freq=hpmodes so the eigenvectors "
                       "are printed at full precision, or use --modes archive.".format(
                           label, "" if log.printed_hpmodes() else "without freq=hpmodes ",
                           error))
    try:
        printed = log.last_orientation()    # the frame the modes are printed in
    except Exception as e:   # the check must never take a run down
        logger.debug("Frame check skipped for {}: {}".format(label, e))
        printed = []
    try:
        archive_hessian, archive_atoms = _archive_hessian(log_path)
    except ValueError as e:
        logger.debug("{}: no archive Hessian to check against: {}".format(label, e))
        archive_hessian, archive_atoms = None, []
    comparable = bool(printed) and len(archive_atoms) == len(printed)
    # Without nosymm Gaussian prints the modes in its standard orientation,
    # rotated from the input orientation of the archive (and so of -gh).
    if comparable:
        rotated = _centered_deviation(_coordinates(archive_atoms),
                                      _coordinates(printed)) > FRAME_TOLERANCE
    else:
        rotated = not log.used_nosymm()
    if rotated:
        hint = ("The frequency job ran without nosymm, so Gaussian printed the normal modes "
                "in its rotated standard orientation, not in the input orientation of the "
                "archive (or of a mol2 built from it). Rerun the frequency job with nosymm "
                "freq=hpmodes, build the mol2 in the log's standard orientation, or use "
                "--modes archive.")
    else:
        hint = "Build the mol2 from the geometry the frequency job printed."
    deviation = _check_mode_frame(printed, mol2_path, label,
                                  "the geometry the log prints its normal modes in",
                                  hint, allow_frame_mismatch)
    if rotated and deviation is not None and deviation <= FRAME_TOLERANCE:
        logger.warning("{}: the frequency job ran without nosymm. {} matches the standard "
                       "orientation of the printed normal modes, but the archive Hessian "
                       "that -gh reads is in the input orientation, so this mol2 cannot "
                       "serve -gh as well; rerun the frequency job with nosymm, or use "
                       "--modes archive, to keep one frame.".format(
                           label, os.path.basename(mol2_path)))
    if comparable and archive_hessian is not None:
        rotation = math_util.kabsch_rotation(_coordinates(archive_atoms), _coordinates(printed))
        _self_check_modes(math_util.rotate_hessian(archive_hessian, rotation), evecs,
                          np.asarray(log.evals, dtype=float) * co.HESSIAN_CONVERSION, label)
    return evecs


def _self_check_modes(qm_hessian, evecs, evals, label):
    """Log how well the chosen modes diagonalize the QM Hessian they belong to
    (given in their frame). Exact modes give diag(evals); the score logged is
    what a force field reproducing the QM Hessian exactly would get on the
    eigenmode data -- the floor these modes put under the fit. Returns
    (largest frequency error in cm^-1, RMS coupling, floor), or None when the
    check cannot be made."""
    try:
        matrix = math_util.project_hessian(qm_hessian, evecs)
    except ValueError as e:
        logger.debug("{}: self-check skipped: {}".format(label, e))
        return None
    if len(evals) != len(matrix):
        logger.debug("{}: self-check skipped: {} eigenvalues for {} modes".format(
            label, len(evals), len(matrix)))
        return None
    diagonal = np.diag(matrix)
    coupling = (matrix - np.diag(diagonal))[np.tril_indices_from(matrix, -1)]
    worst = float(np.abs(_wavenumber(diagonal) - _wavenumber(evals)).max())
    rms_coupling = float(np.sqrt(np.mean(coupling ** 2))) if coupling.size else 0.0
    reference = datums_from_eigenmatrix(np.diag(evals), "selfcheck")
    score.import_weights(reference)
    floor = float(score.score_data(reference, datums_from_eigenmatrix(matrix, "selfcheck")))
    logger.log(20, "{}: self-check, the QM Hessian projected on these modes: largest "
               "frequency error {:.1f} cm^-1, RMS coupling {:.2f} kJ/mol/A^2/amu; a force "
               "field reproducing it exactly would score {:.4f} on the eigenmode data "
               "(0 for exact modes).".format(label, worst, rms_coupling, floor))
    return worst, rms_coupling, floor


def _compare_with_printed_modes(log_path, evals, label):
    """Log how the archive modes' frequencies compare with the ones Gaussian
    printed; sound logs agree up to masses and rounding. Returns the largest
    difference in cm^-1, or None when the log printed no comparable modes."""
    try:
        printed = np.asarray(utilities.GaussLog(log_path).evals, dtype=float) * co.HESSIAN_CONVERSION
    except Exception as e:   # the comparison must never take a run down
        logger.debug("{}: no printed frequencies to compare with: {}".format(label, e))
        return None
    if printed.size == 0:
        return None
    if printed.size != evals.size:
        logger.warning("{}: {} normal modes from the archive Hessian but {} printed in the "
                       "log.".format(label, evals.size, printed.size))
        return None
    difference = float(np.abs(_wavenumber(evals) - _wavenumber(printed)).max())
    logger.log(20, "{}: {} normal modes from the archive Hessian; their frequencies match the "
               "printed ones to {:.1f} cm^-1 (mode 1: {:.1f} here, {:.1f} printed).".format(
                   label, evals.size, difference,
                   float(_wavenumber(evals[0])), float(_wavenumber(printed[0]))))
    return difference


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
    ("geigz", "eig", lambda p, opts: _gauss_log_eigenmatrix(
        p, invert=opts.invert, modes=getattr(opts, "modes", "printed"))),
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

    modes = getattr(opts, "modes", "printed")
    # A -geigz in this same call is read after the calculators are built
    # (collect_data), so record its source now for the -ageig check.
    for filename in _flag_files(opts, "geigz"):
        _record_mode_source(_full_path(opts, filename), modes)
    calcs = []
    for key, commands in calculated.items():
        in_path = key[0]
        eigenvectors = None
        if key in mode_logs:
            eigenvectors = reference_modes(
                mode_logs[key], _mol2_of(in_path), modes=modes,
                allow_frame_mismatch=getattr(opts, "allow_frame_mismatch", False))
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
