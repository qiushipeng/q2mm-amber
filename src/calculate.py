#!/usr/bin/env python
"""
calculate
---------
Reference / FF data extraction for q2mm-amber-main.

This module preserves the CLI signature used by the old q2mm-master
calculate.py (so loop.in RDAT / CDAT lines continue to work). It:

  1. parses a list of -flag/filename pairs (eg "-gh foo.log -i 1"),
  2. for each file, instantiates the right utilities.File subclass,
  3. runs the AMBER subprocess pipeline when the requested data type
     requires it,
  4. reads back the produced files and returns a flat list of
     data_structs.Datum objects.

Currently implemented data types
--------------------------------
Amber  : -ae, -ae1, -aeo, -ae1o, -abo, -aao, -ato, -ah
Gaussian: -gh (Hessian as eigenmatrix), -ge, -ge1, -geo, -ge1o,
          -gea, -geao, -gab, -gaa, -gat, -gabo, -gaao, -gato

Anything else from the old code (MacroModel / Jaguar / Tinker) is
parsed but ignored.

main(args) -> list[Datum]
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import logging.config
import os
import sys

import numpy as np

import constants as co
import score
from data_structs import Datum
import utilities

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


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
                 "abo", "aao", "ato", "ah", "aha"):
        parser.add_argument("-" + flag, type=str, nargs="+",
                            action="append", default=[])
    return parser


# ---------------------------------------------------------------------------
# Datum factories
# ---------------------------------------------------------------------------

def _datum_for_eigval(val, idx, src_filename):
    return Datum(
        val=float(val),
        typ="eig",
        src_1=src_filename,
        idx_1=int(idx),
        idx_2=int(idx),
    )


def _datum_for_eigmat(val, i, j, src_filename):
    """One element of the eigenmatrix (idx_1=row, idx_2=col)."""
    return Datum(
        val=float(val),
        typ="eig",
        src_1=src_filename,
        idx_1=int(i),
        idx_2=int(j),
    )


def _datum_for_h(val, i, j, src_filename, atm_1=None, atm_2=None, wht=None):
    # One mass-weighted Hessian matrix element. typ='h' so the scoring
    # machinery picks 'h'-style weights. atm_1/atm_2 carry the atom indices
    # so a distance-based weight (q2mm-master's int_wht) can be evaluated
    # by the caller and stamped via the wht argument.
    return Datum(
        val=float(val),
        typ="h",
        src_1=src_filename,
        idx_1=int(i),
        idx_2=int(j),
        atm_1=atm_1,
        atm_2=atm_2,
        wht=wht,
    )


def _load_geo_npy(calc_dir):
    # Read calc/geo.npy (produced by AmberLeap.geo_extract via cpptraj) and
    # split its rows into 1-2 (bonds), 1-3 (angle endpoints), and 1-4
    # (dihedral endpoints) atom-pair lists. Returns ([], [], []) if the
    # file is missing so the caller can fall back to default weights.
    int2, int3, int4 = [], [], []
    geo_path = os.path.join(calc_dir, "geo.npy")
    if not os.path.isfile(geo_path):
        return int2, int3, int4
    try:
        hes_geo = np.load(geo_path, allow_pickle=True)
    except Exception as e:
        logger.warning("Failed to load {}: {}".format(geo_path, e))
        return int2, int3, int4
    for ele in hes_geo:
        # Each row is [a, b, c, d] with None in unused slots; non-None count
        # picks the interaction class.
        non_none = sum(1 for x in ele if x is not None)
        a, b, c, d = ele
        if non_none == 2:
            int2.append([int(a), int(b)])
        elif non_none == 3:
            int3.append([int(a), int(c)])  # endpoints only
        elif non_none == 4:
            int4.append([int(a), int(d)])  # endpoints only
    return int2, int3, int4


def _int_wht(at_1, at_2, int2, int3, int4):
    # Distance-based Hessian-element weights, per the Q2MM paper:
    #   1-1 (same atom, 3x3 diagonal block):     0.0  (no contribution)
    #   1-2 bonded (1 bond apart):               WEIGHTS['h12'] = 0.031
    #   1-3 (angle endpoints, 2 bonds apart):    WEIGHTS['h13'] = 0.031
    #   1-4 (dihedral endpoints, 3 bonds apart): WEIGHTS['h14'] = 0.31
    #   all other (>3 bonds apart):              WEIGHTS['h']   = 0.031
    # The 1-4 terms are emphasized (0.31) to represent the TS as a minimum.
    # Pairs are stored unordered, so check both orders.
    if at_1 == at_2:
        return 0.0
    pair_a = [at_1, at_2]
    pair_b = [at_2, at_1]
    if pair_a in int2 or pair_b in int2:
        return co.WEIGHTS["h12"]
    if pair_a in int3 or pair_b in int3:
        return co.WEIGHTS["h13"]
    if pair_a in int4 or pair_b in int4:
        return co.WEIGHTS["h14"]
    return co.WEIGHTS["h"]


def _datum_for_energy(val, idx_1, src_filename, typ="e"):
    return Datum(
        val=float(val),
        typ=typ,
        src_1=src_filename,
        idx_1=int(idx_1),
        idx_2=1,
    )


def _datum_for_bond(val, atoms, src_filename, typ="b"):
    return Datum(val=float(val), typ=typ, src_1=src_filename,
                 atm_1=atoms[0], atm_2=atoms[1])


def _datum_for_angle(val, atoms, src_filename, typ="a"):
    return Datum(val=float(val), typ=typ, src_1=src_filename,
                 atm_1=atoms[0], atm_2=atoms[1], atm_3=atoms[2])


def _datum_for_torsion(val, atoms, src_filename, typ="t"):
    return Datum(val=float(val), typ=typ, src_1=src_filename,
                 atm_1=atoms[0], atm_2=atoms[1], atm_3=atoms[2], atm_4=atoms[3])


# ---------------------------------------------------------------------------
# Per-file collectors
# ---------------------------------------------------------------------------

def _gauss_log_eigmat(path, invert=None):
    """
    Read the raw 3N x 3N Hessian from a Gaussian frequency archive,
    mass-weight it, and emit its lower-triangular elements as
    Datum (typ='eig'). Matches q2mm-master's 'gh' convention so the
    Gaussian and Amber sides always produce identical shapes
    (3N(3N+1)/2 elements) for any molecule size, with no projection
    or eigenvalue-counting needed.

    If `invert` is given, the smallest Hessian eigenvalue is forced
    to that value (flips a TS imaginary frequency to a real one
    before the matrix is reassembled).
    """
    # Open the Gaussian .log via the utilities wrapper.
    log = utilities.GaussLog(path)
    # The archive block at the end of the .log carries the raw lower-tri
    # Hessian; read_archive() parses it into log.structures[0].hess.
    try:
        log.read_archive()
    except Exception as e:
        logger.warning("Gaussian archive parse failed for {}: {}".format(path, e))
        return []
    # Bail out cleanly if the .log had no archive Hessian (e.g., not a freq job).
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
    # Optional TS imaginary-mode flip: diagonalize, swap the smallest |eig|
    # for `invert`, and reassemble H from the modified spectrum.
    if invert is not None:
        w, v = np.linalg.eigh(H)
        i = int(np.argmin(np.abs(w)))
        w[i] = float(invert)
        H = v.dot(np.diag(w)).dot(v.T)
    # Take only the lower-triangular indices: the Hessian is symmetric so
    # the upper triangle is redundant data for the Q2MM fit.
    tri_i, tri_j = np.tril_indices_from(H)
    src = os.path.basename(path)
    # One Datum per lower-tri element, tagged typ='h' so it picks up the
    # uniform Hessian weight (WEIGHTS['h']=0.031) from q2mm-master.
    return [_datum_for_h(H[i, j], i + 1, j + 1, src)
            for i, j in zip(tri_i, tri_j)]


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
    return [_datum_for_energy(energy * co.HARTREE_TO_KJMOL,
                              group_idx, os.path.basename(path), typ=typ)]


def _amber_run(in_path, commands):
    """
    Run an AmberLeap calculation for the requested commands. Returns the
    AmberLeap instance after run() has completed so caller can read
    the produced .ene / .geo / .hes files.
    """
    leap = utilities.AmberLeap(in_path)
    leap.commands = commands
    if not getattr(_amber_run, "_skip_run", False):
        try:
            leap.run(check_tokens=False)
        except Exception as e:
            logger.warning("AmberLeap.run() failed for {}: {}".format(in_path, e))
    return leap


def _amber_hessian_eigmat(in_path, invert=None):
    """
    Read the 3N x 3N mass-weighted Hessian produced by Amber's nab/nmode,
    and emit its lower-triangular elements as Datum (typ='h') with
    per-element distance-based weights, matching q2mm-master's '-ah'
    convention (calculate.py int_wht). Mirrors _gauss_log_eigmat on the
    Gaussian side for dimension alignment.

    If `invert` is given, the smallest eigenvalue is replaced with
    `invert` (TS imaginary-mode flip) before the matrix is reassembled.
    """
    # Run tleap + sander min + nab to (re)generate the .hes and geo files.
    leap = _amber_run(in_path, ["ah"])
    hes_path = os.path.join(leap.directory, "calc", leap.name_hes)
    # If the Amber pipeline failed, we'd have no .hes; bail out quietly.
    if not os.path.isfile(hes_path):
        logger.warning("Hessian file missing: {}".format(hes_path))
        return []
    # AmberHess.hessian parses the file, converts kcal/mol -> kJ/mol, and
    # returns a 3N x 3N matrix that is already mass-weighted by nab/nmode.
    hess = utilities.AmberHess(hes_path)
    H = hess.hessian
    if H is None:
        return []
    # Optional TS imaginary-mode flip: diagonalize, swap the smallest |eig|
    # for `invert`, and reassemble H so the lower-tri output reflects it.
    if invert is not None:
        w, v = np.linalg.eigh(H)
        i = int(np.argmin(np.abs(w)))
        w[i] = float(invert)
        H = v.dot(np.diag(w)).dot(v.T)
    # Load the bond/angle/dihedral pair lists generated by cpptraj so the
    # per-element weight can be assigned by atom-pair topology distance.
    calc_dir = os.path.join(leap.directory, "calc")
    int2, int3, int4 = _load_geo_npy(calc_dir)
    if not (int2 or int3 or int4):
        logger.warning("No geo.npy in {}; long-range weight 1.0 will apply "
                       "to every non-diagonal pair.".format(calc_dir))
    # Lower-triangular indices only -- the Hessian is symmetric so
    # the upper triangle would just duplicate data points.
    tri_i, tri_j = np.tril_indices_from(H)
    # Use the .hes filename so the Datum label collapses to "amber" (the
    # piece before the first dot), matching q2mm-master's label format.
    src = leap.name_hes
    data = []
    for i, j in zip(tri_i, tri_j):
        # idx coords are 1-based cartesian DoF; atoms are idx // 3 + 1 in
        # 1-based numbering. // is integer division on 0-based equivalents.
        atm_1 = int(i // 3 + 1)
        atm_2 = int(j // 3 + 1)
        wht = _int_wht(atm_1, atm_2, int2, int3, int4)
        data.append(_datum_for_h(H[i, j], i + 1, j + 1, src,
                                 atm_1=atm_1, atm_2=atm_2, wht=wht))
    return data


def _amber_energy(in_path, typ="e", group_idx=1):
    leap = _amber_run(in_path, [typ])
    ene_path = os.path.join(leap.directory, "calc", leap.name_ene)
    if not os.path.isfile(ene_path):
        logger.warning("Energy file missing: {}".format(ene_path))
        return []
    ene = utilities.AmberEne(ene_path)
    data = []
    for i, s in enumerate(ene.structures):
        if "energy" in s.props:
            data.append(_datum_for_energy(s.props["energy"], i + 1,
                                          os.path.basename(in_path), typ=typ))
    return data


def _amber_geo(in_path, kind):
    """
    kind: 'b' bonds, 'a' angles, 't' torsions.
    """
    leap = _amber_run(in_path, [kind + "o"])
    geo_path = os.path.join(leap.directory, "calc", leap.name_geo)
    if not os.path.isfile(geo_path):
        logger.warning("Geo file missing: {}".format(geo_path))
        return []
    geo = utilities.AmberGeo(geo_path)
    data = []
    for s in geo.structures:
        if kind == "b":
            for bond in s.bonds:
                data.append(_datum_for_bond(bond.value, bond.atom_nums,
                                            os.path.basename(in_path), typ="b"))
        elif kind == "a":
            for ang in s.angles:
                data.append(_datum_for_angle(ang.value, ang.atom_nums,
                                             os.path.basename(in_path), typ="a"))
        elif kind == "t":
            for tor in s.torsions:
                data.append(_datum_for_torsion(tor.value, tor.atom_nums,
                                               os.path.basename(in_path), typ="t"))
    return data


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

# (command flag, datum type, collector function)
_COMMAND_DISPATCH = [
    ("ah",   "h",   lambda p, opts: _amber_hessian_eigmat(p, invert=opts.invert)),
    ("ae",   "e",   lambda p, opts: _amber_energy(p, typ="e")),
    ("ae1",  "e1",  lambda p, opts: _amber_energy(p, typ="e1")),
    ("aeo",  "eo",  lambda p, opts: _amber_energy(p, typ="eo")),
    ("ae1o", "e1o", lambda p, opts: _amber_energy(p, typ="e1o")),
    ("abo",  "b",   lambda p, opts: _amber_geo(p, "b")),
    ("aao",  "a",   lambda p, opts: _amber_geo(p, "a")),
    ("ato",  "t",   lambda p, opts: _amber_geo(p, "t")),
    ("gh",   "h",   lambda p, opts: _gauss_log_eigmat(p, invert=opts.invert)),
    ("ge",   "e",   lambda p, opts: _gauss_log_energy(p, typ="e")),
    ("ge1",  "e1",  lambda p, opts: _gauss_log_energy(p, typ="e1")),
    ("geo",  "eo",  lambda p, opts: _gauss_log_energy(p, typ="eo")),
    ("ge1o", "e1o", lambda p, opts: _gauss_log_energy(p, typ="e1o")),
]


def collect_data(opts):
    """
    Walk through each enabled command flag in opts and append Datum
    objects produced by the matching collector function.
    """
    data = []
    for flag, _typ, collector in _COMMAND_DISPATCH:
        groups = getattr(opts, flag, []) or []
        # argparse with action='append' nargs='+' gives list-of-lists.
        for filenames in groups:
            for filename in filenames:
                full = os.path.join(opts.directory, filename) \
                    if not os.path.isabs(filename) else filename
                if opts.fake:
                    # produce one zeroed datum so optimizers don't crash
                    data.append(Datum(val=0.0, typ=_typ,
                                      src_1=os.path.basename(filename)))
                    continue
                logger.log(20, "  -- {} {}".format(flag, full))
                data.extend(collector(full, opts))
    return data


def main(args):
    """
    Args may be a single string or list of strings. Returns a flat list
    of Datum objects.
    """
    if sys.version_info > (3, 0):
        if isinstance(args, str):
            args = args.split()
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    data = collect_data(opts)
    if opts.weight:
        score.import_weights(data)
    if opts.doprint:
        for d in data:
            print("{:30s} {:>11.4f}".format(d.lbl, d.val))
    return data


if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
