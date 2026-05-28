#!/usr/bin/env python
"""
score
-----
Objective-function scoring for Q2MM (q2mm-amber-main).

Computes

    chi^2 = sum_i w_i^2 * (x_r_i - x_c_i)^2

over all matched reference (x_r) and calculated (x_c) data points,
normalised per data-type bucket. This module is the replacement for the
old compare.py from q2mm-master and is intended to be called by opt.py
optimizers (Gradient, SwarmOptimizer, Simplex) and by loop.py.

Public functions
----------------
data_by_type(data)              Bucket Datum objects by their .typ string.
trim_data(r_dict, c_dict)       Drop unmatched data points from both sides.
correlate_energies(r, c)        Zero each energy group to its minimum.
import_weights(data)            Stamp weights from constants.WEIGHTS onto data.
compare_data(r_dict, c_dict)    Run the objective function. Returns score (float).
"""
from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import logging
import logging.config

import numpy as np

import constants as co
import data_structs

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


def data_by_type(data_iterable):
    """
    Bucket Datum objects by their .typ string.

    Returns
    -------
    dict[str, list[Datum]]
    """
    buckets = {}
    for datum in data_iterable:
        buckets.setdefault(datum.typ, []).append(datum)
    return buckets


def trim_data(dict1, dict2):
    """
    Remove data points that appear in only one of the two dictionaries.
    Torsion labels are matched via the regex co.RE_T_LBL when available so
    that pre/opt structures from a single command file align properly.

    Modifies both dicts in place and also returns them.
    """
    for typ in list(dict1.keys()):
        if typ == "t" and hasattr(co, "RE_T_LBL"):
            to_remove = []
            for d1 in dict1[typ]:
                if not any(
                    co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d1.lbl)[1]
                    and co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d1.lbl)[2]
                    for x in dict2.get(typ, [])
                ):
                    to_remove.append(d1)
            for d2 in dict2.get(typ, []):
                if not any(
                    co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d2.lbl)[1]
                    and co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d2.lbl)[2]
                    for x in dict1[typ]
                ):
                    to_remove.append(d2)
            for datum in to_remove:
                if datum in dict1[typ]:
                    dict1[typ].remove(datum)
                if datum in dict2.get(typ, []):
                    dict2[typ].remove(datum)
            if to_remove:
                logger.log(20, ">>> Removed Data: {}".format(len(to_remove)))
        dict1[typ] = np.array(dict1[typ], dtype=data_structs.Datum)
        if typ in dict2:
            dict2[typ] = np.array(dict2[typ], dtype=data_structs.Datum)
    return dict1, dict2


def select_group_of_energies(data):
    """
    Yields numpy index arrays for each (energy-type, group) bucket.
    """
    for energy_type in ["e", "eo"]:
        indices = np.where([x.typ == energy_type for x in data])[0]
        unique_group_nums = set(x.idx_1 for x in np.asarray(data)[indices])
        for grp in unique_group_nums:
            more_indices = np.where(
                [x.typ == energy_type and x.idx_1 == grp for x in data]
            )[0]
            yield more_indices


def correlate_energies(r_data, c_data):
    """
    For each group of energies in c_data, find the minimum and zero both
    r_data and c_data energies relative to that conformer.
    """
    r_arr = np.asarray(r_data)
    c_arr = np.asarray(c_data)
    for indices in select_group_of_energies(c_arr):
        if len(indices) == 0:
            continue
        _, zero_local = min((x.val, i) for i, x in enumerate(r_arr[indices]))
        zero_ind = indices[zero_local]
        zero = c_arr[zero_ind].val
        for ind in indices:
            c_arr[ind].val -= zero


def import_weights(data):
    """
    Stamp the weight attribute on every Datum that doesn't already have one.
    Uses constants.WEIGHTS. Eigenvalue weights are handled specially:
      eig_i      first diagonal element
      eig_d_low  remaining diagonals with val < 1100
      eig_d_high remaining diagonals with val >= 1100
      eig_o      off-diagonals
    """
    for datum in data:
        if datum.wht is not None:
            continue
        if datum.typ == "eig":
            if datum.idx_1 == datum.idx_2 == 1:
                datum.wht = co.WEIGHTS["eig_i"]
            elif datum.idx_1 == datum.idx_2:
                datum.wht = (
                    co.WEIGHTS["eig_d_low"]
                    if datum.val < 1100
                    else co.WEIGHTS["eig_d_high"]
                )
            else:
                datum.wht = co.WEIGHTS["eig_o"]
        else:
            datum.wht = co.WEIGHTS[datum.typ]


def compare_data(r_dict, c_dict, output=None, doprint=False):
    """
    Compute chi^2 = sum w^2 * (x_r - x_c)^2, bucket-normalised.

    Energy buckets (e, eo, ea, eao) are normalised by the *total* energy
    count across all energy buckets (matches old compare.py behaviour).
    Hessian (h) buckets are normalised by len(c_dict[typ]). Everything
    else is normalised by len(r_dict[typ]).

    Parameters
    ----------
    r_dict, c_dict : dict[str, list[Datum]]
        Aligned reference and calculated buckets (use trim_data first).
    output : str | None
        Path to write a pretty comparison table.
    doprint : bool
        If True, also print the table to stdout.

    Returns
    -------
    float
        Total objective-function score.
    """
    strings = []
    strings.append(
        "--" + " Label ".ljust(30, "-")
        + "--" + " Weight ".center(7, "-")
        + "--" + " R. Value ".center(11, "-")
        + "--" + " C. Value ".center(11, "-")
        + "--" + " Score ".center(11, "-")
        + "--" + " Row " + "--"
    )

    score_typ = defaultdict(float)
    num_typ = defaultdict(int)
    score_tot = 0.0
    total_num = 0

    data_types = sorted(r_dict.keys())

    total_num_energy = sum(
        len(r_dict[typ]) for typ in data_types if typ in ("e", "eo", "ea", "eao")
    )

    for typ in data_types:
        if typ not in c_dict:
            continue
        total_num += len(r_dict[typ])
        if typ in ("e", "eo", "ea", "eao"):
            correlate_energies(r_dict[typ], c_dict[typ])
        # Stamp weights on BOTH reference and calculated datums. The
        # typ='h' branch below uses c.wht (q2mm-master convention where
        # the FF side carries per-contact weights like h12/h13/h14),
        # while other branches use r.wht -- so we need both to be set.
        import_weights(r_dict[typ])
        import_weights(c_dict[typ])

        for r, c in zip(r_dict[typ], c_dict[typ]):
            if c.typ == "t":
                diff = abs(r.val - c.val)
                if diff > 180.0:
                    diff = 360.0 - diff
            else:
                diff = r.val - c.val

            # disp_wht is the weight actually applied to this datum, so the
            # printed Weight column reflects the real scoring weight. For
            # typ='h' that's the calculated-side distance weight (c.wht);
            # everything else uses the reference-side weight (r.wht).
            if typ in ("e", "eo", "ea", "eao"):
                norm = total_num_energy if total_num_energy else 1
                disp_wht = r.wht
                score = (disp_wht ** 2 * diff ** 2) / norm
            elif typ == "h":
                norm = len(c_dict[typ]) if len(c_dict[typ]) else 1
                disp_wht = c.wht
                score = (disp_wht ** 2 * diff ** 2) / norm
            else:
                norm = len(r_dict[typ]) if len(r_dict[typ]) else 1
                disp_wht = r.wht
                score = (disp_wht ** 2 * diff ** 2) / norm

            score_tot += score
            score_typ[c.typ] += score
            num_typ[c.typ] += 1

            if c.typ == "eig":
                if c.idx_1 == c.idx_2:
                    if r.val < 1100:
                        score_typ[c.typ + "-d-low"] += score
                        num_typ[c.typ + "-d-low"] += 1
                    else:
                        score_typ[c.typ + "-d-high"] += score
                        num_typ[c.typ + "-d-high"] += 1
                else:
                    score_typ[c.typ + "-o"] += score
                    num_typ[c.typ + "-o"] += 1

            if c.ff_row is None:
                strings.append(
                    "  {:<30}  {:>7.2f}  {:>11.4f}  {:>11.4f}  {:>11.4f}  ".format(
                        c.lbl, disp_wht, r.val, c.val, score
                    )
                )
            else:
                strings.append(
                    "  {:<30}  {:>7.2f}  {:>11.4f}  {:>11.4f}  {:>11.4f}  {:>5} ".format(
                        c.lbl, disp_wht, r.val, c.val, score, c.ff_row
                    )
                )

    strings.append("-" * 89)
    strings.append("{:<20} {:20.4f}".format("Total score:", score_tot))
    strings.append("{:<30} {:10d}".format("Total Num. data points:", total_num))
    for k, v in num_typ.items():
        strings.append("{:<30} {:10d}".format(k + ":", v))
    strings.append("-" * 89)
    for k, v in score_typ.items():
        strings.append("{:<20} {:20.4f}".format(k + ":", v))

    if output:
        with open(output, "w") as f:
            for line in strings:
                f.write("{}\n".format(line))
    if doprint:
        for line in strings:
            print(line)

    return score_tot
