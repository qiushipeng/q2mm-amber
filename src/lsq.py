#!/usr/bin/env python
"""
lsq
---
Exact bounded linear least-squares fit of force constants: the loop.in LSQ
command.

nab computes the Amber Hessian at the fixed geometry of <name>.pdb, so the
Hessian is linear in the bond and angle force constants and in the barriers
of the dihedrals and impropers. The -ageig eigenmatrix is that Hessian
projected on fixed QM modes, so it is linear in them too. The weighted
residuals that score.score_data squares and sums are therefore linear in
these parameters k,

    r(k) = r(k0) + J (k - k0),

and the score ||r(k)||^2 is a convex quadratic. LSQ measures J with one
Amber evaluation per parameter (forward differences, which are exact for
linear data), finds the minimum within the PARM bounds with
scipy.optimize.lsq_linear (method "bvls"), and evaluates the solution once
with Amber: N + 2 evaluations for N parameters.

Guards: LSQ fits only the parameter types in LINEAR_PTYPES, against only the
CDAT commands in LINEAR_COMMANDS. Anything else stops the run with an error.
A parameter that has no effect on the data keeps its start value.

Report: root.log, and lsq_NNN.csv in the run directory, one row per
parameter. With --n_processes > 1 the evaluations run in parallel in
<DIR>/lsq_particles/p_NNN.
"""
from __future__ import absolute_import
from __future__ import division

import copy
import csv
import glob
import logging
import logging.config
import os

import numpy as np
from scipy.optimize import lsq_linear

import constants as co
import opt
import score
from utilities import AmberUtilities

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# Parameter types that enter the fixed-geometry Hessian linearly: bond and
# angle force constants, dihedral and improper barriers.
LINEAR_PTYPES = ("bf", "af", "df", "imp1")
# CDAT commands whose data are linear in these parameters: the Hessian and
# the eigenmatrix, both at the QM geometry.
LINEAR_COMMANDS = ("ah", "ageig")
# Forward-difference step: this fraction of the value, and at least MIN_STEP.
STEP_FRACTION = 0.1
MIN_STEP = 1.0
# The largest relative difference between the Amber score of the solution
# and the score the linear model predicts that still counts as agreement.
LINEARITY_TOLERANCE = 1e-6
# How many of the worst-determined parameter combinations the log lists.
N_WEAKEST = 3
CSV_COLUMNS = ["ff_row", "ff_col", "ptype", "atom_types", "start", "solution",
               "lower", "upper", "status", "column_norm"]

# The reference data for the residual reducer. LeastSquares.run sets it
# before the calculator starts its pool, so forked workers inherit it (the
# same pattern as opt._REF_DATA for HYBR).
_REF_DATA = None


def _residuals(data):
    """Reducer for Calculator.evaluate_many: one force field's data -> its weighted residuals."""
    return score.weighted_residuals(_REF_DATA, data)


def frcmod_value(value, ptype):
    """`value` as the frcmod stores it: rounded to the decimals render_frcmod writes."""
    return float("{:.{}f}".format(value, AmberUtilities.FRCMOD_DECIMALS[ptype]))


def label(param):
    """A parameter as the log names it, e.g. 'bf c3-hc (row 10)'."""
    return "{} {} (row {})".format(param.ptype, "-".join(param.atom_types), param.ff_row)


def bounds(params):
    """The lower and upper bound of each parameter: its PARM range, limited
    to the values the frcmod can hold."""
    cap = AmberUtilities.MAX_FRCMOD_VALUE
    lower = np.array([max(p.allowed_range[0], -cap) for p in params], dtype=float)
    upper = np.array([min(p.allowed_range[1], cap) for p in params], dtype=float)
    return lower, upper


def steps(params, start, lower, upper):
    """
    The forward-difference step of each parameter: STEP_FRACTION of its
    start value and at least MIN_STEP, upward unless that crosses the upper
    bound, then downward, else as far as the bounds allow. The step is
    rounded so that start + step is a value the frcmod stores exactly.
    Raises ValueError for a parameter whose start value lies outside its
    bounds, or whose bounds leave no room to move.
    """
    out = []
    for param, value, lo, hi in zip(params, start, lower, upper):
        if not lo <= value <= hi:
            raise ValueError("LSQ: {} starts at {:g}, outside [{:g}, {:g}] (the frcmod holds at "
                             "most {:g}).".format(label(param), value, lo, hi,
                                                  AmberUtilities.MAX_FRCMOD_VALUE))
        step = max(STEP_FRACTION * abs(value), MIN_STEP)
        if value + step > hi:
            if value - step >= lo:
                step = -step
            elif hi - value >= value - lo:
                step = hi - value
            else:
                step = lo - value
        step = frcmod_value(value + step, param.ptype) - value
        if step == 0.0:
            raise ValueError("LSQ: {} cannot move within its bounds [{:g}, {:g}] (the frcmod "
                             "holds at most {:g}). Widen its PARM bounds or remove it from the "
                             "PARM file.".format(label(param), lo, hi,
                                                 AmberUtilities.MAX_FRCMOD_VALUE))
        out.append(step)
    return np.array(out)


def weakest_combinations(matrix, params, n=N_WEAKEST):
    """
    The n worst-determined parameter combinations of a least-squares matrix
    whose columns have unit norm: (relative singular value, [(coefficient,
    parameter), ...]) for the smallest singular values, with the largest
    coefficients of each combination first (at most four).
    """
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    out = []
    for i in range(len(singular) - 1, max(len(singular) - 1 - n, -1), -1):
        order = np.argsort(-np.abs(vt[i]))[:4]
        out.append((singular[i] / singular[0], [(vt[i][j], params[j]) for j in order]))
    return out


class LeastSquares(opt.Optimizer):
    """
    One exact bounded least-squares fit of the parameters in self.ff.params
    (see the module docstring).

    n_processes : int
        Workers for the N + 1 evaluations of the linear model; 1 runs them in
        turn in the run directory.
    """

    def __init__(self, direc=None, ff=None, ff_lines=None, args_ff=None,
                 args_ref=None, calculator=None, n_processes=1):
        super(LeastSquares, self).__init__(
            direc=direc, ff=ff, ff_lines=ff_lines, args_ff=args_ff,
            args_ref=args_ref, calculator=calculator)
        self.n_processes = n_processes

    def check_linear(self):
        """Raise ValueError unless every parameter and every CDAT command is one LSQ fits exactly."""
        nonlinear = [label(p) for p in self.ff.params if p.ptype not in LINEAR_PTYPES]
        if nonlinear:
            raise ValueError(
                "LSQ fits only force constants and barriers (parameter types {}); the data "
                "are not linear in {}. Fit these with GRAD, SIMP or HYBR.".format(
                    ", ".join(LINEAR_PTYPES), ", ".join(nonlinear)))
        commands = sorted(set(c for c in self.calculator.commands if c not in LINEAR_COMMANDS))
        if commands:
            raise ValueError(
                "LSQ needs data that are linear in the parameters (CDAT {}), not {}. "
                "Fit these data with GRAD, SIMP or HYBR.".format(
                    " and ".join("-" + c for c in LINEAR_COMMANDS),
                    ", ".join("-" + c for c in commands)))

    @opt.catch_run_errors
    def run(self, ref_data=None):
        """
        Build the linear model, solve it within the bounds and check the
        solution with Amber. Returns the solution when Amber scores it below
        the start, else self.ff; the calculator writes the returned
        parameters to disk either way.
        """
        if ref_data is None:
            ref_data = opt.return_ref_data(self.args_ref)
        self.check_linear()
        params = self.ff.params
        n = len(params)
        lower, upper = bounds(params)
        # Work with the values the frcmod stores, so the differences are
        # exactly the changes Amber sees.
        start = np.array([frcmod_value(p.value, p.ptype) for p in params])
        delta = steps(params, start, lower, upper)

        logger.log(20, "~~ LEAST-SQUARES FIT ~~".rjust(79, "~"))
        logger.log(20, "  -- {} parameter(s): {} Amber evaluations for the linear model, "
                       "then 1 to check the solution.".format(n, n + 1))
        ffs = [opt.trial_ff(self.ff, start)]
        for j in range(n):
            values = start.copy()
            values[j] += delta[j]
            ffs.append(opt.trial_ff(self.ff, values))
        global _REF_DATA
        _REF_DATA = ref_data
        vectors = self.calculator.evaluate_many(
            ffs, reducer=_residuals, pool_dir=os.path.join(self.direc, "lsq_particles"),
            n_processes=self.n_processes)
        failed = [i for i, v in enumerate(vectors) if v is None]
        if failed:
            raise opt.OptError("LSQ: {} of the {} evaluations failed or gave incomplete data; "
                               "the force field is left unchanged.".format(len(failed), n + 1))
        r0 = vectors[0]
        jacobian = np.column_stack([(v - r0) / d for v, d in zip(vectors[1:], delta)])
        start_score = float(r0 @ r0)

        # A parameter with no effect on the data keeps its start value.
        column_norm = np.linalg.norm(jacobian, axis=0)
        free = column_norm > 0.0
        solution = start.copy()
        status = ["no effect" for _ in params]
        result = matrix = None
        if free.any():
            # Columns scaled to unit norm, for the solver's conditioning.
            scale = column_norm[free]
            matrix = jacobian[:, free] / scale
            result = lsq_linear(matrix, -r0,
                                bounds=((lower[free] - start[free]) * scale,
                                        (upper[free] - start[free]) * scale),
                                method="bvls")
            solution[free] = start[free] + result.x / scale
            names = {-1: "at lower bound", 0: "free", 1: "at upper bound"}
            for j, active in zip(np.flatnonzero(free), result.active_mask):
                status[j] = names[int(active)]
        solution = np.clip([frcmod_value(v, p.ptype) for v, p in zip(solution, params)],
                           lower, upper)
        predicted = float(np.sum((r0 + jacobian @ (solution - start)) ** 2))

        best = copy.deepcopy(self.ff)
        best.set_param_values(solution)
        best.method = "LSQ"
        best.data = self.calculator.evaluate(best)
        best.score = score.score_data(ref_data, best.data)

        self.report(params, start, solution, lower, upper, status, column_norm)
        self.log_results(params, free, result, matrix, start_score, predicted, best.score, status)
        if best.score < start_score:
            logger.log(20, "~~ LSQ FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
            self.best_ff = best
            self.calculator.update_ff(best)
            return best
        logger.log(20, "~~ LSQ FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
        self.calculator.update_ff(self.ff)
        return self.ff

    def report(self, params, start, solution, lower, upper, status, column_norm):
        """Write lsq_NNN.csv, numbered on from those already in the run directory."""
        existing = sorted(glob.glob(os.path.join(self.direc, "lsq_???.csv")))
        num = int(os.path.basename(existing[-1])[4:7]) + 1 if existing else 1
        path = os.path.join(self.direc, "lsq_{:03d}.csv".format(num))
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for j, p in enumerate(params):
                writer.writerow([p.ff_row, p.ff_col, p.ptype, "-".join(p.atom_types),
                                 start[j], solution[j], lower[j], upper[j], status[j],
                                 "{:.6g}".format(column_norm[j])])
        logger.log(20, "  -- Wrote {}".format(path))

    def log_results(self, params, free, result, matrix, start_score, predicted, actual, status):
        """The summary in root.log: scores, the agreement check, bounds and conditioning."""
        logger.log(20, "  -- Start score (Amber):       {:.10g}".format(start_score))
        logger.log(20, "  -- Predicted score:           {:.10g}".format(predicted))
        logger.log(20, "  -- Score of the solution (Amber): {:.10g}".format(actual))
        scale = max(abs(actual), abs(predicted))
        difference = abs(actual - predicted) / scale if scale > 0.0 else 0.0
        if not np.isfinite(actual):
            logger.warning("LSQ: Amber could not score the solution (the evaluation failed or "
                           "gave incomplete data). The force field is left unchanged.")
        elif difference > LINEARITY_TOLERANCE:
            logger.warning(
                "LSQ: Amber scored the solution {:.10g}, but the linear model predicted "
                "{:.10g} (relative difference {:.2e}, tolerance {:g}). The data are not linear "
                "in the parameters as LSQ assumes, or an evaluation went wrong. The solution "
                "is kept only if Amber scores it below the start.".format(
                    actual, predicted, difference, LINEARITY_TOLERANCE))
        else:
            logger.log(20, "  -- The Amber score agrees with the prediction (relative "
                           "difference {:.2e}).".format(difference))
        for label_text, name in (("No effect on the data; kept at start", "no effect"),
                                 ("At the lower bound", "at lower bound"),
                                 ("At the upper bound", "at upper bound")):
            chosen = [label(p) for p, s in zip(params, status) if s == name]
            if chosen:
                logger.log(20, "  -- {}: {}".format(label_text, ", ".join(chosen)))
        if result is None:
            return
        logger.log(20, "  -- Solver: {} First-order optimality {:.2e} (0 at the exact "
                       "optimum).".format(result.message, result.optimality))
        fitted = [p for p, f in zip(params, free) if f]
        combinations = weakest_combinations(matrix, fitted)
        logger.log(20, "  -- Condition number {:.3g}. Worst-determined combinations of the "
                       "parameters (each scaled by its column norm):".format(
                           1.0 / combinations[0][0] if combinations[0][0] > 0 else float("inf")))
        for relative, terms in combinations:
            logger.log(20, "       {:.2e}: {}".format(relative, "  ".join(
                "{:+.2f} {}".format(c, label(p)) for c, p in terms)))
