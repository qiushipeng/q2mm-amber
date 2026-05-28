#!/usr/bin/env python
"""
gradient
--------
Gradient-based parameter optimization for q2mm-amber-main.

Port of q2mm-master/q2mm/gradient.py adapted to use:
    data_structs.* (was datatypes.*)
    score.*        (was compare.*)
    calculate.main (returns Datum list)
    opt.*          helper functions

The Gradient class is configured via the loop.in GRAD command, e.g.:
    GRAD lstsq=True,radii=[1./10.]
    GRAD newton=True,cutoffs=[0.1/10.]
    GRAD lagrange=True,factor=[0.01/0.1/1./10.]
    GRAD levenberg=True,factor=[0.01/0.1/1./10.]
    GRAD svd=True,factor=[0.001/0.01/0.1/1.]
    GRAD                         (uses defaults: lagrange + newton)
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

import calculate
import constants as co
import data_structs
import opt
import score

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


def _score_weight(r, c):
    # Return the weight that score.compare_data actually applies to this
    # datum pair, so the optimizer's Jacobian and residual are built with
    # the same weights it is judged by. For typ='h' the score uses the
    # CALCULATED side's weight (Amber int_wht distance weights); every
    # other type uses the REFERENCE side's weight.
    return c.wht if c.typ == "h" else r.wht


class Gradient(opt.Optimizer):
    """
    Gradient-based optimization. Each enabled method (Newton, Lagrange,
    Levenberg, SVD, least-squares) produces one or more trial FFs; the
    best trial replaces the input FF if it improves the score.

    Attributes (radii / cutoffs / factors)
    --------------------------------------
    do_lagrange, do_levenberg, do_lstsq, do_newton, do_svd : bool
        Enable flags for each method.
    *_radii : list[float] or None
        If set, scale trial changes so the unsigned change vector's
        radius does not exceed each radius in the list.
    *_cutoffs : list[float] or None
        If set, reject trials whose change-vector radius falls outside
        [min(cutoffs), max(cutoffs)].
    *_factors : list[float] or None
        Method-specific damping/regularisation strengths.
    """

    def __init__(self, direc=None, ff=None, ff_lines=None,
                 args_ff=None, args_ref=None):
        super(Gradient, self).__init__(
            direc=direc, ff=ff, ff_lines=ff_lines,
            args_ff=args_ff, args_ref=args_ref)

        # method enables
        self.do_lstsq = False
        self.do_lagrange = True
        self.do_levenberg = False
        self.do_newton = True
        self.do_svd = False

        # method-specific config
        self.lstsq_cutoffs = None
        self.lstsq_radii = [1.0, 10.0]
        self.lagrange_factors = [0.01, 0.1, 1.0, 10.0]
        self.lagrange_cutoffs = None
        self.lagrange_radii = [0.1, 10.0]
        self.levenberg_factors = [0.01, 0.1, 1.0, 10.0]
        self.levenberg_cutoffs = None
        self.levenberg_radii = [0.1, 10.0]
        self.newton_cutoffs = None
        self.newton_radii = [1.0, 10.0]
        self.svd_factors = [0.001, 0.01, 0.1, 1.0, 10.0]
        self.svd_cutoffs = [0.1, 10.0]
        self.svd_radii = None

    @property
    def best_ff(self):
        if not self.new_ffs:
            return None
        return sorted(self.new_ffs, key=lambda x: x.score)[0]

    @best_ff.setter
    def best_ff(self, value):
        # Optimizer.__init__ assigns None; allow it. The real "best" comes
        # from the @property reading new_ffs.
        if value is not None:
            self.new_ffs.append(value)

    @opt.catch_run_errors
    def run(self, ref_data=None, restart=None):
        if ref_data is None:
            ref_data = opt.return_ref_data(self.args_ref)

        if self.ff.data is None:
            logger.log(20, "~~ GATHERING INITIAL FF DATA ~~".rjust(79, "~"))
            self.ff.export_ff()
            self.ff.data = calculate.main(self.args_ff)
            score.correlate_energies(ref_data, self.ff.data)

        r_dict = score.data_by_type(ref_data)
        c_dict = score.data_by_type(self.ff.data)
        r_dict, c_dict = score.trim_data(r_dict, c_dict)
        # Stamp weights on both sides so the optimizer can read the same
        # per-datum weight that scoring uses. import_weights only fills
        # None, so the Amber-side distance weights (int_wht) are preserved.
        for dt in r_dict:
            score.import_weights(r_dict[dt])
            if dt in c_dict:
                score.import_weights(c_dict[dt])
        if self.ff.score is None:
            self.ff.score = score.compare_data(r_dict, c_dict)

        data_types = sorted(r_dict.keys())

        logger.log(20, "~~ GRADIENT OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        # --- central differentiation -------------------------------------
        logger.log(20, "~~ CENTRAL DIFFERENTIATION ~~".rjust(79, "~"))
        if restart:
            par_file = restart
            logger.log(20, "  -- Restarting from {}.".format(par_file))
        else:
            par_files = glob.glob(os.path.join(self.direc, "par_diff_???.txt"))
            if par_files:
                par_files.sort()
                num = int(os.path.basename(par_files[-1])[9:12]) + 1
                par_file = "par_diff_{:03d}.txt".format(num)
            else:
                par_file = "par_diff_001.txt"
            par_path = os.path.join(self.direc, par_file)
            logger.log(20, "  -- Generating {}.".format(par_path))

            f = open(par_path, "w")
            csv_writer = csv.writer(f)
            writerows = [[], [], [], []]
            for dt in data_types:
                writerows[0].extend([x.lbl for x in r_dict[dt]])
                # Row 1 = weights the Jacobian will use. Use the scoring
                # weight (c.wht for 'h') so the optimizer and scorer agree.
                writerows[1].extend([_score_weight(r, c)
                                     for r, c in zip(r_dict[dt], c_dict[dt])])
                writerows[2].extend([x.val for x in r_dict[dt]])
                writerows[3].extend([x.val for x in c_dict[dt]])
            for row in writerows:
                csv_writer.writerow(row)

            logger.log(20, "~~ DIFFERENTIATING PARAMETERS ~~".rjust(79, "~"))
            ffs = opt.differentiate_ff(self.ff)
            logger.log(20, "~~ SCORING DIFFERENTIATED PARAMETERS ~~".rjust(79, "~"))
            for ff in ffs:
                ff.export_ff(lines=self.ff.lines)
                logger.log(20, "  -- Calculating {}.".format(ff))
                data = calculate.main(self.args_ff)
                c_data = score.data_by_type(data)
                r_dict, c_data = score.trim_data(r_dict, c_data)
                ff.score = score.compare_data(r_dict, c_data)
                opt.pretty_ff_results(ff)
                row = []
                for dt in data_types:
                    row.extend([x.val for x in c_data[dt]])
                csv_writer.writerow(row)
            f.close()
            opt.param_derivs(self.ff, ffs)

        # --- Jacobian / residual -----------------------------------------
        ma = vb = jacob = resid = None
        if self.do_lstsq or self.do_lagrange or self.do_levenberg or self.do_svd:
            logger.log(20, "~~ JACOBIAN AND RESIDUAL VECTOR ~~".rjust(79, "~"))
            num_d = sum(len(r_dict[dt]) for dt in r_dict)
            resid = np.empty((num_d, 1), dtype=float)
            count = 0
            for dt in data_types:
                for r, c in zip(r_dict[dt], c_dict[dt]):
                    # Use the same weight the scorer applies (c.wht for 'h')
                    # so the residual the optimizer minimizes matches the
                    # objective being scored.
                    resid[count, 0] = _score_weight(r, c) * (r.val - c.val)
                    count += 1
            logger.log(20, "  -- Residual vector shape: {}".format(resid.shape))

            num_p = len(self.ff.params)
            jacob = np.empty((num_d, num_p), dtype=float)
            jacob = return_jacobian(jacob, os.path.join(self.direc, par_file))
            logger.log(20, "  -- Jacobian shape: {}".format(jacob.shape))

            ma = jacob.T.dot(jacob)
            vb = jacob.T.dot(resid)

        # --- generate trial parameter sets -------------------------------
        if self.do_newton and not restart:
            logger.log(20, "~~ NEWTON-RAPHSON ~~".rjust(79, "~"))
            changes = do_newton(self.ff.params,
                                radii=self.newton_radii,
                                cutoffs=self.newton_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)
        if self.do_lstsq:
            logger.log(20, "~~ LEAST SQUARES ~~".rjust(79, "~"))
            changes = do_lstsq(ma, vb,
                               radii=self.lstsq_radii,
                               cutoffs=self.lstsq_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)
        if self.do_lagrange:
            logger.log(20, "~~ LAGRANGE ~~".rjust(79, "~"))
            for factor in sorted(self.lagrange_factors):
                changes = do_lagrange(ma, vb, factor,
                                      radii=self.lagrange_radii,
                                      cutoffs=self.lagrange_cutoffs)
                cleanup(self.new_ffs, self.ff, changes)
        if self.do_levenberg:
            logger.log(20, "~~ LEVENBERG ~~".rjust(79, "~"))
            for factor in sorted(self.levenberg_factors):
                changes = do_levenberg(ma, vb, factor,
                                       radii=self.levenberg_radii,
                                       cutoffs=self.levenberg_cutoffs)
                cleanup(self.new_ffs, self.ff, changes)
        if self.do_svd:
            logger.log(20, "~~ SINGULAR VALUE DECOMPOSITION ~~".rjust(79, "~"))
            mu, vs, mvt = return_svd(jacob)
            if self.svd_factors:
                changes = do_svd_w_thresholds(mu, vs, mvt, resid,
                                              self.svd_factors,
                                              radii=self.svd_radii,
                                              cutoffs=self.svd_cutoffs)
            else:
                changes = do_svd_wo_thresholds(mu, vs, mvt, resid,
                                               radii=self.svd_radii,
                                               cutoffs=self.svd_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)

        logger.log(20, "  -- Generated {} trial force field(s).".format(
            len(self.new_ffs)))

        if self.new_ffs:
            logger.log(20, "~~ EVALUATING TRIAL FF(S) ~~".rjust(79, "~"))
            for ff in self.new_ffs:
                data = opt.cal_ff(ff, self.args_ff, parent_ff=self.ff)
                c_data = score.data_by_type(data)
                r_dict, c_data = score.trim_data(r_dict, c_data)
                ff.score = score.compare_data(r_dict, c_data)
                opt.pretty_ff_results(ff)
            ranked = sorted(self.new_ffs, key=lambda x: x.score)
            best = ranked[0]
            if best.score < self.ff.score:
                logger.log(20, "~~ GRADIENT FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
                opt.pretty_ff_results(self.ff, level=20)
                opt.pretty_ff_results(best, level=20)
                copy_derivs(self.ff, best)
                self.new_ffs = [best]
                ff = best
            else:
                ff = self.ff
        else:
            ff = self.ff
        ff.export_ff(ff.path)
        return ff


# ---------------------------------------------------------------------------
# Method helpers
# ---------------------------------------------------------------------------

def copy_derivs(old_ff, new_ff):
    for i in range(len(new_ff.params)):
        new_ff.params[i].d1 = old_ff.params[i].d1
        new_ff.params[i].d2 = old_ff.params[i].d2


def check_cutoffs(par_rad, cutoffs):
    if min(cutoffs) <= par_rad <= max(cutoffs):
        return True
    logger.warning("  -- Radius outside cutoffs ({} <= {} <= {}).".format(
        min(cutoffs), par_rad, max(cutoffs)))
    return False


def check_radius(par_rad, max_rad):
    if par_rad > max_rad:
        logger.warning("  -- Radius {} exceeded max {}.".format(par_rad, max_rad))
        return max_rad / par_rad
    return 1.0


def check(changes, max_radii, cutoffs):
    new_changes = []
    for change in changes:
        radius = opt.calculate_radius(change[1])
        if max_radii:
            for max_rad in sorted(max_radii):
                scale = check_radius(radius, max_rad)
                if scale == 1:
                    new_changes.append(change)
                    break
                else:
                    new_changes.append(
                        (change[0] + " R{}".format(max_rad),
                         [x * scale for x in change[1]]))
        elif cutoffs:
            if check_cutoffs(radius, cutoffs):
                new_changes.append(change)
        else:
            new_changes.append(change)
    return new_changes


def cleanup(ffs, ff, changes):
    if changes:
        for method, change in changes:
            opt.pretty_param_changes(ff.params, change, method)
            new_ff = return_ff(ff, change, method)
            if new_ff:
                ffs.append(new_ff)
    else:
        logger.warning("  -- No changes generated! Confirm parameter has effect on objective.")


def do_method(func):
    def wrapper(*args, **kwargs):
        try:
            changes = func(*args)
        except opt.OptError as e:
            logger.warning(e)
        else:
            return check(changes, kwargs["radii"], kwargs["cutoffs"])
    return wrapper


@do_method
def do_lagrange(ma, vb, factor):
    mac = copy.deepcopy(ma)
    ind = np.diag_indices_from(mac)
    mac[ind] = mac[ind] + factor
    changes = solver(mac, vb)
    return [("LAGRANGE F{}".format(factor), changes)]


@do_method
def do_levenberg(ma, vb, factor):
    mac = copy.deepcopy(ma)
    ind = np.diag_indices_from(mac)
    mac[ind] = mac[ind] + factor
    changes = solver(mac, vb)
    return [("LM {}".format(factor), changes)]


@do_method
def do_lstsq(ma, vb):
    return [("LSTSQ", solver(ma, vb))]


@do_method
def do_newton(params):
    changes = []
    for p in params:
        if p.d1 != 0.0:
            if p.d2 > 0.00000001:
                changes.append(-p.d1 / p.d2)
            else:
                logger.warning("  -- 2nd derivative of {} is {:.4f}.".format(p, p.d2))
                logger.warning("  -- 1st derivative of {} is {:.4f}.".format(p, p.d1))
                changes.append(-1.0 if p.d1 > 0 else 1.0)
        else:
            raise opt.OptError("1st derivative of {} is 0; skipping NR.".format(p))
    return [("NR", changes)]


@do_method
def do_svd_w_thresholds(mu, vs, mvt, resid, factors):
    factors.sort(reverse=True)
    all_changes = []
    vsi = invert_vector(vs)
    msi = np.diag(vsi)

    changes = mvt.T.dot(msi.dot(mu.T.dot(resid))).flatten()
    for i, factor in enumerate(factors):
        old_msi = copy.deepcopy(msi)
        for j in range(len(vs)):
            if msi[j, j] > factor:
                msi[j, j] = 0.0
        if i != 0 and np.all(msi == old_msi):
            continue
        if np.all(msi == np.zeros(msi.shape)):
            break
        changes = mvt.T.dot(msi.dot(mu.T.dot(resid))).flatten()
        all_changes.append(("SVD T{}".format(factor), changes))
    return all_changes


@do_method
def do_svd_wo_thresholds(mu, vs, mvt, resid):
    all_changes = []
    vsi = invert_vector(vs)
    msi = np.diag(vsi)
    changes = mvt.T.dot(msi.dot(mu.T.dot(resid))).flatten()
    all_changes.append(("SVD Z0", changes))
    for i in range(len(vs) - 1):
        old_msi = copy.deepcopy(msi)
        msi[-(i + 1), -(i + 1)] = 0.0
        if np.allclose(msi, old_msi):
            continue
        changes = mvt.T.dot(msi.dot(mu.T.dot(resid))).flatten()
        all_changes.append(("SVD Z{}".format(i + 1), changes))
    return all_changes


def invert_vector(vector, threshold=0.0001):
    new_vec = np.empty(vector.shape, dtype=float)
    for i, x in enumerate(vector):
        new_vec[i] = 0.0 if abs(x) < threshold else 1.0 / x
    return new_vec


def return_ff(orig_ff, changes, method):
    """Build a new FF by adding scaled `changes` to the parameters."""
    new_ff = orig_ff.__class__()
    new_ff.method = method
    new_ff.params = copy.deepcopy(orig_ff.params)
    new_ff.path = orig_ff.path
    try:
        update_params(new_ff.params, changes)
    except data_structs.ParamError as e:
        logger.warning(e)
        return None
    return new_ff


def return_jacobian(jacob, par_file):
    """Read par_diff_*.txt and assemble the central-difference Jacobian."""
    with open(par_file, "r") as f:
        f.readline()  # labels
        whts = [float(x) for x in f.readline().split(",")]
        f.readline()  # reference values
        f.readline()  # original values
        ff_ind = 0
        while True:
            l1 = f.readline()
            l2 = f.readline()
            if not l2:
                break
            inc_data = list(map(float, l1.split(",")))
            dec_data = list(map(float, l2.split(",")))
            for data_ind, (inc, dec) in enumerate(zip(inc_data, dec_data)):
                dydp = (inc - dec) / 2.0
                jacob[data_ind, ff_ind] = whts[data_ind] * dydp
            ff_ind += 1
    return jacob


def return_svd(matrix):
    return np.linalg.svd(matrix, full_matrices=False)


def solver(ma, vb):
    changes, _, _, _ = np.linalg.lstsq(ma, vb, rcond=10 ** -12)
    return np.concatenate(changes).tolist()


def update_params(params, changes):
    """Increment parameter values by scaled changes."""
    try:
        for p, change in zip(params, changes):
            p.value += change * p.step
    except data_structs.ParamError as e:
        logger.warning(str(e))
        raise
