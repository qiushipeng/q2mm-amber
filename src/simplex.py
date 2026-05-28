#!/usr/bin/env python
"""
simplex
-------
In-house simplex (Nelder-Mead-like) optimizer for q2mm-amber-main.

Port of q2mm-master/q2mm/simplex.py. Imports the new data_structs and
score modules. Invoked by the loop.in SIMP command, e.g.:
    SIMP max_params=10
"""
from __future__ import absolute_import
from __future__ import division

import copy
import logging
import logging.config
import textwrap

import calculate
import constants as co
import data_structs
import opt
import score

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


class Simplex(opt.Optimizer):
    """
    Custom simplex optimizer. Configuration knobs:

    max_cycles : int
        Hard cap on simplex cycles.
    max_params : int
        If the FF has more parameters than this, only the `max_params`
        with the lowest simp_var (d2 / d1**2) are optimized this round.
    do_massive_contraction : bool
        Contract all vertices toward the best if a single contraction fails.
    do_weighted_reflection : bool
        Use score-weighted inversion when generating the reflection point.
    """

    def __init__(self, direc=None, ff=None, ff_lines=None,
                 args_ff=None, args_ref=None):
        super(Simplex, self).__init__(
            direc=direc, ff=ff, ff_lines=ff_lines,
            args_ff=args_ff, args_ref=args_ref)
        self._max_cycles_wo_change = None
        self.do_massive_contraction = True
        self.do_weighted_reflection = True
        self.max_cycles = 100
        self.max_params = 3

    @opt.catch_run_errors
    def run(self, r_data=None):
        if r_data is None:
            r_data = opt.return_ref_data(self.args_ref)

        if self.ff.score is None:
            logger.log(20, "~~ CALCULATING INITIAL FF SCORE ~~".rjust(79, "~"))
            self.ff.export_ff()
            data = calculate.main(self.args_ff)
            r_dict = score.data_by_type(r_data)
            c_dict = score.data_by_type(data)
            r_dict, c_dict = score.trim_data(r_dict, c_dict)
            self.ff.score = score.compare_data(r_dict, c_dict)
        else:
            logger.log(20, "  -- Reused existing score and data for initial FF.")

        logger.log(20, "~~ SIMPLEX OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        # parameter-count branching: if too many params, pick the low d2/d1^2 ones
        if self.max_params and len(self.ff.params) > self.max_params:
            logger.log(20, "  -- More parameters than max_params={}".format(self.max_params))
            if None in [p.d1 for p in self.ff.params]:
                logger.log(15, "  -- Calculating new parameter derivatives.")
                ffs = opt.differentiate_ff(self.ff, central=True)
                for ff in ffs:
                    ff.export_ff(path=self.ff.path, lines=self.ff_lines)
                    data = calculate.main(self.args_ff)
                    r_dict = score.data_by_type(r_data)
                    c_dict = score.data_by_type(data)
                    r_dict, c_dict = score.trim_data(r_dict, c_dict)
                    ff.score = score.compare_data(r_dict, c_dict)
                    opt.pretty_ff_results(ff)
                opt.param_derivs(self.ff, ffs)
                ffs = opt.extract_forward(ffs)
            else:
                logger.log(15, "  -- Reusing existing parameter derivatives.")
                ffs = opt.differentiate_ff(self.ff, central=False)

            params = select_simp_params_on_derivs(
                self.ff.params, max_params=self.max_params)
            self.new_ffs = opt.extract_ff_by_params(ffs, params)

            ff_rows = [p.ff_row for p in params]
            ff_cols = [p.ff_col for p in params]
            for ff in self.new_ffs:
                ff.params = [p for p in ff.params
                             if p.ff_row in ff_rows and p.ff_col in ff_cols]
            ff_copy = copy.deepcopy(self.ff)
            ff_copy.params = [p for p in self.ff.params
                              if p.ff_row in ff_rows and p.ff_col in ff_cols]
        else:
            self.new_ffs = opt.differentiate_ff(self.ff, central=False)
            ff_copy = copy.deepcopy(self.ff)

        # ensure all forward-differentiated FFs are scored
        for ff in self.new_ffs:
            if ff.score is None:
                ff.export_ff(path=self.ff.path, lines=self.ff_lines)
                data = calculate.main(self.args_ff)
                r_dict = score.data_by_type(r_data)
                c_dict = score.data_by_type(data)
                r_dict, c_dict = score.trim_data(r_dict, c_dict)
                ff.score = score.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(ff)

        self.new_ffs = sorted(self.new_ffs + [ff_copy], key=lambda x: x.score)
        self._max_cycles_wo_change = 3 * (len(self.new_ffs) - 1)
        wrapper = textwrap.TextWrapper(width=79)
        opt.pretty_ff_params(self.new_ffs)

        current_cycle = 0
        cycles_wo_change = 0
        while current_cycle < self.max_cycles \
                and cycles_wo_change < self._max_cycles_wo_change:
            current_cycle += 1
            last_best_ff = copy.deepcopy(self.new_ffs[0])
            logger.log(20, "~~ START SIMPLEX CYCLE {} ~~".format(
                current_cycle).rjust(79, "~"))
            logger.log(20, "ORDERED FF SCORES:")
            logger.log(20, wrapper.fill("{}".format(
                " ".join("{:15.4f}".format(x.score) for x in self.new_ffs))))

            inv_ff = self.ff.__class__()
            inv_ff.method = ("WEIGHTED INVERSION" if self.do_weighted_reflection
                             else "INVERSION")
            inv_ff.params = copy.deepcopy(last_best_ff.params)
            ref_ff = self.ff.__class__()
            ref_ff.method = "REFLECTION"
            ref_ff.params = copy.deepcopy(last_best_ff.params)

            if self.do_weighted_reflection:
                score_diff_sum = sum(x.score - self.new_ffs[-1].score
                                     for x in self.new_ffs[:-1])
                if score_diff_sum == 0.0:
                    raise opt.OptError("No difference between FF scores. Exiting simplex.")

            for i in range(len(last_best_ff.params)):
                if self.do_weighted_reflection:
                    inv_val = (
                        sum(x.params[i].value * (x.score - self.new_ffs[-1].score)
                            for x in self.new_ffs[:-1])
                        / score_diff_sum)
                else:
                    inv_val = (sum(x.params[i].value for x in self.new_ffs[:-1])
                               / len(self.new_ffs[:-1]))
                inv_ff.params[i].value = inv_val
                ref_ff.params[i].value = 2 * inv_val - self.new_ffs[-1].params[i].value

            ref_ff.export_ff(path=self.ff.path, lines=self.ff.lines)
            data = calculate.main(self.args_ff)
            r_dict = score.data_by_type(r_data)
            c_dict = score.data_by_type(data)
            r_dict, c_dict = score.trim_data(r_dict, c_dict)
            ref_ff.score = score.compare_data(r_dict, c_dict)
            opt.pretty_ff_results(ref_ff)

            if ref_ff.score < last_best_ff.score:
                logger.log(20, "~~ ATTEMPTING EXPANSION ~~".rjust(79, "~"))
                exp_ff = self.ff.__class__()
                exp_ff.method = "EXPANSION"
                exp_ff.params = copy.deepcopy(last_best_ff.params)
                for i in range(len(last_best_ff.params)):
                    exp_ff.params[i].value = (
                        3 * inv_ff.params[i].value
                        - 2 * self.new_ffs[-1].params[i].value)
                exp_ff.export_ff(path=self.ff.path, lines=self.ff.lines)
                data = calculate.main(self.args_ff)
                r_dict = score.data_by_type(r_data)
                c_dict = score.data_by_type(data)
                r_dict, c_dict = score.trim_data(r_dict, c_dict)
                exp_ff.score = score.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(exp_ff)
                if exp_ff.score < ref_ff.score:
                    self.new_ffs[-1] = exp_ff
                    logger.log(20, "  -- Expansion succeeded.")
                else:
                    self.new_ffs[-1] = ref_ff
                    logger.log(20, "  -- Expansion failed. Keeping reflected.")
            elif ref_ff.score < self.new_ffs[-2].score:
                logger.log(20, "  -- Keeping reflected parameters.")
                self.new_ffs[-1] = ref_ff
            else:
                logger.log(20, "~~ ATTEMPTING CONTRACTION ~~".rjust(79, "~"))
                con_ff = self.ff.__class__()
                con_ff.method = "CONTRACTION"
                con_ff.params = copy.deepcopy(last_best_ff.params)
                for i in range(len(last_best_ff.params)):
                    if ref_ff.score > self.new_ffs[-1].score:
                        con_val = ((inv_ff.params[i].value
                                    + self.new_ffs[-1].params[i].value) / 2)
                    else:
                        con_val = ((3 * inv_ff.params[i].value
                                    - self.new_ffs[-1].params[i].value) / 2)
                    con_ff.params[i].value = con_val
                self.ff.export_ff(params=con_ff.params)
                data = calculate.main(self.args_ff)
                r_dict = score.data_by_type(r_data)
                c_dict = score.data_by_type(data)
                r_dict, c_dict = score.trim_data(r_dict, c_dict)
                con_ff.score = score.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(con_ff)

                if con_ff.score < self.new_ffs[-2].score:
                    logger.log(20, "  -- Contraction succeeded.")
                    self.new_ffs[-1] = con_ff
                elif self.do_massive_contraction:
                    logger.log(20, "~~ DOING MASSIVE CONTRACTION ~~".rjust(79, "~"))
                    for ff in self.new_ffs[1:]:
                        for i in range(len(last_best_ff.params)):
                            ff.params[i].value = (
                                (ff.params[i].value
                                 + self.new_ffs[0].params[i].value) / 2)
                        self.ff.export_ff(params=ff.params)
                        data = calculate.main(self.args_ff)
                        r_dict = score.data_by_type(r_data)
                        c_dict = score.data_by_type(data)
                        r_dict, c_dict = score.trim_data(r_dict, c_dict)
                        ff.score = score.compare_data(r_dict, c_dict)
                        ff.method += " MC"
                        opt.pretty_ff_results(ff)
                else:
                    logger.log(20, "  -- Contraction failed.")
                    self.new_ffs[-1] = con_ff

            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            if self.new_ffs[0].score < last_best_ff.score:
                cycles_wo_change = 0
            else:
                cycles_wo_change += 1
                logger.log(20, "  -- {} / {} cycles without improvement.".format(
                    cycles_wo_change, self._max_cycles_wo_change))
            logger.log(20, "BEST:")
            opt.pretty_ff_results(self.new_ffs[0], level=20)
            logger.log(20, "~~ END SIMPLEX CYCLE {} ~~".format(
                current_cycle).rjust(79, "~"))

        self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
        best_ff = self.new_ffs[0]
        if best_ff.score < self.ff.score:
            logger.log(20, "~~ SIMPLEX FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
            best_ff = restore_simp_ff(best_ff, self.ff)
        else:
            logger.log(20, "~~ SIMPLEX FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
            best_ff = self.ff
        opt.pretty_ff_results(self.ff, level=20)
        opt.pretty_ff_results(best_ff, level=20)
        logger.log(20, "  -- Writing best force field from simplex.")
        best_ff.export_ff(best_ff.path)
        return best_ff


def calc_simp_var(params):
    """Compute simp_var = d2 / d1**2 for each parameter."""
    for p in params:
        p.simp_var = p.d2 / (p.d1 ** 2.0)


def select_simp_params_on_derivs(params, max_params=10):
    """Pick the `max_params` parameters with the lowest simp_var."""
    calc_simp_var(params)
    keep = sorted(params, key=lambda x: x.simp_var)[:max_params]
    logger.log(20, "KEEPING PARAMS FOR SIMPLEX:\n{}".format(
        " ".join(str(p) for p in keep)))
    return keep


def restore_simp_ff(new_ff, old_ff):
    """
    Copy non-parameter attributes from old_ff and merge in any parameters
    old_ff had that new_ff is missing.
    """
    old_ff.copy_attributes(new_ff)
    if len(old_ff.params) > len(new_ff.params):
        new_params = copy.deepcopy(new_ff.params)
        new_ff.params = copy.deepcopy(old_ff.params)
        for i, p_old in enumerate(old_ff.params):
            for p_new in new_params:
                if p_old.ff_row == p_new.ff_row and p_old.ff_col == p_new.ff_col:
                    new_ff.params[i] = copy.deepcopy(p_new)
    return new_ff
