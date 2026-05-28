#!/usr/bin/env python
"""
opt
---
General optimization scaffolding for q2mm-amber-main.

Contains:
    Optimizer            Base class shared by Gradient and Simplex.
    SwarmOptimizer       Adapter that drives PSO_DE from hybrid_optimizer.py.
    catch_run_errors     Decorator that returns the best FF if an exception
                         is raised inside an optimizer's run() method.

Helper functions used by gradient.py / simplex.py / loop.py:
    return_ref_data, calculate_radius, differentiate_params,
    differentiate_ff, cal_ff, param_derivs, pretty_param_changes,
    pretty_ff_results, pretty_ff_params, pretty_derivs.
"""
from __future__ import absolute_import
from __future__ import division

import copy
import logging
import logging.config
import textwrap

import numpy as np

import calculate
import constants as co
import data_structs
import score

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


class OptError(Exception):
    """Raised when an optimizer hits an unrecoverable internal error."""
    pass


def catch_run_errors(func):
    """
    Decorator wrapping Optimizer.run(). If a known optimization error escapes,
    fall back to the best FF found so far (or the initial FF) and write it
    to disk before returning.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            return func(*args, **kwargs)
        except (ZeroDivisionError, OptError, data_structs.ParamError) as e:
            logger.warning("opt.catch_run_errors caught an error!")
            logger.warning(e)
            if getattr(self, "best_ff", None) is None:
                logger.warning("Exiting {} and returning initial FF.".format(
                    self.__class__.__name__.lower()))
                self.ff.export_ff(self.ff.path)
                return self.ff
            else:
                logger.warning("Exiting {} and returning best FF.".format(
                    self.__class__.__name__.lower()))
                self.best_ff.export_ff(self.best_ff.path)
                return self.best_ff
    return wrapper


class Optimizer(object):
    """
    Base class for serial gradient-style optimizers. Mirrors the old
    q2mm-master Optimizer interface so that the gradient and simplex
    modules can be ported with minimal changes.

    Parameters
    ----------
    direc : str
        Working directory for intermediate files.
    ff : data_structs.FF (or subclass)
        Initial force field.
    ff_lines : list[str] | None
        Lines of the FF file (used to reconstitute when writing).
    args_ff : list[str]
        Arguments for calculate.main to produce FF data.
    args_ref : list[str]
        Arguments for calculate.main to produce reference data.
    """

    def __init__(self, direc=None, ff=None, ff_lines=None,
                 args_ff=None, args_ref=None):
        logger.log(20, "~~ {} SETUP ~~".format(
            self.__class__.__name__.upper()).rjust(79, "~"))
        self.direc = direc
        self.ff = ff
        self.ff_lines = ff_lines
        self.args_ff = args_ff
        self.args_ref = args_ref
        self.new_ffs = []
        self.best_ff = None
        if self.ff_lines is None and self.ff is not None and self.ff.lines:
            self.ff_lines = self.ff.lines


def return_ref_data(args_ref):
    """Calculate the reference data set once and import weights."""
    logger.log(20, "~~ GATHERING REFERENCE DATA ~~".rjust(79, "~"))
    ref_data = calculate.main(args_ref)
    score.import_weights(ref_data)
    return ref_data


def calculate_radius(changes):
    """Euclidean radius of an unscaled parameter-change vector."""
    return float(np.sqrt(sum(x ** 2 for x in changes)))


def differentiate_params(params, central=True):
    """
    Build perturbed parameter sets around each parameter in `params`.

    For central differentiation each parameter contributes two sets
    (forward and backward); forward-only contributes one. Step size
    self-adjusts if a step would push a parameter out of its allowed
    range.
    """
    if central:
        logger.log(20, "~~ CENTRAL DIFFERENTIATION ON {} PARAMS ~~".format(
            len(params)).rjust(79, "~"))
    else:
        logger.log(20, "~~ FORWARD DIFFERENTIATION ON {} PARAMS ~~".format(
            len(params)).rjust(79, "~"))

    param_sets = []
    for i, param in enumerate(params):
        while True:
            original_value = float(param.value)
            forward_params = copy.deepcopy(params)
            backward_params = copy.deepcopy(params) if central else None
            try:
                ori_step = float(param.step)
                forward_params[i].value = original_value + ori_step
                if central:
                    backward_params[i].value = original_value - ori_step
            except data_structs.ParamFE as e:
                logger.warning(str(e))
                forward_params[i].value = forward_params[i].allowed_range[1]
                param.step = param.step / 2.0
                if central:
                    backward_params[i].value = original_value - ori_step
                param_sets.append(forward_params)
                if central:
                    param_sets.append(backward_params)
                break
            except data_structs.ParamBE as e:
                logger.warning(str(e))
                backward_params[i].value = backward_params[i].allowed_range[0]
                param.step = param.step / 2.0
                param_sets.append(forward_params)
                if central:
                    param_sets.append(backward_params)
                break
            except data_structs.ParamError as e:
                logger.warning(str(e))
                old_step = param.step
                upper = abs(param.value - max(param.allowed_range))
                lower = abs(param.value - min(param.allowed_range))
                param.step = min(upper, lower) * 0.1
                logger.warning("  -- Changed step size of {} from {} to {}.".format(
                    param, old_step, param.step))
            else:
                param_sets.append(forward_params)
                if central:
                    param_sets.append(backward_params)
                break
    logger.log(20, "  -- Generated {} differentiated parameter sets.".format(
        len(param_sets)))
    return param_sets


def differentiate_ff(ff, central=True):
    """
    Like differentiate_params but returns a list of FF objects, each
    with its `method` attribute marking which parameter was perturbed
    and in which direction.
    """
    param_sets = differentiate_params(ff.params, central=central)
    ffs = []
    for i, param_set in enumerate(param_sets):
        new_ff = ff.__class__()
        new_ff.params = param_set
        new_ff.path = ff.path
        if central and i % 2 == 1:
            new_ff.method = "BACKWARD {}".format(param_set[int(np.floor(i / 2.))])
        else:
            if central:
                new_ff.method = "FORWARD {}".format(param_set[int(np.floor(i / 2.))])
            else:
                new_ff.method = "FORWARD {}".format(param_set[i])
        ffs.append(new_ff)
    return ffs


def cal_ff(ff, ff_args, parent_ff=None, store_data=False):
    """
    Export an FF to disk, run calculate.main against it, and return
    the resulting Datum list.
    """
    if ff.path is None and parent_ff is not None:
        ff.path = parent_ff.path
    lines = parent_ff.lines if (parent_ff is not None and parent_ff.lines) else None
    if lines is not None:
        ff.export_ff(ff.path, lines=lines)
    else:
        ff.export_ff(ff.path)
    data = calculate.main(ff_args)
    if store_data:
        ff.data = data
    return data


def param_derivs(ff, ffs):
    """
    Use scored pairs of forward/backward FFs to populate ff.params[i].d1
    and ff.params[i].d2 (1st and 2nd numerical derivatives of the
    objective function wrt each parameter).
    """
    for i in range(0, len(ffs), 2):
        idx = i // 2
        ff.params[idx].d1 = (ffs[i].score - ffs[i + 1].score) * 0.5
        ff.params[idx].d2 = ffs[i].score + ffs[i + 1].score - 2 * ff.score
    pretty_derivs(ff.params)


def pretty_derivs(params, level=5):
    if logger.getEffectiveLevel() > level:
        return
    logger.log(level,
               "--" + " Parameter ".ljust(33, "-")
               + "--" + " 1st der. ".center(19, "-")
               + "--" + " 2nd der. ".center(19, "-")
               + "--")
    for p in params:
        try:
            logger.log(level,
                       "  " + "{}".format(p).ljust(33, " ")
                       + "  " + "{:15.4f}".format(p.d1).ljust(19, " ")
                       + "  " + "{:15.4f}".format(p.d2).ljust(19, " "))
        except (ValueError, TypeError):
            logger.log(level,
                       "  " + "{}".format(p).ljust(33, " ")
                       + "  " + "None".ljust(19, " ")
                       + "  " + "None".ljust(19, " "))
    logger.log(level, "-" * 79)


def pretty_ff_params(ffs, level=20):
    if logger.getEffectiveLevel() > level:
        return
    wrapper = textwrap.TextWrapper(width=79, subsequent_indent=" " * 29)
    logger.log(level,
               "--" + " PARAMETER ".ljust(25, "-")
               + "--" + " VALUES ".ljust(48, "-")
               + "--")
    for i in range(len(ffs[0].params)):
        wrapper.initial_indent = " {:25s} ".format(repr(ffs[0].params[i]))
        values = ["{:8.4f}".format(x.params[i].value) for x in ffs]
        logger.log(level, wrapper.fill(" ".join(values)))
    logger.log(level, "-" * 79)


def pretty_ff_results(ff, level=20):
    if logger.getEffectiveLevel() > level:
        return
    wrapper = textwrap.TextWrapper(width=79)
    logger.log(level, " {} ".format(ff.method).center(79, "="))
    logger.log(level, "SCORE: {}".format(ff.score))
    logger.log(level, "PARAMETERS:")
    logger.log(level, wrapper.fill(" ".join(map(str, ff.params))))
    logger.log(level, "=" * 79)
    logger.log(level, "")


def extract_forward(ffs):
    """Return only the FFs whose method string indicates a forward step."""
    return [x for x in ffs if "forward" in x.method.lower()]


def extract_ff_by_params(ffs, params):
    """
    Filter FFs whose differentiation 'method' string targets one of the
    given parameters (matched by ff_row/ff_col). Uses the trailing
    "ParAMBER[bf][row,col](value)" repr produced by ParAMBER.__repr__.
    """
    rows = [p.ff_row for p in params]
    cols = [p.ff_col for p in params]
    keep = []
    for ff in ffs:
        method = ff.method or ""
        row = col = None
        # Look for "[row,col]" in the method string.
        l_bracket = method.find("[", method.find("[") + 1)
        if l_bracket != -1:
            r_bracket = method.find("]", l_bracket)
            if r_bracket != -1:
                inner = method[l_bracket + 1:r_bracket]
                if "," in inner:
                    try:
                        r_str, c_str = inner.split(",")
                        row, col = int(r_str), int(c_str)
                    except ValueError:
                        pass
        if row in rows and col in cols:
            keep.append(ff)
    logger.log(20, "KEEPING FFS FOR SIMPLEX:\n{}".format(
        " ".join(str(x) for x in keep)))
    return keep


def pretty_param_changes(params, changes, method=None, level=20):
    if logger.getEffectiveLevel() > level:
        return
    if method:
        logger.log(level, " {} ".format(method).center(79, "="))
    else:
        logger.log(level, "=" * 79)
    logger.log(level,
               "--" + " PARAMETER ".ljust(34, "-")
               + "--" + " UNSCALED CHANGES ".center(19, "-")
               + "--" + " CHANGES ".center(18, "-")
               + "--")
    for p, change in zip(params, changes):
        logger.log(level,
                   "  " + "{}".format(p).ljust(34, " ")
                   + "  " + "{:7.4f}".format(change).center(19, " ")
                   + "  " + "{:7.4f}".format(change * p.step).center(18, " ")
                   + "  ")
    logger.log(level, "RADIUS: {}".format(calculate_radius(changes)))
    logger.log(level, "=" * 79)
    logger.log(level, "")


# ---------------------------------------------------------------------------
# Swarm optimizer adapter
# ---------------------------------------------------------------------------

class SwarmOptimizer(Optimizer):
    """
    Thin adapter that lets the LOOP command driver use the new hybrid
    PSO-DE optimizer from hybrid_optimizer.py with the same constructor
    signature as Gradient / Simplex.

    The fitness function passed to PSO_DE is a closure that:
      1) writes the candidate parameters into a deep-copied FF,
      2) exports the FF to disk,
      3) runs calculate.main(self.args_ff) to obtain calculated data,
      4) returns score.compare_data(...) against the trimmed reference.

    Notes
    -----
    Parallel execution (multiple AMBER calculations concurrently) requires
    file-isolated working directories per particle. This serial fallback
    runs one particle at a time but otherwise behaves correctly; set
    `n_processes=1` (the default).
    """

    DEFAULT_CONFIG = {
        "vectorize_func": False,
        "taper_GA": True,
        "mutation_strategy": "DE/best/1",
        "differential_weight": (0.4, 0.1),
        "recombination_constant": (0.7, 0.7),
        "inertia": (0.9, 0.4),
        "cognitive": (2.5, 0.5),
        "social": (0.5, 2.5),
    }

    def __init__(self, direc=None, ff=None, ff_lines=None,
                 args_ff=None, args_ref=None,
                 max_iter=200, pop_size=24, precision=0.001,
                 tight_spread=True):
        super(SwarmOptimizer, self).__init__(
            direc=direc, ff=ff, ff_lines=ff_lines,
            args_ff=args_ff, args_ref=args_ref)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.precision = precision
        self.tight_spread = tight_spread

    @catch_run_errors
    def run(self, ref_data=None):
        from hybrid_optimizer import PSO_DE, Bounds_Handler

        if ref_data is None:
            ref_data = return_ref_data(self.args_ref)
        r_dict = score.data_by_type(ref_data)

        # initial FF score
        if self.ff.data is None:
            self.ff.export_ff()
            self.ff.data = calculate.main(self.args_ff)
        c_dict = score.data_by_type(self.ff.data)
        r_dict, c_dict = score.trim_data(r_dict, c_dict)
        if self.ff.score is None:
            self.ff.score = score.compare_data(r_dict, c_dict)
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        pretty_ff_results(self.ff, level=20)

        # bounds + deviations for each parameter
        lb, ub, deviations = [], [], []
        for p in self.ff.params:
            lb.append(p.allowed_range[0])
            ub.append(p.allowed_range[1])
            if p.ptype in ("af", "bf"):
                deviations.append(0.125 if self.tight_spread else 1.0)
            elif p.ptype == "ae":
                deviations.append(15.0)
            elif p.ptype == "be":
                deviations.append(0.5)
            elif p.ptype == "df":
                deviations.append(5.0)
            else:
                deviations.append(1.0)

        initial = [p.value for p in self.ff.params]
        config = dict(self.DEFAULT_CONFIG)
        config.update({
            "lb": lb,
            "ub": ub,
            "size_pop": self.pop_size,
            "max_iter": self.max_iter,
            "initial_guesses": initial,
            "guess_deviation": deviations,
            "guess_ratio": 0.7 if self.tight_spread else 0.3,
            "bounds_strategy": Bounds_Handler.REFLECTIVE,
        })

        def fitness(enumerable_input):
            # PSO_DE may pass either a bare array or (idx, array) when
            # pass_particle_num=True; handle both.
            if isinstance(enumerable_input, tuple) and len(enumerable_input) == 2:
                _, params_vec = enumerable_input
            else:
                params_vec = enumerable_input
            trial_ff = copy.deepcopy(self.ff)
            trial_ff.set_param_values(params_vec)
            try:
                trial_ff.export_ff(self.ff.path, lines=self.ff.lines)
                data = calculate.main(self.args_ff)
                cdict = score.data_by_type(data)
                rdict, cdict = score.trim_data(score.data_by_type(ref_data), cdict)
                return score.compare_data(rdict, cdict)
            except Exception as e:
                logger.warning("Particle evaluation failed: {}".format(e))
                return float("inf")

        opt = PSO_DE(
            fitness,
            len(self.ff.params),
            config=config,
            n_processes=1,
            pass_particle_num=False,
            verbose=True,
        )
        opt.Y = opt.cal_y()
        opt.update_pbest()
        opt.update_gbest()
        opt.recorder()
        best_x, best_y = opt.run(precision=self.precision)

        # build the best FF
        self.best_ff = copy.deepcopy(self.ff)
        self.best_ff.set_param_values(best_x)
        self.best_ff.score = best_y
        self.best_ff.method = "SWARM"
        if best_y < self.ff.score:
            logger.log(20, "~~ SWARM FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
            self.best_ff.export_ff(self.ff.path, lines=self.ff.lines)
            return self.best_ff
        logger.log(20, "~~ SWARM FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
        # restore initial parameters
        self.ff.export_ff(self.ff.path, lines=self.ff.lines)
        return self.ff
