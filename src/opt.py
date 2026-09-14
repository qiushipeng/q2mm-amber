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

Every optimizer talks to the MM engine through a calculators.Calculator: it
hands over a data_structs.FF and gets back the Datum list to score, so no
optimizer writes an engine file or runs a program itself.

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
import os
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


# --- Parallel HYBR support -------------------------------------------------
#
# Calculator.evaluate_many scores each particle inside the worker that ran
# it, through a reducer that has to be a *top-level* (picklable) function.
# SwarmOptimizer.run stashes the reference data here BEFORE the calculator
# spawns its pool; on Linux fork, workers inherit it via memory
# copy-on-write, so only a float travels back per particle.
_REF_DATA = None


def _score_particle(data):
    """Reducer for Calculator.evaluate_many: one particle's data -> score."""
    return score.score_data(_REF_DATA, data)


def catch_run_errors(func):
    """
    Decorator wrapping Optimizer.run(). If a known optimization error escapes,
    fall back to the best FF found so far (or the initial FF) and hand it to
    the calculator, which writes it to disk, before returning.
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
                self.calculator.update_ff(self.ff)
                return self.ff
            else:
                logger.warning("Exiting {} and returning best FF.".format(
                    self.__class__.__name__.lower()))
                self.calculator.update_ff(self.best_ff)
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
        CDAT arguments; used to build a calculator when none is given.
    args_ref : list[str]
        Arguments for calculate.main to produce reference data.
    calculator : calculators.Calculator | None
        Evaluates trial force fields (calculator.evaluate(ff) -> Datum
        list). Built from args_ff on first use when not given.
    """

    def __init__(self, direc=None, ff=None, ff_lines=None,
                 args_ff=None, args_ref=None, calculator=None):
        logger.log(20, "~~ {} SETUP ~~".format(
            self.__class__.__name__.upper()).rjust(79, "~"))
        self.direc = direc
        self.ff = ff
        self.ff_lines = ff_lines
        self.args_ff = args_ff
        self.args_ref = args_ref
        self.new_ffs = []
        self.best_ff = None
        self._calculator = calculator
        if self.ff_lines is None and self.ff is not None and self.ff.lines:
            self.ff_lines = self.ff.lines

    @property
    def calculator(self):
        if self._calculator is None:
            if not self.args_ff:
                raise OptError("No calculator and no CDAT arguments to build one from.")
            self._calculator = calculate.build_calculator(self.args_ff, ff=self.ff)
        return self._calculator

    @calculator.setter
    def calculator(self, value):
        self._calculator = value


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


def trial_ff(template, values):
    """
    A lean copy of `template` carrying only what a Calculator needs to
    evaluate it -- path and parameters, set to `values` -- and none of the
    template's data, which would otherwise be deep-copied (and pickled to a
    worker) once per particle.
    """
    ff = template.__class__()
    ff.path = template.path
    ff.params = copy.deepcopy(template.params)
    ff.set_param_values(values)
    return ff


def cal_ff(ff, calculator, parent_ff=None, store_data=False):
    """
    Evaluate an FF with the calculator and return the resulting Datum list.
    """
    if ff.path is None and parent_ff is not None:
        ff.path = parent_ff.path
    data = calculator.evaluate(ff)
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
    Adapter that lets the loop.in HYBR command drive the hybrid PSO-DE
    optimizer from hybrid_optimizer.py with the same constructor signature
    as Gradient / Simplex. Follows the HYBR logic of upstream q2mm's
    hybrid-opt branch:

    * The swarm is built by the first run() and kept: every later run()
      continues the same particles from where they stopped. loop.py calls
      run() once per cycle of the LOOP block holding the HYBR command and
      starts a new SwarmOptimizer for a new block.
    * The LOOP's convergence is passed in as `precision`, the swarm's own
      early-stop tolerance (every particle within `precision` of the best
      one, in every parameter, for 20 consecutive iterations).
    * The PSO/DE hyperparameters taper from exploring to exploiting over
      the first cycle (strategy "exp_decay"); continued cycles run with
      them frozen at their final values (strategy "").

    PSO_DE hands the whole swarm to the fitness function at once (one row
    of parameter values per particle). The fitness:
      1) turns every row into a trial FF (data_structs),
      2) has the calculator evaluate them all -- in per-particle working
         directories on a worker pool when n_processes > 1,
      3) scores each particle's data against the reference inside the
         worker that produced it (score.score_data),
    and returns one score per particle.
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
                 args_ff=None, args_ref=None, calculator=None,
                 max_iter=200, pop_size=24, tight_spread=True, n_processes=1):
        super(SwarmOptimizer, self).__init__(
            direc=direc, ff=ff, ff_lines=ff_lines,
            args_ff=args_ff, args_ref=args_ref, calculator=calculator)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.tight_spread = tight_spread
        # n_processes = 1 -> serial (safe default). n_processes > 1 has the
        # calculator run the particles on a multiprocessing pool, each in
        # its own copy of the working directory so concurrent
        # tleap/sander/nab runs don't clobber each other's files.
        self.n_processes = n_processes
        # The PSO_DE, built by the first run() and continued by later ones.
        # Its record_value is the swarm history loop.py persists. Stays
        # None if run() dies before the optimizer exists.
        self.hybrid_opt = None

    def _fitness(self):
        """The batch fitness PSO_DE calls: swarm rows -> trial FFs -> calculator -> scores."""
        pool_dir = os.path.join(self.direc, "swarm_particles")
        n_processes = self.n_processes

        def fitness(X):
            ffs = [trial_ff(self.ff, row) for row in X]
            scores = self.calculator.evaluate_many(
                ffs, reducer=_score_particle, pool_dir=pool_dir,
                n_processes=n_processes)
            # A particle whose evaluation raised comes back as None: never
            # a silent good score, so it is discarded as inf.
            return [float("inf") if s is None else float(s) for s in scores]
        # Tell PSO_DE the fitness takes the whole swarm (its own pool is
        # bypassed; the calculator manages the workers).
        fitness.mode = "vectorization"
        return fitness

    def _build(self):
        """A PSO_DE around the current FF: bounds, starting spread, and the first evaluation of the swarm."""
        from hybrid_optimizer import PSO_DE, Bounds_Handler

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
        if self.n_processes > 1:
            logger.log(20, "HYBR parallel: {} workers x {} particle dirs under {}".format(
                self.n_processes, self.pop_size, os.path.join(self.direc, "swarm_particles")))

        opt = PSO_DE(
            self._fitness(),
            len(self.ff.params),
            config=config,
            n_processes=self.n_processes,
            pass_particle_num=False,
            verbose=True,
        )
        # Publish before evaluating: @catch_run_errors swallows exceptions,
        # and a partial record_value is still worth writing out.
        self.hybrid_opt = opt
        opt.Y = opt.cal_y()
        opt.update_pbest()
        opt.update_gbest()
        opt.recorder()
        return opt

    @catch_run_errors
    def run(self, ref_data=None, precision=None, strategy="exp_decay"):
        """
        One cycle of max_iter swarm iterations.

        precision : float | None
            The swarm's early-stop tolerance; loop.py passes the LOOP's
            convergence. None runs the full max_iter.
        strategy : str
            "exp_decay" tapers the PSO/DE hyperparameters over the cycle,
            "" leaves them where they are (continued cycles).

        Returns the best FF found so far when it beats self.ff, else self.ff;
        the calculator writes the returned parameters to disk either way.
        """
        if ref_data is None:
            ref_data = return_ref_data(self.args_ref)

        # initial FF score
        if self.ff.data is None:
            self.ff.data = self.calculator.evaluate(self.ff)
        if self.ff.score is None:
            self.ff.score = score.score_data(ref_data, self.ff.data)
        logger.log(20, "~~ HYBRID OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        pretty_ff_results(self.ff, level=20)

        # Stash the reference data on the module global BEFORE the
        # calculator spawns its pool so Linux-fork workers inherit it.
        global _REF_DATA
        _REF_DATA = ref_data

        if self.hybrid_opt is None:
            opt = self._build()
        else:
            logger.log(20, "  -- Continuing the swarm from the previous cycle.")
            opt = self.hybrid_opt
        best_x, best_y = opt.run(precision=precision, strategy=strategy)

        # build the best FF
        self.best_ff = copy.deepcopy(self.ff)
        self.best_ff.set_param_values(best_x)
        self.best_ff.score = best_y
        self.best_ff.method = "HYBRID"
        if best_y < self.ff.score:
            logger.log(20, "~~ HYBRID FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
            self.calculator.update_ff(self.best_ff)
            return self.best_ff
        logger.log(20, "~~ HYBRID FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
        # restore initial parameters
        self.calculator.update_ff(self.ff)
        return self.ff
