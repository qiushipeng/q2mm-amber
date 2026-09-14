#!/usr/bin/env python
"""
loop
----
Top-level driver for q2mm-amber-main. Reads a loop.in file using the
same command vocabulary as q2mm-master, dispatches each line to the
appropriate module, and runs nested LOOP / END optimization blocks
until convergence.

Supported commands (same as q2mm-master)
----------------------------------------
DIR <path>                Set working directory for all relative paths.
FFLD read <path>          Import a force field (.frcmod -> AmberFF).
FFLD write <path>         Export the current best force field.
PARM <pfile>              Trim the FF parameters via parameters.py.
RDAT <args ...>           Calculate reference data (calculate.main).
CDAT <args ...>           Build the Calculator for these arguments
                          (calculate.build_calculator) and calculate FF
                          data with the current FF parameters. The
                          optimizers evaluate every trial FF through it.
COMP [-o out] [-p]        Score reference vs calculated; write pretty table.
LOOP <conv> ... END       Iterate the enclosed block until score change
                          < conv. Block typically contains GRAD, SIMP,
                          or HYBR commands.
GRAD [opts ...]           Run gradient.Gradient.run(). Options use the
                          old grammar, eg "lstsq=True,radii=[1./10.]".
SIMP [max_params=N]       Run simplex.Simplex.run().
HYBR [--max_iter N] [--pop_size N] [--tight T|F] [--n_processes N]
                          Run one cycle of the hybrid PSO-DE optimizer
                          (opt.SwarmOptimizer), as upstream q2mm's
                          hybrid-opt branch does: inside a LOOP block the
                          swarm persists from cycle to cycle, the LOOP's
                          convergence is also the swarm's own precision,
                          and a new LOOP block starts a new swarm.
WGHT <typ> <weight>       Override constants.WEIGHTS[typ] (eg "WGHT b 100.").
STEP <ptype> <step>       Override constants.STEPS[ptype].
FXATM <file>              Exclude fixed atoms from the Hessian fit. <file>
                          lists one 1-based atom index per line; Hessian
                          elements coupling those atoms get weight 0. Put it
                          before COMP/HYBR so the exclusion is in effect.
END                       Terminates an inner LOOP block (no-op outside).

Usage
-----
    cd <directory holding your reference / .frcmod / .in files>
    python /path/to/q2mm-amber-main/src/loop.py loop.in

The 'DIR ./' line should match the directory you cd'd to.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import glob
import logging
import logging.config
import os
import pickle
import re
import shutil
import sys

import constants as co
import parameters as parameters_module
import score
from utilities import AmberUtilities

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


# ---------------------------------------------------------------------------
# Loop driver
# ---------------------------------------------------------------------------

class Loop(object):
    """
    Stateful driver. One Loop instance corresponds to one nesting level
    in the loop.in file. The top-level driver creates a single Loop,
    and each LOOP ... END block produces a child Loop.
    """

    def __init__(self):
        self.convergence = 0.01
        self.cycle_num = 0
        self.direc = "."
        self.ff = None
        self.args_ff = None
        self.args_ref = None
        self.loop_lines = None
        self.ref_data = None
        self.calculator = None
        # The swarm of the HYBR command, kept across the cycles of one LOOP
        # block (upstream q2mm logic) and dropped when the block ends.
        self.swarm = None

    def _get_calculator(self):
        """The Calculator built at CDAT, or one built now from the CDAT args."""
        if self.calculator is None:
            if not self.args_ff:
                raise ValueError("CDAT must precede the optimizer commands so a "
                                 "calculator can be built from its arguments.")
            import calculate
            self.calculator = calculate.build_calculator(self.args_ff, ff=self.ff)
        return self.calculator

    # -- inner LOOP block ----------------------------------------------------

    def opt_loop(self):
        """
        Execute the inner LOOP / END block repeatedly until
        |last_score - new_score| / last_score < self.convergence.
        Backs up the current best FF to ff_NNN.frcmod after each cycle,
        numbering on from any such files already in the directory.
        """
        change = None
        last_score = None
        if self.ff.score is None:
            logger.warning("No initial score; computing one to seed loop.")
            self.ff.data = self._get_calculator().evaluate(self.ff)
            self.ff.score = score.score_data(self.ref_data, self.ff.data)

        while last_score is None \
                or change is None \
                or change > self.convergence:
            self.cycle_num += 1
            last_score = self.ff.score
            self.ff = self.run_loop_input(self.loop_lines, score=self.ff.score)
            new_score = self.ff.score
            if last_score == 0:
                change = 0.0
            else:
                change = abs(last_score - new_score) / abs(last_score)
            pretty_loop_summary(self.cycle_num, new_score, change)

            backup_files = glob.glob(os.path.join(self.direc, "ff_???.frcmod"))
            if backup_files:
                backup_files.sort()
                last_num = int(os.path.basename(backup_files[-1])[3:6])
                backup = os.path.join(self.direc, "ff_{:03d}.frcmod".format(last_num + 1))
            else:
                backup = os.path.join(self.direc, "ff_001.frcmod")
            AmberUtilities.write_frcmod(self.ff, backup)
            logger.log(20, "  -- Wrote best FF to {}".format(backup))

        for p in self.ff.params:
            p.value_at_limits()
        self.swarm = None
        return self.ff

    # -- top-level command interpreter --------------------------------------

    def run_loop_input(self, lines, score=None):
        """
        Walk through `lines` and dispatch each one. Returns the current
        force field after the block completes.
        """
        lines_iter = iter(lines)
        while True:
            try:
                line = next(lines_iter)
            except StopIteration:
                return self.ff
            cols = line.split()
            if not cols:
                continue
            cmd = cols[0]

            if cmd == "DIR":
                self.direc = cols[1]
                logger.log(20, "DIR -> {}".format(self.direc))

            elif cmd == "FFLD":
                self._handle_ffld(cols)

            elif cmd == "PARM":
                logger.log(20, "~~ SELECTING PARAMETERS ~~".rjust(79, "~"))
                self.ff.params = parameters_module.trim_params_by_file(
                    self.ff.params, os.path.join(self.direc, cols[1])
                )

            elif cmd == "RDAT":
                logger.log(20, "~~ CALCULATING REFERENCE DATA ~~".rjust(79, "~"))
                if len(cols) > 1:
                    self.args_ref = " ".join(cols[1:]).split()
                import opt as opt_module
                self.ref_data = opt_module.return_ref_data(self.args_ref)

            elif cmd == "CDAT":
                logger.log(20, "~~ CALCULATING FF DATA ~~".rjust(79, "~"))
                if len(cols) > 1:
                    self.args_ff = " ".join(cols[1:]).split()
                    self.calculator = None
                self.ff.data = self._get_calculator().evaluate(self.ff)

            elif cmd == "COMP":
                self._handle_comp(cols)

            elif cmd == "LOOP":
                inner = []
                inner_line = next(lines_iter)
                while inner_line.split()[0] != "END":
                    inner.append(inner_line)
                    inner_line = next(lines_iter)
                inner_loop = Loop()
                inner_loop.convergence = float(cols[1])
                inner_loop.direc = self.direc
                inner_loop.ff = self.ff
                inner_loop.args_ff = self.args_ff
                inner_loop.args_ref = self.args_ref
                inner_loop.ref_data = self.ref_data
                inner_loop.calculator = self.calculator
                inner_loop.loop_lines = inner
                pretty_loop_input(inner, name="OPTIMIZATION LOOP",
                                  score=self.ff.score)
                self.ff = inner_loop.opt_loop()

            elif cmd == "GRAD":
                self._handle_grad(cols)

            elif cmd == "SIMP":
                self._handle_simp(cols)

            elif cmd == "HYBR":
                self._handle_hybr(cols)

            elif cmd == "WGHT":
                co.WEIGHTS[cols[1]] = float(cols[2])
                logger.log(20, "WGHT {} = {}".format(cols[1], cols[2]))

            elif cmd == "STEP":
                co.STEPS[cols[1]] = float(cols[2])
                logger.log(20, "STEP {} = {}".format(cols[1], cols[2]))

            elif cmd == "FXATM":
                import calculate
                calculate.load_fixed_atoms(os.path.join(self.direc, cols[1]))

            elif cmd == "END":
                # Stray END outside of a LOOP block - skip.
                continue

            else:
                logger.warning("Unknown command: {}".format(line))

    # -- per-command helpers ------------------------------------------------

    def _handle_ffld(self, cols):
        action = cols[1]
        target = cols[2]
        full = os.path.join(self.direc, target)
        if action == "read":
            if "frcmod" in target:
                # Safeguard: the optimizer rewrites `full` in place on every
                # FF evaluation (tleap reloads it each time), so the pristine
                # starting FF would be destroyed by a run. Keep a one-time
                # ".orig" backup and restore from it at the start of every
                # run -- this both preserves the original and makes each run
                # start from the same pristine parameters. To re-baseline
                # (e.g. after intentionally editing the FF), delete the .orig.
                orig = full + ".orig"
                if os.path.isfile(orig):
                    shutil.copyfile(orig, full)
                    logger.log(20, "FFLD read: restored pristine FF from {}".format(orig))
                else:
                    shutil.copyfile(full, orig)
                    logger.log(20, "FFLD read: saved original FF backup to {}".format(orig))
                self.ff = AmberUtilities.read_frcmod(full)
            else:
                raise ValueError(
                    "Only frcmod FFs supported in q2mm-amber-main "
                    "(saw {}).".format(target))
            self.ff.method = "READ"
            # A calculator built for an earlier FF would write on its lines.
            self.calculator = None
            logger.log(20, "FFLD read {}: {} parameters".format(full, len(self.ff.params)))
        elif action == "write":
            AmberUtilities.write_frcmod(self.ff, full)
            logger.log(20, "FFLD write {}".format(full))
        else:
            raise ValueError("FFLD: unknown action {}".format(action))

    def _handle_comp(self, cols):
        out = None
        do_print = False
        if "-o" in cols:
            out = os.path.join(self.direc, cols[cols.index("-o") + 1])
        if "-p" in cols:
            do_print = True
        self.ff.score = score.score_data(
            self.ref_data, self.ff.data, output=out, doprint=do_print)
        logger.log(20, "COMP score: {}".format(self.ff.score))

    def _handle_grad(self, cols):
        import gradient
        grad = gradient.Gradient(
            direc=self.direc, ff=self.ff,
            ff_lines=self.ff.lines, args_ff=self.args_ff,
            args_ref=self.args_ref, calculator=self._get_calculator(),
        )
        for opt_token in cols[1:]:
            _apply_method_token(grad, opt_token)
        self.ff = grad.run(ref_data=self.ref_data)

    def _handle_simp(self, cols):
        import simplex
        simp = simplex.Simplex(
            direc=self.direc, ff=self.ff,
            ff_lines=self.ff.lines, args_ff=self.args_ff,
            args_ref=self.args_ref, calculator=self._get_calculator(),
        )
        for opt_token in cols[1:]:
            if "max_params" in opt_token:
                simp.max_params = int(opt_token.split("=")[1])
            else:
                raise ValueError("SIMP: unrecognised option '{}'".format(opt_token))
        self.ff = simp.run(r_data=self.ref_data)

    def _handle_hybr(self, cols):
        import opt as opt_module
        kwargs = parse_hybr_options(cols[1:])
        if self.swarm is None:
            # First HYBR of this LOOP block: a new swarm around the current
            # FF, hyperparameters tapering over the cycle.
            self.swarm = opt_module.SwarmOptimizer(
                direc=self.direc, ff=self.ff,
                ff_lines=self.ff.lines, args_ff=self.args_ff,
                args_ref=self.args_ref, calculator=self._get_calculator(),
                **kwargs
            )
            strategy = "exp_decay"
        else:
            # Later cycles continue the same swarm from where it stopped,
            # around the best FF so far, with the hyperparameters frozen
            # at their final values. The options were read when the swarm
            # was created.
            self.swarm.ff = self.ff
            strategy = ""
        try:
            self.ff = self.swarm.run(ref_data=self.ref_data,
                                     precision=self.convergence,
                                     strategy=strategy)
        finally:
            self._dump_swarm_history(self.swarm)

    def _dump_swarm_history(self, swarm):
        """Persist the swarm history to hybrid_opt_history.bin.

        PSO_DE accumulates every particle position (X) and score (Y) per
        iteration in record_value, but nothing else writes it to disk, so
        without this the whole history dies with the process. Written in a
        `finally` so a crashed or interrupted run still leaves the partial
        history behind.
        """
        if swarm.hybrid_opt is None:
            logger.warning("HYBR: no optimizer to dump history from")
            return
        history = swarm.hybrid_opt.record_value
        # record_value starts as {"X": [], "V": [], "Y": []}, so test X rather
        # than the dict itself. Writing a history with no iterations would
        # produce a file that indexing X[0]/Y[0] then blows up on.
        if not history["X"]:
            logger.warning("HYBR: optimizer recorded no iterations")
            return
        path = os.path.join(self.direc, "hybrid_opt_history.bin")
        try:
            with open(path, "wb") as fh:
                pickle.dump(history, fh)
        except Exception as e:
            # Never let a failed dump take down an otherwise good run.
            logger.warning("HYBR: could not write %s: %s", path, e)
            return
        logger.log(20, "HYBR history: {} records -> {}".format(
            len(history["X"]), path))


# ---------------------------------------------------------------------------
# HYBR command option parsing
# ---------------------------------------------------------------------------

class _HybrOptionParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("HYBR: {}".format(message))


def parse_hybr_options(tokens):
    """
    Parse "--max_iter N --pop_size N --tight T|F --n_processes N" into
    SwarmOptimizer keyword arguments. Every option is optional; an unknown
    one is an error. The LOOP convergence, not an option, is the swarm's
    precision.
    """
    parser = _HybrOptionParser(prog="HYBR", add_help=False)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=24)
    parser.add_argument("--tight", type=str, default="true")
    parser.add_argument("--n_processes", type=int, default=1)
    opts, unknown = parser.parse_known_args(tokens)
    if unknown:
        raise ValueError("HYBR: unrecognised option(s) {}".format(" ".join(unknown)))
    return {
        "max_iter": opts.max_iter,
        "pop_size": opts.pop_size,
        "tight_spread": opts.tight.lower() in ("t", "true", "1", "yes"),
        "n_processes": opts.n_processes,
    }


# ---------------------------------------------------------------------------
# GRAD command option parsing
# ---------------------------------------------------------------------------

def _parse_value_list(s):
    """Parse a substring like "[1./10.]" into [1.0, 10.0]."""
    m = re.search(r"\[(.+)\]", s)
    if not m:
        return None
    inner = m.group(1)
    if inner == "None":
        return None
    return [float(x) for x in inner.split("/")]


def _apply_method_token(grad, token):
    """Apply a single GRAD option such as "lstsq=True,radii=[1./10.]"."""
    if "=" not in token:
        raise ValueError("GRAD option lacks '=': {}".format(token))
    method, args_str = token.split("=", 1)
    args = args_str.split(",")

    if method == "lstsq":
        prefix = "lstsq"
    elif method == "newton":
        prefix = "newton"
    elif method == "lagrange":
        prefix = "lagrange"
    elif method == "levenberg":
        prefix = "levenberg"
    elif method == "svd":
        prefix = "svd"
    else:
        raise ValueError("'{}' : Not Recognized".format(token))

    enabled = False
    for a in args:
        if a == "True":
            enabled = True
        elif a == "False":
            enabled = False
        elif "radii" in a:
            setattr(grad, prefix + "_radii", _parse_value_list(a))
        elif "cutoff" in a:
            vals = _parse_value_list(a)
            if vals is not None and len(vals) != 2:
                raise Exception("Cutoff values must be exactly two numbers.")
            setattr(grad, prefix + "_cutoffs", vals)
        elif "factor" in a:
            setattr(grad, prefix + "_factors", _parse_value_list(a))
    setattr(grad, "do_" + prefix, enabled)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def pretty_loop_input(lines, name="Q2MM", score=None):
    logger.log(20, " {} ".format(name).center(79, "="))
    logger.log(20, "COMMANDS:")
    for line in lines:
        logger.log(20, "> " + line)
    if score is not None:
        logger.log(20, "SCORE: {}".format(score))
    logger.log(20, "=" * 79)


def pretty_loop_summary(cycle_num, score_value, change):
    logger.log(20, " Cycle {} Summary ".format(cycle_num).center(50, "-"))
    logger.log(20, "| PF Score: {:36.15f} |".format(score_value))
    logger.log(20, "| % change: {:36.15f} |".format(change * 100.0))
    logger.log(20, "-" * 50)


# ---------------------------------------------------------------------------
# Input-file parsing
# ---------------------------------------------------------------------------

def read_loop_input(filename):
    """Strip comments and blank lines from loop.in."""
    with open(filename, "r") as f:
        raw = f.readlines()
    lines = [x.partition("#")[0].strip("\n") for x in raw]
    lines = [x for x in lines if x.strip() != ""]
    pretty_loop_input(lines)
    return lines


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str,
                        help="Path to loop.in")
    opts = parser.parse_args(argv)
    lines = read_loop_input(opts.input)
    loop = Loop()
    loop.run_loop_input(lines)


if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
