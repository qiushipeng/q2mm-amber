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
CDAT <args ...>           Calculate FF data with current FF parameters.
COMP [-o out] [-p]        Score reference vs calculated; write pretty table.
LOOP <conv> ... END       Iterate the enclosed block until score change
                          < conv. Block typically contains GRAD, SIMP,
                          or SWARM commands.
GRAD [opts ...]           Run gradient.Gradient.run(). Options use the
                          old grammar, eg "lstsq=True,radii=[1./10.]".
SIMP [max_params=N]       Run simplex.Simplex.run().
SWARM [opts ...]          Run opt.SwarmOptimizer.run(). Options:
                          max_iter=N pop_size=N precision=F tight=T|F
WGHT <typ> <weight>       Override constants.WEIGHTS[typ] (eg "WGHT b 100.").
STEP <ptype> <step>       Override constants.STEPS[ptype].
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
import re
import shutil
import sys

import constants as co
import data_structs
import parameters as parameters_module
import score

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

    # -- inner LOOP block ----------------------------------------------------

    def opt_loop(self):
        """
        Execute the inner LOOP / END block repeatedly until
        |last_score - new_score| / last_score < self.convergence.
        Backs up the current best FF to mm3_NNN.fld after each cycle.
        """
        change = None
        last_score = None
        if self.ff.score is None:
            logger.warning("No initial score; computing one to seed loop.")
            import calculate
            self.ff.export_ff()
            self.ff.data = calculate.main(self.args_ff)
            r_dict = score.data_by_type(self.ref_data)
            c_dict = score.data_by_type(self.ff.data)
            r_dict, c_dict = score.trim_data(r_dict, c_dict)
            self.ff.score = score.compare_data(r_dict, c_dict)

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

            backup_files = glob.glob(os.path.join(self.direc, "mm3_???.fld"))
            if backup_files:
                backup_files.sort()
                last_num = int(os.path.basename(backup_files[-1])[4:7])
                backup = os.path.join(self.direc, "mm3_{:03d}.fld".format(last_num + 1))
            else:
                backup = os.path.join(self.direc, "mm3_001.fld")
            self.ff.export_ff(path=backup)
            logger.log(20, "  -- Wrote best FF to {}".format(backup))

        for p in self.ff.params:
            p.value_at_limits()
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
                import calculate
                self.ff.data = calculate.main(self.args_ff)

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
                inner_loop.loop_lines = inner
                pretty_loop_input(inner, name="OPTIMIZATION LOOP",
                                  score=self.ff.score)
                self.ff = inner_loop.opt_loop()

            elif cmd == "GRAD":
                self._handle_grad(cols)

            elif cmd == "SIMP":
                self._handle_simp(cols)

            elif cmd == "SWARM":
                self._handle_swarm(cols)

            elif cmd == "WGHT":
                co.WEIGHTS[cols[1]] = float(cols[2])
                logger.log(20, "WGHT {} = {}".format(cols[1], cols[2]))

            elif cmd == "STEP":
                co.STEPS[cols[1]] = float(cols[2])
                logger.log(20, "STEP {} = {}".format(cols[1], cols[2]))

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
                self.ff = data_structs.AmberFF(full)
            else:
                raise ValueError(
                    "Only frcmod FFs supported in q2mm-amber-main "
                    "(saw {}).".format(target))
            self.ff.import_ff()
            self.ff.method = "READ"
            with open(full, "r") as f:
                self.ff.lines = f.readlines()
            logger.log(20, "FFLD read {}: {} parameters".format(full, len(self.ff.params)))
        elif action == "write":
            self.ff.export_ff(full)
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
        r_dict = score.data_by_type(self.ref_data)
        c_dict = score.data_by_type(self.ff.data)
        r_dict, c_dict = score.trim_data(r_dict, c_dict)
        self.ff.score = score.compare_data(
            r_dict, c_dict, output=out, doprint=do_print)
        logger.log(20, "COMP score: {}".format(self.ff.score))

    def _handle_grad(self, cols):
        import gradient
        grad = gradient.Gradient(
            direc=self.direc, ff=self.ff,
            ff_lines=self.ff.lines, args_ff=self.args_ff,
            args_ref=self.args_ref,
        )
        for opt_token in cols[1:]:
            _apply_method_token(grad, opt_token)
        self.ff = grad.run(ref_data=self.ref_data)

    def _handle_simp(self, cols):
        import simplex
        simp = simplex.Simplex(
            direc=self.direc, ff=self.ff,
            ff_lines=self.ff.lines, args_ff=self.args_ff,
            args_ref=self.args_ref,
        )
        for opt_token in cols[1:]:
            if "max_params" in opt_token:
                simp.max_params = int(opt_token.split("=")[1])
            else:
                raise ValueError("SIMP: unrecognised option '{}'".format(opt_token))
        self.ff = simp.run(r_data=self.ref_data)

    def _handle_swarm(self, cols):
        import opt as opt_module
        kwargs = {}
        for opt_token in cols[1:]:
            if "=" not in opt_token:
                continue
            k, v = opt_token.split("=", 1)
            if k == "max_iter":
                kwargs["max_iter"] = int(v)
            elif k == "pop_size":
                kwargs["pop_size"] = int(v)
            elif k == "precision":
                kwargs["precision"] = float(v)
            elif k == "tight":
                kwargs["tight_spread"] = v.lower() in ("t", "true", "1", "yes")
        swarm = opt_module.SwarmOptimizer(
            direc=self.direc, ff=self.ff,
            ff_lines=self.ff.lines, args_ff=self.args_ff,
            args_ref=self.args_ref,
            **kwargs
        )
        self.ff = swarm.run(ref_data=self.ref_data)


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
