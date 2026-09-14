#!/usr/bin/env python
"""
calculators
-----------
The interface between the optimizers and an external molecular-mechanics
engine. An optimizer never touches engine files: it hands a data_structs.FF
to a Calculator and gets back a list of data_structs.Datum, so data_structs
is the universal language on both sides of this module.

    Calculator          abstract engine interface
    AmberCalculator     runs tleap / sander / cpptraj / nab for one system
    CalculatorGroup     several calculators driven as one (several .in files)

A Calculator does three things, which can also be run one at a time:

    update_ff(ff)       write the force field to disk (through utilities)
    calculate()         run the engine for self.commands
    gather_results()    read the engine's output files back into Datum objects

evaluate(ff) chains the three. evaluate_many(ffs) does it for a whole swarm
of trial force fields: with n_processes > 1 every trial gets its own copy of
the working directory (so concurrent engine runs cannot collide on shared
files) and the runs go through a multiprocessing pool that the calculator
owns; the optimizer only ever sees data_structs objects.

Every engine command goes through the calculator's ``runner`` callable
(default: run_shell), so a test can stand in a fake Amber that writes canned
output files and no Amber installation is needed to unit test this module.

Results come back as data_structs objects (datums_from_hessian,
datums_from_geometry, datum_from_energy). The calculator stamps what it knows
from the topology, such as how many bonds apart the two atoms of a Hessian
element are, and leaves weights to score.py.
"""
from __future__ import absolute_import
from __future__ import division

import logging
import logging.config
import multiprocessing
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List

import numpy as np

import constants as co
import math_util
from data_structs import (AmberFF, Datum, Structure, datum_from_energy,
                          datums_from_eigenmatrix, datums_from_geometry,
                          datums_from_hessian)
from utilities import AmberLeapInput, AmberUtilities, Frcmod

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


class CalculationError(Exception):
    """An engine step did not produce the files the next step needs."""


def run_shell(command, cwd, log_path):
    """
    Default runner: run one engine command in `cwd` with stdout and stderr
    appended to `log_path`. Returns the exit status.
    """
    with open(log_path, "a") as log:
        log.write("$ {}\n".format(command))
        log.flush()
        return subprocess.call(command, shell=True, cwd=cwd,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=log)


# ---------------------------------------------------------------------------
# Parallel evaluation. multiprocessing.Pool pickles the callable per task, so
# the worker has to be a module-level function; the calculator clone and the
# trial FF travel to the worker inside the task tuple.
# ---------------------------------------------------------------------------

def _evaluate_task(task):
    """
    One particle: (calculator, ff, reducer) -> reducer(data) or data.
    A failure is logged and comes back as None -- never a silent bad value,
    but never a crash of the whole swarm either.
    """
    calc, ff, reducer = task
    try:
        data = calc.evaluate(ff)
        return reducer(data) if reducer is not None else data
    except Exception as e:
        logger.warning("Evaluation failed in %s: %s", calc.work_dir, e)
        return None


class Calculator(ABC):
    """
    Manages the methods, inputs and files necessary to run back-end
    calculations with an external engine like AMBER, and acts as the
    interface with that engine. Subclasses set `ff_type` to the
    data_structs.FF subclass they understand, so Calculator.for_ff can pick
    the right engine for a force field.
    """

    ff_type = None

    def __init__(self, work_dir, commands, n_processes=1):
        """
        Args:
            work_dir (str): directory holding the engine's input files; the
                engine runs here and writes its scratch under it.
            commands (List[str]): what to compute, in the engine's own
                vocabulary (for Amber, the CDAT flags: "ah", "abo", ...).
            n_processes (int): parallel evaluations for evaluate_many;
                1 (default) evaluates serially, in place.
        """
        self.work_dir = os.path.abspath(work_dir)
        self.commands = list(commands)
        self.n_processes = int(n_processes)

    @classmethod
    def for_ff(cls, ff, *args, **kwargs):
        """
        Build the Calculator subclass registered for type(ff), e.g.
        Calculator.for_ff(amber_ff, work_dir, "MOL.in", ["ah"]) is an
        AmberCalculator. A future TinkerCalculator would set
        ff_type = TinkerFF and be picked here for a TinkerFF.
        """
        for subclass in _all_subclasses(cls):
            if subclass.ff_type is not None and isinstance(ff, subclass.ff_type):
                return subclass(*args, ff=ff, **kwargs)
        raise TypeError("No Calculator knows how to run a {}".format(type(ff).__name__))

    # -- the three steps ----------------------------------------------------

    @abstractmethod
    def update_ff(self, ff):
        """Write `ff` to the engine's force-field file (through utilities)."""

    @abstractmethod
    def calculate(self):
        """Run the engine for self.commands. Raises CalculationError."""

    @abstractmethod
    def gather_results(self) -> List[Datum]:
        """Read the engine's output files back into Datum objects."""

    @abstractmethod
    def clone_into(self, work_dir):
        """A copy of this calculator that works in `work_dir`, with the input files copied there."""

    # -- what the optimizers call ------------------------------------------

    def evaluate(self, ff=None) -> List[Datum]:
        """
        update_ff (when an FF is given) -> calculate -> gather_results.
        An engine failure is logged and yields [] (no data), which the
        scoring treats as an incomplete calculation (score inf).
        """
        if ff is not None:
            self.update_ff(ff)
        try:
            self.calculate()
        except CalculationError as e:
            logger.warning("Calculation failed in {}: {}".format(self.work_dir, e))
            return []
        return self.gather_results()

    def evaluate_many(self, ffs, reducer=None, pool_dir=None, n_processes=None):
        """
        Evaluate several trial force fields -- one particle each -- and
        return their results in order.

        Serial (n_processes <= 1): each FF is evaluated in place, in turn.
        Parallel: particle i runs in <pool_dir>/p_<i> (default
        <work_dir>/swarm_particles), a copy of the working directory made by
        clone_into, on a multiprocessing pool of n_processes workers
        (default self.n_processes).

        `reducer(data)`, when given, is applied to each particle's Datum list
        inside the worker and its value returned instead of the data (so a
        score, not a Hessian's worth of Datum objects, crosses the process
        boundary). It must be picklable -- a module-level function. A
        particle whose evaluation raised comes back as None.
        """
        ffs = list(ffs)
        if n_processes is None:
            n_processes = self.n_processes
        n_workers = min(int(n_processes), len(ffs))
        if n_workers <= 1:
            return [_evaluate_task((self, ff, reducer)) for ff in ffs]
        base = pool_dir or os.path.join(self.work_dir, "swarm_particles")
        tasks = []
        for i, ff in enumerate(ffs):
            clone = self.clone_into(os.path.join(base, "p_{:03d}".format(i)))
            tasks.append((clone, ff, reducer))
        logger.log(20, "Evaluating {} force fields on {} workers under {}".format(
            len(ffs), n_workers, base))
        with multiprocessing.Pool(n_workers) as pool:
            return pool.map(_evaluate_task, tasks)


def _all_subclasses(cls):
    found = []
    for sub in cls.__subclasses__():
        found.append(sub)
        found.extend(_all_subclasses(sub))
    return found


# ---------------------------------------------------------------------------
# AMBER
# ---------------------------------------------------------------------------

# The Amber command vocabulary: the CDAT flags without the leading "-".
#   command -> (datum type, minimize, geometry, hessian)
# minimize: sander minimization (maxcyc 700) before measuring, else a single
# point (maxcyc 0). geometry: measure bonds/angles/torsions with cpptraj.
# hessian: nab/nmode Hessian at the mol2 geometry. "ageig" is that Hessian
# projected on the reference normal modes (eigenmode fitting; needs the
# calculator's `eigenvectors`). gather_results emits data in this table's
# order.
AMBER_COMMANDS = OrderedDict([
    ("ah",    ("h",   True,  True,  True)),
    ("ageig", ("eig", True,  False, True)),
    ("ae",   ("e",   False, False, False)),
    ("ae1",  ("e1",  False, False, False)),
    ("aeo",  ("eo",  True,  False, False)),
    ("ae1o", ("e1o", True,  False, False)),
    ("abo",  ("b",   True,  True,  False)),
    ("aao",  ("a",   True,  True,  False)),
    ("ato",  ("t",   True,  True,  False)),
    ("ab",   ("b",   False, True,  False)),
    ("aa",   ("a",   False, True,  False)),
    ("at",   ("t",   False, True,  False)),
])

# Reference-side geometry: the QM geometry in the mol2 measured with the same
# cpptraj enumeration as the calculated side, so the two align one-for-one.
REFERENCE_GEOMETRY_COMMANDS = {"gabo": "ab", "gaao": "aa", "gato": "at"}

ENERGY_TYPES = ("e", "e1", "eo", "e1o")


class AmberCalculator(Calculator):
    """
    Runs Amber for one system: the tleap script `leap_input` (MOL.in) in
    `work_dir`, which loads the mol2 and the frcmod being fitted and saves
    calc/prmtop + calc/inpcrd.

    The pipeline, per calculate():
        tleap                      calc/prmtop, calc/inpcrd   (update_topology)
        sander (min or sp)         calc/<prefix>.<name>.ene, .rst
        sander 0-step MD + cpptraj calc/<prefix>.<name>.geo   (geometry commands)
        antechamber + nab/nmode    calc/<prefix>.<name>.hes   (ah / ageig)
    Everything Amber prints goes to <work_dir>/<prefix>.<name>.log.

    "ah" reports the Hessian element by element; "ageig" reports it projected
    on the reference normal modes given as `eigenvectors` (eigenmode fitting).

    `prefix` is "amber" for calculated data. calculate.py uses "gaus" with
    the single-point geometry commands (ab/aa/at) to measure the reference
    geometry through the same machinery.
    """

    ff_type = AmberFF
    PRMTOP = os.path.join("calc", "prmtop")
    INPCRD = os.path.join("calc", "inpcrd")
    MIN_CYCLES = (700, 5)   # maxcyc, ncyc for minimized ("...o") commands
    SP_CYCLES = (0, 0)
    MAX_RESTARTS = 3

    def __init__(self, work_dir, leap_input, commands, ff=None, invert=None,
                 prefix="amber", n_processes=1, runner=None, src_name=None,
                 eigenvectors=None):
        """
        Args:
            work_dir (str): directory holding the leap input, mol2 and frcmod.
            leap_input (str): the tleap script; its basename is used, the
                file lives in work_dir.
            commands (List[str]): AMBER_COMMANDS keys, e.g. ["ah"]. All must
                agree on minimize vs single point (one sander run).
            ff (AmberFF, optional): the force field being fitted; update_ff
                writes trial force fields to ff.path on ff.lines. Without it
                the frcmod on disk is used as is.
            invert (float, optional): replace the most-negative Hessian
                eigenvalue with this value (the -i flag).
            prefix (str): output-file prefix, "amber" or "gaus".
            n_processes (int): workers for evaluate_many.
            runner (callable, optional): runner(command, cwd, log_path) ->
                exit status; default run_shell. Tests inject a fake Amber.
            src_name (str, optional): Datum src_1 for energies and geometry;
                default the leap input's basename.
            eigenvectors (np.ndarray, optional): the reference normal modes
                (n_modes x 3N, normalized, mass-weighted) for "ageig".
        """
        super(AmberCalculator, self).__init__(work_dir, commands, n_processes)
        unknown = [c for c in self.commands if c not in AMBER_COMMANDS]
        if unknown:
            raise ValueError("Unknown Amber command(s) {}; known: {}".format(
                unknown, list(AMBER_COMMANDS)))
        minimize = {AMBER_COMMANDS[c][1] for c in self.commands}
        if len(minimize) > 1:
            raise ValueError(
                "Commands {} mix minimized and single-point calculations; "
                "use one AmberCalculator per kind.".format(self.commands))
        self.minimize = minimize.pop() if minimize else False
        self.leap_input = AmberLeapInput(os.path.join(self.work_dir, os.path.basename(leap_input)))
        self.name = self.leap_input.name
        self.prefix = prefix
        self.invert = invert
        self.runner = runner
        self.src_name = src_name or self.leap_input.filename
        self.eigenvectors = None if eigenvectors is None else np.asarray(eigenvectors, dtype=float)
        if "ageig" in self.commands and self.eigenvectors is None:
            raise ValueError("The ageig command needs the reference eigenvectors.")
        self.frcmod = None
        if ff is not None:
            if ff.path is None:
                raise ValueError("The force field needs a path to be written to.")
            self.frcmod = Frcmod(ff.path, force_field=ff, lines=ff.lines)

    # -- file names ---------------------------------------------------------

    @property
    def calc_dir(self):
        return os.path.join(self.work_dir, "calc")

    @property
    def log_path(self):
        return os.path.join(self.work_dir, "{}.{}.log".format(self.prefix, self.name))

    def _calc_name(self, ext):
        """calc/<prefix>.<name>.<ext>, relative to work_dir (Amber runs there)."""
        return os.path.join("calc", "{}.{}.{}".format(self.prefix, self.name, ext))

    @property
    def name_ene(self):
        return "{}.{}.ene".format(self.prefix, self.name)

    @property
    def name_geo(self):
        return "{}.{}.geo".format(self.prefix, self.name)

    @property
    def name_hes(self):
        return "{}.{}.hes".format(self.prefix, self.name)

    @property
    def needs_geometry(self):
        return any(AMBER_COMMANDS[c][2] for c in self.commands)

    @property
    def needs_hessian(self):
        return any(AMBER_COMMANDS[c][3] for c in self.commands)

    def _path(self, rel):
        return os.path.join(self.work_dir, rel)

    def _write(self, rel, text):
        with open(self._path(rel), "w") as f:
            f.write(text)

    def _run(self, command):
        runner = self.runner if self.runner is not None else run_shell
        logger.log(5, "RUNNING in {}: {}".format(self.work_dir, command))
        status = runner(command, self.work_dir, self.log_path)
        if status:
            logger.warning("'{}' exited with status {} (see {})".format(
                command, status, self.log_path))
        return status

    # -- the three steps ----------------------------------------------------

    def update_ff(self, ff):
        """Write the trial force field `ff` to the frcmod the leap input loads."""
        if self.frcmod is None:
            if ff.path is None:
                raise ValueError("The force field needs a path to be written to.")
            self.frcmod = Frcmod(ff.path, force_field=ff, lines=ff.lines)
        self.frcmod.write_ff(ff)

    def update_topology(self):
        """
        Rebuild calc/prmtop and calc/inpcrd with tleap. The old files are
        deleted first so a failing leap script cannot leave a stale topology
        behind; both must exist afterwards.
        """
        for rel in (self.PRMTOP, self.INPCRD):
            if os.path.isfile(self._path(rel)):
                os.remove(self._path(rel))
        self._run("tleap -f {}".format(self.leap_input.filename))
        missing = [rel for rel in (self.PRMTOP, self.INPCRD)
                   if not os.path.isfile(self._path(rel))]
        if missing:
            raise CalculationError("tleap -f {} did not produce {} (see {})".format(
                self.leap_input.filename, ", ".join(missing), self.log_path))

    def calculate(self):
        """Run the Amber pipeline for self.commands."""
        os.makedirs(self.calc_dir, exist_ok=True)
        open(self.log_path, "w").close()
        logger.log(logging.DEBUG, "  CALCULATE {} in {}: {}".format(
            self.commands, self.work_dir, self.leap_input.filename))
        self.update_topology()
        self._run_sander()
        if self.needs_geometry:
            self._measure_geometry()
        if self.needs_hessian:
            self._compute_hessian()

    def gather_results(self) -> List[Datum]:
        """Datum objects for every command, in AMBER_COMMANDS order."""
        data = []
        for command, (typ, _, _, _) in AMBER_COMMANDS.items():
            if command not in self.commands:
                continue
            if typ == "h":
                data.extend(self._hessian_data())
            elif typ == "eig":
                data.extend(self._eigenmatrix_data())
            elif typ in ENERGY_TYPES:
                data.extend(self._energy_data(typ))
            else:
                data.extend(self._geometry_data(typ))
        return data

    def clone_into(self, work_dir):
        """
        A calculator for the same system working in `work_dir`: the leap
        input, every file it names, the mol2/pdb and the frcmod are copied
        there so a concurrent run cannot collide with this one.
        """
        work_dir = os.path.abspath(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        for rel in self.input_files():
            dst = os.path.join(work_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(self._path(rel), dst)
        clone = AmberCalculator(work_dir, self.leap_input.filename, self.commands,
                                invert=self.invert, prefix=self.prefix,
                                runner=self.runner, src_name=self.src_name,
                                eigenvectors=self.eigenvectors)
        if self.frcmod is not None:
            clone.frcmod = Frcmod(os.path.join(work_dir, self.frcmod.filename),
                                  force_field=self.frcmod.force_field,
                                  lines=self.frcmod.lines)
        return clone

    def input_files(self):
        """Input files of this system, relative to work_dir."""
        files = [self.leap_input.filename]
        for rel in self.leap_input.referenced_files():
            if rel not in files:
                files.append(rel)
        for ext in (".mol2", ".pdb"):
            rel = self.name + ext
            if rel not in files and os.path.isfile(self._path(rel)):
                files.append(rel)
        if self.frcmod is not None:
            rel = os.path.relpath(self.frcmod.path, self.work_dir)
            if rel not in files and os.path.isfile(self.frcmod.path):
                files.append(rel)
        return files

    # -- pipeline steps ------------------------------------------------------

    def _run_sander(self):
        """Minimize (or single-point) from calc/inpcrd; writes .ene and .rst."""
        maxcyc, ncyc = self.MIN_CYCLES if self.minimize else self.SP_CYCLES
        min_in = self._calc_name("min")
        ene = self._calc_name("ene")
        rst = self._calc_name("rst")
        self._write(min_in, AmberUtilities.sander_min_input(maxcyc, ncyc))
        self._run("sander -O -i {} -o {} -p {} -c {} -r {}".format(
            min_in, ene, self.PRMTOP, self.INPCRD, rst))
        # sander sometimes asks to be restarted; continue from the restart file.
        for _ in range(self.MAX_RESTARTS):
            if not self._sander_wants_restart(ene):
                break
            logger.warning("sander asked for a restart in {}; restarting.".format(ene))
            self._run("sander -O -i {} -o {} -p {} -c {} -r {}".format(
                min_in, ene, self.PRMTOP, rst, rst))

    def _sander_wants_restart(self, ene):
        path = self._path(ene)
        if not os.path.isfile(path):
            return False
        with open(path, "r") as f:
            return any("restarting should resolve the error" in line for line in f)

    def _measure_geometry(self):
        """
        Write the minimized (or single-point) coordinates as a trajectory
        frame, list every bond/angle/dihedral of the topology with cpptraj,
        measure them all on that frame, and leave the BONDS/ANGLES/TORSIONS
        summary in calc/<prefix>.<name>.geo for AmberGeo.
        """
        dyn = self._calc_name("dyn")
        rst = self._calc_name("rst")
        nc = self._calc_name("nc")
        listing_in = self._calc_name("int")
        geo = self._calc_name("geo")
        self._write(dyn, AmberUtilities.sander_traj_input())
        self._run("sander -O -i {} -o calc/traj.out -p {} -c {} -x {}".format(
            dyn, self.PRMTOP, rst, nc))
        self._write(listing_in, AmberUtilities.cpptraj_list_input())
        self._run("cpptraj -p {} < {} > {}".format(self.PRMTOP, listing_in, geo))
        if not os.path.isfile(self._path(geo)):
            raise CalculationError("cpptraj produced no interaction listing {}".format(geo))
        with open(self._path(geo), "r") as f:
            bonds, angles, torsions = AmberUtilities.parse_cpptraj_listing(f.readlines())
        out_prefix = os.path.join("calc", self.prefix)
        measure_in = os.path.join("calc", self.name + ".temp")
        self._write(measure_in, AmberUtilities.cpptraj_measure_input(
            nc, bonds, angles, torsions, out_prefix))
        self._run("cpptraj -p {} < {}".format(self.PRMTOP, measure_in))
        summary = AmberUtilities.geo_summary(
            bonds, angles, torsions,
            AmberUtilities.read_measurements(self._path(out_prefix + ".bonds")),
            AmberUtilities.read_measurements(self._path(out_prefix + ".angles")),
            AmberUtilities.read_measurements(self._path(out_prefix + ".torsions")))
        self._write(geo, summary)

    def _compute_hessian(self):
        """
        Mass-weighted Hessian with nab/nmode at the mol2 geometry (the QM
        transition state, via <name>.pdb), on the current topology. Needs
        the nmode patch that writes calc/hessian.mat (OPTIMIZATION.md
        section 0); the file is moved to calc/<prefix>.<name>.hes.
        """
        pdb = self.name + ".pdb"
        if not os.path.isfile(self._path(pdb)):
            self._run("antechamber -dr no -i {0}.mol2 -fi mol2 -o {0}.pdb -fo pdb".format(self.name))
        nab_in = os.path.join("calc", self.name + ".nab")
        nab_bin = os.path.join("calc", self.name)
        self._write(nab_in, AmberUtilities.nab_hessian_input(pdb, "./" + self.PRMTOP))
        self._run("nab -v {} -o {}".format(nab_in, nab_bin))
        self._run("./" + nab_bin)
        produced = self._path(os.path.join("calc", "hessian.mat"))
        if os.path.isfile(produced):
            os.replace(produced, self._path(self._calc_name("hes")))
        else:
            logger.warning("nmode wrote no calc/hessian.mat in {} (AmberTools "
                           "not patched? see {})".format(self.work_dir, self.log_path))

    # -- results ---------------------------------------------------------------

    def _read_hessian(self):
        """The mass-weighted Hessian nmode wrote, in kJ/mol/A^2/amu; None if missing."""
        hes_path = os.path.join(self.calc_dir, self.name_hes)
        if not os.path.isfile(hes_path):
            logger.warning("Hessian file missing: {}".format(hes_path))
            return None
        # AmberHess parses the file, converts kcal/mol -> kJ/mol, and returns
        # a 3N x 3N matrix that is already mass-weighted by nab/nmode.
        return AmberUtilities.read_hessian(hes_path).hessian

    def _eigenmatrix_data(self):
        """The Hessian projected on the reference normal modes, one Datum per element."""
        H = self._read_hessian()
        if H is None:
            return []
        try:
            matrix = math_util.project_hessian(H, self.eigenvectors)
        except ValueError as e:
            logger.warning("Cannot project the Hessian of {} on the reference "
                           "modes: {}".format(self.name_hes, e))
            return []
        return datums_from_eigenmatrix(matrix, self.name_hes)

    def _hessian_data(self):
        H = self._read_hessian()
        if H is None:
            return []
        if self.invert is not None:
            H = math_util.invert_lowest_eigenvalue(H, self.invert, label=self.name_hes)
        structure = self._measured_structure()
        if structure is None:
            logger.warning("No geometry summary in {}; every off-diagonal Hessian "
                           "element counts as long range.".format(self.calc_dir))
            structure = Structure(self.name_geo)
        # The .hes filename makes the Datum label collapse to "amber" (the
        # piece before the first dot), matching q2mm-master's label format.
        return datums_from_hessian(H, self.name_hes, structure=structure)

    def _measured_structure(self):
        """
        The structure cpptraj measured (calc/<prefix>.<name>.geo); its bonds,
        angles and torsions give the Hessian elements' atom-pair separations.
        None when the summary is missing.
        """
        geo_path = os.path.join(self.calc_dir, self.name_geo)
        if not os.path.isfile(geo_path):
            return None
        structures = AmberUtilities.read_geometry(geo_path).structures
        return structures[0] if structures else None

    def _energy_data(self, typ):
        ene_path = os.path.join(self.calc_dir, self.name_ene)
        if not os.path.isfile(ene_path):
            logger.warning("Energy file missing: {}".format(ene_path))
            return []
        data = []
        for i, s in enumerate(AmberUtilities.read_energies(ene_path).structures):
            if "energy" in s.props:
                data.append(datum_from_energy(s.props["energy"], i + 1, self.src_name, typ=typ))
        return data

    def _geometry_data(self, kind):
        geo_path = os.path.join(self.calc_dir, self.name_geo)
        if not os.path.isfile(geo_path):
            logger.warning("Geo file missing: {}".format(geo_path))
            return []
        return datums_from_geometry(AmberUtilities.read_geometry(geo_path).structures,
                                    kind, self.src_name)


class CalculatorGroup(Calculator):
    """
    Several calculators driven as one, e.g. one AmberCalculator per leap
    input named on a CDAT line. Results are concatenated in member order.
    """

    def __init__(self, calculators, n_processes=1):
        members = list(calculators)
        if not members:
            raise ValueError("CalculatorGroup needs at least one calculator.")
        super(CalculatorGroup, self).__init__(
            members[0].work_dir, [c for m in members for c in m.commands], n_processes)
        self.members = members
        self.ff_type = members[0].ff_type

    def update_ff(self, ff):
        for member in self.members:
            member.update_ff(ff)

    def calculate(self):
        for member in self.members:
            member.calculate()

    def gather_results(self):
        data = []
        for member in self.members:
            data.extend(member.gather_results())
        return data

    def evaluate(self, ff=None):
        # Each member reports its own failure and contributes nothing.
        data = []
        for member in self.members:
            data.extend(member.evaluate(ff))
        return data

    def clone_into(self, work_dir):
        return CalculatorGroup([m.clone_into(work_dir) for m in self.members])
