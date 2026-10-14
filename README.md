# q2mm-amber

q2mm-amber does quantum-guided molecular mechanics (Q2MM) force-field
parameterization on the **AMBER** path. It fits GAFF/GAFF2 `frcmod` force
fields to Gaussian reference data. It builds and evaluates each force field
with `tleap`, `sander`, `cpptraj` and `nab`.

The main use is to build **transition-state force fields (TSFFs)**. A TSFF
holds a molecule at a QM transition state (TS). You can then use the TS in
usual MM and MD calculations, also inside an enzyme.

---

## Scope

Upstream [Q2MM](https://github.com/Q2MM/q2mm) uses the Schrödinger MM3\*
force field (`mm3.fld`) and MacroModel. This package is the AMBER equivalent:

| | upstream Q2MM | q2mm-amber |
|---|---|---|
| force field | MM3\* `mm3.fld` | GAFF/GAFF2 `frcmod` |
| MM engine | MacroModel (`bmin`) | `tleap` + `sander` + `cpptraj` + `nab`/`nmode` |
| requires Schrödinger | yes | **no** |
| reference data | Jaguar / Gaussian | Gaussian |

The command words (`DIR`, `FFLD`, `PARM`, `RDAT`, `CDAT`, `COMP`,
`LOOP`/`END`, `GRAD`, `SIMP`, `HYBR`) are the same as in upstream Q2MM on
purpose. Thus loop files and workflows look the same in the two packages.

---

## Documentation

| document | contents |
|---|---|
| **[`tools/README.md`](tools/README.md)** | **Start here.** How to build the initial TSFF with `build_tsff.py`: atom types, parameter sources, outputs, and the changes that the fit needs. How to read a `HYBR` history with `extract_swarm.py`. |
| **[`OPTIMIZATION.md`](OPTIMIZATION.md)** | How to run the fit: the AmberTools patch, the run directory, the parameter file, `loop.in`, Hessian and eigenmode fitting, the `GRAD` and `HYBR` optimizers, outputs, and troubleshooting. |

---

## Requirements

* **Python 3** with `numpy`. A conda environment is sufficient. You do not
  need Schrödinger.
* **AmberTools**: `tleap`, `sander`, `antechamber`, `parmchk2`, `cpptraj`.
* **AmberClassic** (or an AmberTools build) that supplies `nab`.
* A `nab`/`nmode` build with **the patch that writes the Hessian**. Refer to
  [`OPTIMIZATION.md` §0](OPTIMIZATION.md). Without the patch, the `-ah` and
  `-ageig` flags give no data.

Set `$AMBERHOME` to your AmberTools installation. Set `$AMBERCLASSICHOME` to
your AmberClassic installation. Then run these commands:

```bash
conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh
```

> Ask your site administrator before you patch AmberTools. A shared
> AmberTools build may already contain the patch.

---

## The workflow

```
  QM transition state (Gaussian freq job)
              |
   1. RESP charges + GAFF typing            antechamber / parmchk2
              |
   2. build the initial TSFF  ----------->  tools/README.md
        tools/build_tsff.py
              |            => MOL.mol2 + MOL.frcmod + reports of the TS terms to fit
              |
   3. seed force constants (optional)  --->  src/qfuerza.py
              |            => Seminario/FUERZA estimates
              |
   4. fit against QM  -------------------->  OPTIMIZATION.md
        loop.py  +  GRAD / SIMP / HYBR
              |            => optimized frcmod
              |
   5. validate: MD with the TSFF
```

This package does its main work in steps 2 and 4. Steps 1, 3 and 5 use
standard AmberTools and the helper scripts in this package.

---

## Quick start

This example uses these four files in one run directory:

* `MOL.mol2`: the re-typed structure at the TS geometry.
* `MOL.frcmod`: the force field to fit, with the Q2MM header (step 2 below).
* `MOL.in`: the tleap script. It saves `calc/prmtop` and `calc/inpcrd`.
* `MOL.log`: the Gaussian frequency job.

**1. Check that the topology builds.** This check finds most failures before
a run:

```bash
mkdir -p calc
tleap -f MOL.in && ls -l calc/prmtop     # the size must not be zero
```

**2. Check the frcmod header.** Q2MM reads parameters only after two comment
lines: `# Q2MM`, and then `# OPT`. Each line must be on its own, before
`MASS`. `build_tsff.py` does not write these lines, so add them yourself.
With a wrong header, the fit reads 0 parameters and gives no error. Refer to
[`OPTIMIZATION.md` §2](OPTIMIZATION.md).

```bash
grep -n -m3 '^#' MOL.frcmod
```

**3. Select the parameters to fit** (`params.txt`). Write one parameter on
each line: `ff_row ff_col lower upper`. `ff_row` is the line number in the
frcmod. `ff_col` is `1` for a force constant and `2` for an equilibrium
value. `HYBR` needs **finite** bounds:

```
49 1 10 1500
104 1 1 250
```

**4. Write `loop.in`:**

```
DIR /absolute/path/to/rundir
FFLD read MOL.frcmod
PARM params.txt
RDAT -gh MOL.log -i 1
FXATM fixedatoms.txt          # optional: exclude QM-frozen atoms
CDAT -ah MOL.in
COMP -o start.txt
LOOP 0.001
HYBR --max_iter 200 --pop_size 24 --tight false --n_processes 24
END
FFLD write frcmod.gaff.01
CDAT
COMP -o opt.txt
```

**5. Start the run from the run directory.** `RDAT` and `CDAT` find their
files relative to the directory where you start `loop.py`:

```bash
cd /absolute/path/to/rundir
python /path/to/q2mm-amber/src/loop.py loop.in
```

To see the improvement, compare `start.txt` with `opt.txt`.
[`OPTIMIZATION.md`](OPTIMIZATION.md) gives the full procedure.

---

## Optimizers

The three optimizers minimize the same objective. The objective is a weighted
sum of squared differences between the QM reference values and the
force-field values. Q2MM normalizes each data type separately:

```
score = sum over types (1 / N_type) * sum over points  w^2 * (reference - calculated)^2
```

| command | method | character |
|---|---|---|
| `GRAD` | gradient least squares (lagrange, newton, lstsq, levenberg, svd) | Local. Fast and precise near a good start. Put it inside `LOOP … END`. |
| `SIMP` | simplex | Local. For a small final polish. |
| `HYBR` | particle swarm + differential evolution | Global and slow. It tolerates bad seeds. Put it inside `LOOP … END`. The `LOOP` convergence value also stops the swarm. |

For a TSFF, the forming and breaking coordinates start from placeholder seeds.
Thus you usually need a `HYBR` pass before `GRAD` can make useful progress.

---

## Repository layout

```
src/          the package
tools/        the TSFF builder, the swarm-history reader, and their documentation
tests/        unit tests (no Amber installation necessary)
```

### `src/`

| module | role |
|---|---|
| `loop.py` | **Entry point.** Reads `loop.in` and sends each command to the module that runs it. |
| `calculate.py` | The `RDAT`/`CDAT` command line. Reads the Gaussian reference data. Builds the `Calculator` from the `CDAT` arguments. |
| `calculators.py` | `Calculator`: the interface between the optimizers and an MM engine. A force field (`FF`) goes in, and `Datum` objects come out. `AmberCalculator` runs tleap, sander, cpptraj and nab for one system. `CalculatorGroup` runs several systems as one calculator. For a parallel `HYBR`, `Calculator.evaluate_many` gives each particle its own directory and runs the particles on a pool of worker processes. |
| `score.py` | The objective function (`compare_data`), the weights for each data type and for each Hessian element, and data trimming. |
| `opt.py` | The optimizer base class, and `SwarmOptimizer` (the adapter for `HYBR`). |
| `hybrid_optimizer.py` | The `PSO_DE` engine (particle swarm + differential evolution). |
| `gradient.py` | The gradient optimizer (`GRAD`). |
| `simplex.py` | The simplex optimizer (`SIMP`). |
| `parameters.py` | Parameter selection and trimming (`PARM`). |
| `data_structs.py` | `Datum`, `Param`, `AmberFF`, `Structure` and related types. Also the builders that make `Datum` objects from a Hessian, a geometry or an energy. |
| `utilities.py` | File input and output: `AmberUtilities` (the frcmod, and the sander, cpptraj and nab files), `Frcmod`, and the readers for Gaussian logs and mol2 files. |
| `math_util.py` | Linear algebra. |
| `qfuerza.py` | Seminario/FUERZA force-constant estimation (standalone CLI) |
| `constants.py` | Weights, step sizes, unit conversions, and the logging configuration. |

### `tools/`

| script | role |
|---|---|
| `build_tsff.py` | Builds the initial TSFF from a RESP mol2 and a mapping file. Writes the re-typed mol2, the frcmod, the LEaP files, and CSV reports of the TS terms to fit. |
| `extract_swarm.py` | Reads the `HYBR` history of a run from `root.log` and from the particle directories. |

### `tests/`

Unit tests for the calculators, the file input and output, the data types,
the math functions, the scoring, the `loop.in` parser, the
optimizer-calculator contract, and `build_tsff.py`. The tests do not need an
Amber installation. A fake Amber (`tests/fake_amber.py`) writes the files
that each Amber program would write.

```bash
python -m unittest discover -s tests -v
```

---

## Seeding force constants — `qfuerza.py`

Estimates bond and angle force constants from a QM Hessian
(Seminario/FUERZA), giving the optimizer a far better starting point than
generic GAFF analogues:

```bash
python src/qfuerza.py -i MOL.frcmod -o MOL_seeded.frcmod \
                      -m MOL.mol2 -gl MOL.log --invert
```

| flag | meaning |
|---|---|
| `-i` / `-o` | input / output force field |
| `-m` | mol2 structure(s) |
| `-gl` / `-gf` / `-ml` | Gaussian `.log` / Gaussian `.fchk` / MacroModel `.log` Hessian |
| `--invert` | invert the Hessian curvature for a transition state |
| `--prep` | also zero bond dipoles/charges and torsion terms `V1–V3` |
| `--raw-fuerza` | plain FUERZA, without the QFUERZA correction |
| `--individualize` | one force field per structure instead of an average |

Adapted in part from [Samuel Genheden's Seminario
implementation](https://github.com/SGenheden/Seminario).

---

## Differences from upstream Q2MM

In addition to the AMBER backend, this fork adds these features:

* **`HYBR`**: the global hybrid PSO/DE optimizer from the upstream
  [`hybrid-opt`](https://github.com/Q2MM/q2mm/tree/hybrid-opt) branch. It
  keeps the same `LOOP` logic. This fork adapts it to the AMBER backend and
  uses `--name value` options. It can evaluate the particles in parallel,
  each particle in its own working directory.
* **`FXATM <file>`**: you give the fixed atoms in a named file in `loop.in`.
  Upstream Q2MM reads a hardcoded `fixedatoms.txt`. Q2MM applies the
  exclusion when it calculates a score. Thus `FXATM` can come before or
  after `CDAT`. It applies to each score that Q2MM calculates after the
  `FXATM` line. It sets the weight of the long-range Hessian couplings of a
  fixed atom to zero. It keeps the bonded 1-2, 1-3 and 1-4 terms. This
  agrees with the upstream weighting.
* **`build_tsff.py`**: a tool that builds a TSFF from a published parameter
  set.
* **`FFLD read` keeps a `.orig` backup** of the initial frcmod. At the start
  of each run, Q2MM restores the frcmod from this backup. The fit rewrites
  the frcmod during a run, but each run starts from the same parameters. If
  you change the initial force field on purpose, delete the `.orig` file.
* **Electrostatics from the QM reference**: sander and nab use the
  Generalized Born (GB) model. Q2MM reads the dielectric constant ε from the
  Gaussian reference logs. It uses the `Eps` of the PCM block, or 1 for a
  gas-phase log. Upstream Q2MM uses a constant dielectric of 80.40 (water) in
  the nab Hessian. Refer to [`OPTIMIZATION.md` §4](OPTIMIZATION.md).

---

## Gotchas

| symptom | cause |
|---|---|
| `Trimmed number of parameters down to 0` | The frcmod header is wrong or missing. `# Q2MM` and `# OPT` must be on **separate** lines, in that order, before `MASS`. Otherwise Q2MM reads no parameters and fits nothing, and it gives no error. `build_tsff.py` does not write this header. |
| The score is `inf`. The log shows `Calculation failed in …` and `the calculation is incomplete. Scoring as inf.` | An Amber step failed, usually `tleap`. Check that `calc/prmtop` exists and that its size is not zero. Read `amber.MOL.log` and `leap.log`. A usual cause is `DU` or SYBYL atom types in the mol2. When the score is `inf`, `COMP` writes no table. |
| `OverflowError: Range exceeds valid bounds` | A `HYBR` parameter has an infinite bound. `HYBR` needs a finite `lower upper` pair on each line. |
| The score changes very little, but the parameters change a lot | Residuals that the selected parameters cannot change control the objective. The usual cause is a forming or breaking contact that is **not bonded** in the mol2. Amber then scores the contact as a nonbonded clash with a very large Hessian element. Correct the topology, not the optimizer. |
| `Hessian file missing: …hes` | AmberTools does not have the patch, or an earlier Amber step failed. |
| `%FLAG ATOMIC_NUMBER` in `calc/prmtop` has −1, or 5 (boron), for custom atom types | tleap does not know the elements of the custom atom types. Put `source MOL_atomtypes.leap` after the two `source leaprc` lines in `MOL.in`. A QM reference in a solvent (ε > 1) needs the correct elements, because GB takes its radii from them. |

`DIR` sets the directory for the `FFLD`, `PARM`, `FXATM` and `COMP -o` files.
The run outputs (`ff_NNN.frcmod`, `swarm_particles/`,
`hybrid_opt_history.bin`) also go to that directory. `RDAT` and `CDAT` do not
use `DIR`. They find their files relative to the directory where you start
`loop.py`, or relative to their `-d` option. Thus start `loop.py` from the
run directory, and set `DIR` to the absolute path of that directory.

---

## License and credit

MIT license (refer to [`LICENSE`](LICENSE)).

This package uses the Q2MM method and codebase from the
[Q2MM project](https://github.com/Q2MM/q2mm) (Norrby, Wiest and co-workers).
Please cite the Q2MM literature for the method.
