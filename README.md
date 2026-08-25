# q2mm-amber

Quantum-guided molecular mechanics (Q2MM) force-field parameterization on the
**AMBER** path — GAFF/GAFF2 `frcmod` force fields, built and evaluated with
`tleap` / `sander` / `nab`, fitted against Gaussian reference data.

Its main use is building **transition-state force fields (TSFFs)**: force
fields that hold a molecule at a QM transition state, so the TS can be carried
into ordinary MM/MD — including inside an enzyme.

---

## Scope

Upstream [Q2MM](https://github.com/Q2MM/q2mm) is built around Schrödinger's
MM3\* (`mm3.fld`) and MacroModel. This package is the AMBER counterpart:

| | upstream Q2MM | q2mm-amber |
|---|---|---|
| force field | MM3\* `mm3.fld` | GAFF/GAFF2 `frcmod` |
| MM engine | MacroModel (`bmin`) | `tleap` + `sander` + `nab`/`nmode` |
| requires Schrödinger | yes | **no** |
| reference data | Jaguar / Gaussian | Gaussian |

The command vocabulary (`DIR`, `FFLD`, `PARM`, `RDAT`, `CDAT`, `COMP`,
`LOOP`/`END`, `GRAD`, `SIMP`) is deliberately the same as upstream, so loop
files and workflows read alike.

---

## Documentation

| document | covers |
|---|---|
| **[`tools/README.md`](tools/README.md)** | **Start here.** Atom typing: turning a GAFF `mol2` into a tleap-ready TSFF (`map_published.py`, `clone_atom_types.py`) |
| **[`OPTIMIZATION.md`](OPTIMIZATION.md)** | Running the fit: parameter files, `loop.in`, the `GRAD` and `SWARM` optimizers in full, the AmberTools patch |

---

## Requirements

* **Python 3** with `numpy` (a conda environment is fine — no Schrödinger)
* **AmberTools** — `tleap`, `sander`, `antechamber`, `parmchk2`, `cpptraj`
* **AmberClassic** (or an AmberTools build) providing `nab`
* the `nab`/`nmode` source **patched to write the Hessian** — see
  [`OPTIMIZATION.md` §0](OPTIMIZATION.md). Without it the `-ah`
  flag produces no data.

Point `$AMBERHOME` at your AmberTools installation and `$AMBERCLASSICHOME` at
your AmberClassic one, then:

```bash
conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh
```

> Check with your site administrator before patching — a shared AmberTools
> build may already include it.

---

## The workflow

```
  QM transition state (Gaussian freq job)
              |
   1. RESP charges + GAFF typing            antechamber / parmchk2
              |
   2. custom TS atom types  ------------->  tools/README.md
        map_published.py, clone_atom_types.py
              |            => MOL_ts.mol2 + MOL_ts.frcmod (seeds)
              |
   3. seed force constants (optional)  --->  src/qfuerza.py
              |            => Seminario/FUERZA estimates
              |
   4. fit against QM  -------------------->  OPTIMIZATION.md
        loop.py  +  GRAD / SIMP / SWARM
              |            => optimized frcmod
              |
   5. validate: MD with the TSFF
```

Steps 2 and 4 are where this package does its work; 1, 3 and 5 use standard
AmberTools plus the helpers here.

---

## Quick start

Assuming you already have `MOL.mol2` (re-typed, TS geometry), `MOL.frcmod`,
`MOL.in` (tleap script) and `MOL.log` (Gaussian frequency job):

**1. Confirm the topology builds** — this catches most failures up front:

```bash
tleap -f MOL.in && ls -l calc/prmtop     # must be non-zero
```

**2. Choose parameters to fit** (`params.txt`) — one per line,
`ff_row ff_col lower upper`, where column 1 is a force constant and column 2 an
equilibrium value. Bounds must be **finite** for `SWARM`:

```
49 1 10 1500
104 1 1 250
```

**3. Write `loop.in`:**

```
DIR /absolute/path/to/rundir
FFLD read MOL.frcmod
PARM params.txt
RDAT -gh MOL.log -i 1
FXATM fixedatoms.txt          # optional: exclude QM-frozen atoms
CDAT -ah MOL.in
COMP -o start.txt
SWARM max_iter=200 pop_size=24 tight=false n_processes=24
FFLD write frcmod.gaff.01
CDAT
COMP -o opt.txt
```

**4. Run:**

```bash
python /path/to/q2mm-amber/src/loop.py loop.in
```

Compare `start.txt` with `opt.txt` for the improvement. Full detail in
[`OPTIMIZATION.md`](OPTIMIZATION.md).

---

## Optimizers

All three minimise the same objective — a per-type-normalised weighted sum of
squared residuals between QM reference and force-field values:

```
score = sum over types (1 / N_type) * sum over points  w^2 * (reference - calculated)^2
```

| command | method | character |
|---|---|---|
| `GRAD` | gradient least-squares (lagrange, newton, lstsq, levenberg, svd) | local, fast, precise near a good start; goes inside `LOOP … END` |
| `SIMP` | simplex | local, small polish |
| `SWARM` | particle swarm + differential evolution | global, slow, tolerates bad seeds; manages its own iterations |

For a TSFF the forming/breaking coordinates start from placeholder seeds, so a
`SWARM` pass is usually needed before `GRAD` can do useful work.

---

## Repository layout

```
src/          the package
tools/        atom-typing helpers + documentation
```

### `src/`

| module | role |
|---|---|
| `loop.py` | **entry point** — reads `loop.in` and dispatches every command |
| `calculate.py` | builds reference (Gaussian) and calculated (Amber) data; Hessian handling, `-i` inversion, per-element weights |
| `score.py` | the objective function (`compare_data`) and data trimming |
| `opt.py` | optimizer base class + `SwarmOptimizer` (the `SWARM` adapter) |
| `hybrid_optimizer.py` | the `PSO_DE` engine (particle swarm + differential evolution) |
| `gradient.py` | gradient optimizer (`GRAD`) |
| `simplex.py` | simplex optimizer (`SIMP`) |
| `parameters.py` | parameter selection / trimming (`PARM`) |
| `data_structs.py` | `Datum`, `Param`, `AmberFF` and related types |
| `utilities.py` | file I/O and the Amber pipeline driver (`AmberLeap`) |
| `calculators.py` | data-extraction helpers |
| `math_util.py` | linear algebra |
| `qfuerza.py` | Seminario/FUERZA force-constant estimation (standalone CLI) |
| `constants.py` | weights, steps, unit conversions, logging config |

### `tools/`

| script | role |
|---|---|
| `map_published.py` | transfer published atom types onto your molecule by graph matching |
| `clone_atom_types.py` | re-type selected atoms and generate the `frcmod` they require |

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

Beyond the AMBER backend, this fork adds:

* **`SWARM`** — the global hybrid PSO/DE optimizer from upstream's
  [`hybrid-opt`](https://github.com/Q2MM/q2mm/tree/hybrid-opt) branch (where it
  is the `HYBR` command), adapted to the AMBER backend with `key=value` options
  and parallel evaluation in one working directory per particle.
* **`FXATM <file>`** — fixed atoms are supplied by a named file in `loop.in`
  rather than a hardcoded `fixedatoms.txt`, and the exclusion is applied at
  **score time**, so it works regardless of where the command sits. It zeroes
  only the long-range Hessian couplings of a fixed atom, keeping the bonded
  1-2 / 1-3 / 1-4 terms — matching upstream's weighting.
* **atom-typing tools** for building a TSFF from a published parameter set.
* **`FFLD read` keeps a `.orig` backup** of the starting `frcmod` and restores
  it at the start of every run, so runs are reproducible from identical
  parameters (the file is rewritten in place during a fit). Delete the `.orig`
  to re-baseline after deliberately editing the force field.

---

## Gotchas

| symptom | cause |
|---|---|
| `Trimmed number of parameters down to 0` | the `frcmod` header is missing its flags. `# Q2MM` and `# OPT` must be on **separate** lines, in that order, before `MASS` — otherwise no parameters are read and nothing is fitted, silently. |
| every score `0.0`, `Total Num. data points: 0` | `tleap` failed — check `calc/prmtop` is non-zero and read `leap.log`. Usually leftover `DU`/SYBYL atom types. |
| `OverflowError: Range exceeds valid bounds` | a `SWARM` parameter has an infinite bound. `SWARM` needs finite `lower upper` on every line. |
| score barely moves while parameters swing wildly | the objective is dominated by residuals the selected parameters cannot affect — classically a forming/breaking contact left **unbonded** in the `mol2`, which Amber scores as a nonbonded clash with an enormous Hessian element. Fix the topology, not the optimizer. |
| `Hessian file missing: …hes` | AmberTools is not patched, or an earlier pipeline step failed. |
| `FileExistsError: swarm_particles/p_000` | leftovers from a previous run; delete `swarm_particles/` first. |

Paths in `DIR` and in `RDAT`/`CDAT -d` should be **absolute** — parallel
workers resolve them from their own directories.

---

## License and credit

MIT (see [`LICENSE`](LICENSE)).

Built on the Q2MM method and codebase from the
[Q2MM project](https://github.com/Q2MM/q2mm) (Norrby, Wiest and co-workers).
Please cite the Q2MM literature for the underlying method.
