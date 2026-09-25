# Running the Q2MM fit — gradient and hybrid optimizers

This picks up where [`tools/README.md`](tools/README.md) stops. That guide ends
with a re-typed `mol2` + `frcmod` that pass `tleap`, and a `.review` file listing
every **seed** parameter. This guide turns those seeds into fitted parameters.

The optimizers all minimise the same objective; they differ only in how they
search:

| command | method | use it for |
|---|---|---|
| `GRAD` | gradient least-squares (local) | refining parameters already close to right |
| `SIMP` | simplex (local) | small polish, few parameters |
| `HYBR` | hybrid particle-swarm + differential evolution (global) | seeds far from right, or a rough start |

---

## Conventions

| placeholder | meaning |
|---|---|
| `$SRC` | `/path/to/q2mm-amber/src` |
| `$WORK` | your run directory (absolute path) |
| `$AMBERHOME` | your AmberTools installation |
| `$AMBERCLASSICHOME` | your AmberClassic installation (provides `nab`) |
| `MOL` | your molecule stem, eg `Thio` |

Set the Amber variables to match your site — the exact paths differ per
installation.

---

## 0. Prerequisite — patch AmberTools to emit the Hessian

**Only needed if you build your own AmberTools.** The `-ah` (Amber Hessian)
flag requires a patched `nab`/`nmode` that writes the Hessian to disk; stock
AmberTools does not. Per the upstream project
(<https://github.com/Q2MM/q2mm/tree/master>), edit:

```
AmberTools/src/sff/nmode.c
```

Find the mass-weighting block:

```c
         /*
          * Mass weight the Hessian:
          */

         j = 1;
         for (i = 1; i <= natom; i++) {
            g[j + 2] = g[j + 1] = g[j] = 1.0 / sqrt(m[i]);
            j += 3;
         }

         k = 0;
         for (i = 1; i <= ncopy; i++) {
            for (j = 1; j <= ncopy; j++) {
               h[k] = g[i] * h[k] * g[j];
               k++;
            }
         }
```

and insert **immediately after it**:

```c
            // q2mm
            FILE * hFile;
            hFile = fopen("./calc/hessian.mat","w");
            fprintf( hFile, "Hessian %d\n", natom);
            k = 0;
            for (i = 1; i <= ncopy; i++){
                for (j = 1; j <= ncopy; j++){
                    fprintf( hFile, "%12.5f ", h[k]);
                    k++;
                }
                fprintf( hFile, "\n");
            }
            fclose(hFile);
            // q2mm
```

Then rebuild AmberTools as usual for your installation. The dump must come
*after* the mass weighting — Q2MM expects a **mass-weighted** Hessian and does
not re-weight the Amber side.

> **Check before you patch.** A shared AmberTools/AmberClassic build may
> already carry this modification, in which case you can skip this section
> entirely — ask whoever maintains your installation.

**Verify** — after any Amber run with `-ah`, this file must exist and be
non-empty:

```bash
ls -l calc/amber.MOL.hes
```

---

## 1. Environment

The AMBER path uses the `q2mm` conda environment plus AmberTools — **not**
Schrödinger.

```bash
conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh
```

---

## 2. Run directory

Everything lives in one directory, addressed by an **absolute path**:

| file | role |
|---|---|
| `MOL.frcmod` | the force field being fitted (from `tools/README.md` step 3) |
| `MOL.mol2` | re-typed structure at the TS geometry |
| `MOL.in` | tleap script — Q2MM re-runs it on every FF evaluation |
| `MOL.log` | the QM reference (Gaussian frequency job) |
| `params.txt` | which parameters to fit (§3) |
| `loop.in` | the command file (§4) |
| `fixedatoms.txt` | *optional* — atoms to exclude from the Hessian fit |

A minimal `MOL.in`:

```
source leaprc.gaff2
source leaprc.protein.ff19SB
loadamberparams MOL.frcmod
mol = loadmol2 MOL.mol2
saveamberparm mol calc/prmtop calc/inpcrd
quit
```

### The `frcmod` header — required

Q2MM only reads parameters from a **flagged region** of the `frcmod`. Two
comment lines are required, in this order, **on separate lines**, before the
`MASS` section:

```
# Q2MM
# OPT
MASS
CT 12.010
...
```

* a line containing **`# Q2MM`** opens the Q2MM region
* a **later** comment line containing **`OPT`** switches parameter collection on

Extra free text is fine on either line — `# Q2MM  estimated input TSFF ...`
works — and a title line may precede them.

> **This fails silently.** Get the header wrong and `read_frcmod` returns **zero
> parameters** with no error; `PARM` then reports `Trimmed number of parameters
> down to 0` and the optimizer runs to completion having fitted nothing. All of
> these yield 0 parameters:
>
> | header | result |
> |---|---|
> | `# Q2MM` then `# OPT` | ✅ parameters read |
> | `# OPT` alone, no `# Q2MM` | ❌ 0 |
> | `# Q2MM OPT` — combined on one line | ❌ 0 |
> | `# Q2MM` with no later `OPT` line | ❌ 0 |
>
> Check with `grep -n -m3 '^#' MOL.frcmod` before running.

`tleap` prints `Unknown keyword (# OPT)` when it reads the file — harmless, it
skips the line.

> **Check tleap before you launch anything.** Run `tleap -f MOL.in` and confirm
> `calc/prmtop` is **non-zero**. A 0-byte `prmtop` means no topology, so no
> Hessian, so every score is `0.0` and the optimizer runs to completion having
> fitted nothing.

---

## 3. The parameter file (`PARM`)

One parameter per line:

```
ff_row  ff_col  [lower upper | neg | pos | both]
```

* `ff_row` — line number in the `frcmod`
* `ff_col` — **1 = force constant, 2 = equilibrium value**
* bounds — a finite `lower upper` pair, or a keyword

```
# bond force constants
49 1 10 1500
50 1 10 1500
# angle force constants
104 1 1 250
105 1 1 250
```

**Everything flagged in `MOL.frcmod.review` must appear here.** Those seeded
force constants are exactly what the fit exists to determine; leave one out and
its placeholder value survives into the final TSFF.

> **`HYBR` requires finite bounds.** It seeds particles with
> `np.random.uniform(lower, upper)`, so `inf` — including the `pos` / `neg` /
> `both` keywords, which expand to infinite ranges — raises
> `OverflowError: Range exceeds valid bounds`. `GRAD` and `SIMP` treat infinite
> bounds as hard walls and are unaffected. Bounds must also bracket the current
> values, since the swarm tethers particles near them.

Generate a starting list with `parameters.py`, then add bounds:

```bash
python $SRC/parameters.py -f MOL.frcmod -pt bf af
```

---

## 4. `loop.in` — the command file

Commands run top to bottom.

| command | meaning |
|---|---|
| `DIR <path>` | working directory — **use an absolute path** |
| `FFLD read <file>` | read the starting `frcmod` |
| `FFLD write <file>` | write the optimised `frcmod` |
| `PARM <file>` | select parameters to fit |
| `RDAT <args>` | gather **reference** (QM) data |
| `CDAT <args>` | gather **calculated** (Amber) data |
| `COMP [-o out] [-p]` | score reference vs calculated |
| `LOOP <conv> … END` | repeat the block until the score changes by < `conv` |
| `GRAD [opts]` | gradient optimiser (§5) |
| `SIMP [max_params=N]` | simplex optimiser |
| `HYBR [opts]` | hybrid optimiser (§6), inside `LOOP … END` |
| `WGHT <typ> <w>` | override a per-type weight |
| `STEP <ptype> <s>` | override a differentiation step |
| `FXATM <file>` | exclude fixed atoms from the Hessian fit |

### Data-type flags

`RDAT` takes the `g*` (Gaussian reference) flags, `CDAT` the matching `a*`
(Amber calculated) ones. **Always pair them.**

| property | reference (`RDAT`) | calculated (`CDAT`) |
|---|---|---|
| Hessian | `-gh MOL.log` | `-ah MOL.in` |
| eigenmatrix (normal modes) | `-geigz MOL.log` | `-ageig MOL.in,MOL.log` |
| bond lengths | `-gabo MOL.log` | `-abo MOL.in` |
| angles | `-gaao MOL.log` | `-aao MOL.in` |
| torsions | `-gato MOL.log` | `-ato MOL.in` |
| energies | `-ge / -ge1 / -geo / -ge1o` | `-ae / -ae1 / -aeo / -ae1o` |

### `-i` — the transition-state flip

```
RDAT -gh MOL.log -i 1
```

A transition state has one **negative** Hessian eigenvalue (the imaginary
frequency along the reaction coordinate); an MM force field can only produce
positive curvature. `-i <value>` replaces that most-negative eigenvalue with
`<value>`, turning the saddle point into a fittable minimum. It affects the
**reference only**. Use it on every Hessian-fitting cycle of a TSFF.

### Hessian fitting vs eigenmode fitting

Both fit the curvature of the force field at the QM transition-state
geometry to the QM curvature; they differ in the basis.

* **Hessian fitting** (`-gh` / `-ah`) compares the mass-weighted Cartesian
  Hessians element by element, every element weighted by how many bonds
  apart its two atoms are (§7).
* **Eigenmode fitting** (`-geigz` / `-ageig`) is the original Q2MM
  objective, ported from upstream's hybrid-opt branch. The reference is the
  diagonal matrix of the QM eigenvalues, one per normal mode, in
  kJ/mol/Å²·amu. The calculated side is the force-field Hessian projected
  onto the QM eigenvectors, `V·H·Vᵀ`: its diagonal is the force field's
  curvature along each QM mode, its off-diagonals the coupling between modes,
  which the reference has as zero. Weights: `eig_d_low` / `eig_d_high` for
  diagonal elements (below / above 1100), `eig_o` for off-diagonals, and
  `eig_i` for the transition-state mode *while its eigenvalue is still
  negative*. With `-i` that element becomes positive and is weighted like any
  other diagonal. `eig_i` defaults to 0, so without `-i` the imaginary mode is
  simply left out of the fit.

`-ageig` takes the leap input and the log together, comma-separated with no
space, and can sit on the same line as the Hessian flags:

```
RDAT -gh MOL.log -geigz MOL.log -i 1
CDAT -ah MOL.in -ageig MOL.in,MOL.log
```

Two requirements matter especially here:

* run the Gaussian frequency job with **`nosymm`** and **`freq=hpmodes`**.
  The eigenvectors are read from the printed normal coordinates: without
  `hpmodes` they carry two decimals and are orthonormal only to a few percent,
  and without `nosymm` they can be in a rotated frame.
* the log and the mol2 must hold the same atoms, in the same order and the
  same Cartesian frame (true for `-gh` as well). q2mm-amber compares the two
  geometries and warns when they differ by more than 0.01 Å after centering.

### `FXATM` — fixed atoms

```
FXATM <file>
```

Excludes atoms that were **frozen in the reference QM calculation** — typically
the truncated-cluster boundary atoms held at their crystal/enzyme positions.
Their Hessian curvature is an artefact of the geometric constraint rather than
real molecular stiffness, so fitting force constants to it biases the result.

The file lists **one 1-based atom index per line** (mol2 numbering):

```
# fixedatoms.txt
18
20
32
35
38
41
43
```

Usage in context — the whole point is that it applies to the Hessian, so pair
it with an `-ah` / `-gh` cycle:

```
DIR ./
FFLD read Thio.frcmod
PARM BandA_FC.txt
RDAT -gh Thio.log -i 4500
FXATM fixedatoms.txt          # exclude these atoms from the Hessian fit
CDAT -ah Thio.in
COMP -o ./bafc_start.txt
LOOP 0.001
HYBR --max_iter 200 --pop_size 24 --tight false --n_processes 24
END
FFLD write ./frcmod.gaff.01
CDAT
COMP -o ./bafc_opt.01.txt
```

What it does to the weights:

| Hessian element | weight |
|---|---|
| long-range coupling involving a fixed atom | **0** — dropped from the fit |
| bonded (1-2 / 1-3 / 1-4) term of a fixed atom | unchanged — still fitted |
| everything else | unchanged |

Keeping the bonded terms matches upstream Q2MM's weighting: a constrained
atom's local bond/angle curvature is still useful signal, while its long-range
cross-terms are where the constraint contamination sits.

Two practical notes:

* **Placement does not matter.** The exclusion is applied at *score* time, so
  `FXATM` works wherever it sits in the file — before or after `CDAT`. (In an
  earlier version it was applied when the data was built, which made the
  weights differ between a `COMP` before and after the command.)
* **It is opt-in.** Omit the line and nothing is excluded; a stray
  `fixedatoms.txt` sitting in the directory does nothing on its own, unlike
  upstream Q2MM which auto-reads any file with that name.

> `FXATM` is the right tool for genuinely QM-frozen atoms. It is *not* the fix
> for a reaction-centre contact that blows up the Amber Hessian — that is a
> missing bond in the `mol2`, and excluding the atoms hides the problem rather
> than solving it.

---

## 5. Gradient optimiser (`GRAD`)

A local least-squares fit: it differentiates each selected parameter, builds a
Jacobian, and solves for the step that minimises the objective. Fast and
precise near a good starting point; it will happily sit in a local minimum if
your seeds are poor.

**`GRAD` goes inside a `LOOP … END` block** — the block is what iterates it to
convergence.

```
LOOP 0.01
GRAD
END
```

### Options

`method=True|False` plus optional per-method settings, comma-separated, with
value lists in brackets separated by `/`:

```
GRAD lstsq=False newton=True,cutoffs=[None],radii=[0.01/0.1/2.0] svd=True,factor=[0.01/0.1]
```

| method | default | notes |
|---|---|---|
| `lagrange` | **on** | Lagrange-multiplier damping |
| `newton` | **on** | Newton-Raphson |
| `lstsq` | off | plain least squares |
| `levenberg` | off | Levenberg-Marquardt |
| `svd` | off | singular-value decomposition |

Each accepts `radii=[…]`, `cutoffs=[…]`, `factor=[…]`. Every enabled method
proposes a trial force field; the best-scoring one wins the cycle.

### Complete example

```
DIR $WORK
FFLD read MOL.frcmod
PARM params.txt
RDAT -gh MOL.log -i 1
FXATM fixedatoms.txt
CDAT -ah MOL.in
COMP -o start.txt
LOOP 0.01
GRAD
END
FFLD write frcmod.gaff.01
CDAT
COMP -o opt.txt
```

---

## 6. Hybrid optimiser (`HYBR`)

A population of trial force fields ("particles") explores the parameter space
by particle-swarm dynamics interleaved with differential-evolution steps. It is
a **global** search: far slower than `GRAD` but able to escape local minima and
tolerate seeds that are badly wrong — which is the normal situation for
forming/breaking TS coordinates.

It is the AMBER-path port of the `HYBR` command on upstream's
[`hybrid-opt`](https://github.com/Q2MM/q2mm/tree/hybrid-opt) branch and keeps
its logic. Source: `src/opt.py::SwarmOptimizer` →
`src/hybrid_optimizer.py::PSO_DE`; every particle is evaluated by
`src/calculators.py::AmberCalculator`.

### Synopsis

```
LOOP <conv>
HYBR [--max_iter N] [--pop_size N] [--tight T|F] [--n_processes N]
END
```

* Options are `--name value` pairs, space-separated, **order-independent**.
  An unknown option is an error.
* A bare `HYBR` is valid and uses all defaults.
* **`HYBR` goes inside a `LOOP … END` block**, like `GRAD`. The block's
  convergence value drives the swarm (next section).

### Options

| key | type | default | meaning |
|---|---|---|---|
| `--max_iter` | int | **200** | PSO/DE iterations per `LOOP` cycle (hard cap) |
| `--pop_size` | int | **24** | number of particles — **must be even** |
| `--tight` | T/F | **true** | tight (refine) vs global (explore) starting spread |
| `--n_processes` | int | **1** | parallel Amber evaluations; `1` = serial |

The options are read when the swarm is created, on the first `HYBR` of a
`LOOP` block; the block's later cycles reuse them.

### How the `LOOP` drives it

The `LOOP <conv>` value does two jobs, as in upstream Q2MM:

1. **Between cycles** it is the usual loop test: after each `HYBR` cycle the
   block repeats until the score changes by less than `conv` (relative), or
   stops changing.
2. **Within a cycle** it is the swarm's own early-stop tolerance: the cycle
   ends before `max_iter` once *every* particle has stayed within `conv` of
   the best particle, in *every* parameter, for **20 consecutive
   iterations**. That distance is absolute, in each parameter's own units, so
   for force constants spanning 1–1500 a `conv` of `0.001` essentially never
   fires and `max_iter` governs the cycle length. A larger value (eg `1`)
   lets the early stop trigger.

The swarm **persists across the cycles of one `LOOP` block**: the second
`HYBR` continues the same particles from where the first stopped, around the
best force field found so far, with the PSO/DE hyperparameters frozen at
their final values (the first cycle tapers them from the exploring to the
exploiting end). A new `LOOP` block starts a fresh swarm around the current
force field. A cycle never returns a worse force field than it was given, so
an unimproved cycle ends the block.

### How a particle is scored

Each **particle is one complete parameter vector**. To score it the optimiser:

1. writes the candidate parameters into the `frcmod`,
2. rebuilds the Amber topology (`tleap` via the `.in` file) and runs the
   calculation,
3. compares calculated vs reference data with `score.compare_data`.

The swarm **minimises** that score (§7). A particle whose evaluation throws is
given score `inf` and discarded. `HYBR` uses the **same objective** as
`GRAD`/`SIMP` — only the search differs.

### Option semantics in depth

**`--pop_size`** — the search dimensionality is the number of selected
parameters. Larger populations cover the space better at linearly more cost per
iteration. Rule of thumb: several × the number of parameters (≈6×), and a
global search wants more than a tight one. **Must be even** (asserted, for the
DE step).

**`--max_iter`** — the dominant cost knob:

```
wall-time per cycle ≈ pop_size × max_iter × (Amber eval time) / n_processes
```

(DE steps taper off over the run, so the real cost is a little lower). It is an
upper bound per cycle — a cycle can stop earlier through the `LOOP`
convergence, and the block runs as many cycles as that convergence needs.

**`--tight`** — selects only the **starting spread**, not the core PSO/DE
hyperparameters (a single `DEFAULT_CONFIG`: inertia 0.9→0.4, cognitive
2.5→0.5, social 0.5→2.5, `DE/best/1`, differential weight 0.4→0.1, taper-GA
on). It changes:

| | `--tight true` (refine) | `--tight false` (global) |
|---|---|---|
| particles tethered near current values | 70 % | 30 % |
| initial spread of `af`/`bf` parameters | 0.125 | 1.0 |

Per-type initial deviations: `af`/`bf` → 0.125 or 1.0; `ae` → 15; `be` → 0.5;
`df` → 5; everything else → 1.0.

Accepted true values are `t`, `true`, `1`, `yes`, **case-insensitive**;
anything else is false. So `--tight TRUE` and `--tight T` both mean tight,
while `--tight G`, `--tight false` — and also `--tight tight`, which is not in
the list — all mean global.

**`--n_processes`** — speed only, never the result:

* `1` (default) — **serial**, one Amber evaluation at a time, no extra dirs.
* `>1` — **parallel**: creates `swarm_particles/p_000 … p_{pop_size-1}`, copies
  the `.in` and every input file it names (`frcmod`, `mol2`, …) into each so
  concurrent `tleap`/Amber jobs cannot collide, and runs `n_processes` workers.

Set it ≤ your reserved cores. It is silently capped at `pop_size` (more workers
than particles is useless) and at the available cores.

### Parameters and bounds

Bounds come from the parameter file (cols 3-4) via `p.allowed_range`. All
parameter types are accepted; each gets bounds from its file entry plus the
type-based initial deviation above. The swarm always **biases toward the
current FF values** (they seed the tethered particles), and out-of-bounds
particles are handled by a **reflective** boundary. Finite bounds are
mandatory — see §3.

### Complete example

```
DIR $WORK
FFLD read MOL.frcmod
PARM params.txt
RDAT -gh MOL.log -i 1
FXATM fixedatoms.txt
CDAT -ah MOL.in
COMP -o start.txt
LOOP 0.001
HYBR --max_iter 200 --pop_size 24 --tight false --n_processes 24
END
FFLD write frcmod.gaff.01
CDAT
COMP -o opt.txt
```

### Suggested settings

| situation | command |
|---|---|
| smoke test (does it run at all?) | `HYBR --max_iter 2 --pop_size 4 --n_processes 4` |
| refine an already-good FF | `HYBR --max_iter 300 --pop_size 24 --tight true --n_processes 8` |
| global search from seeds | `HYBR --max_iter 1000 --pop_size 48 --tight false --n_processes 24` |

`HYBR` is **stochastic** — there is no fixed random seed, so two runs differ.
Give it enough `pop_size`/`max_iter` for a stable result, and consider
repeating a production run.

---

## 7. The scoring function

Both optimisers minimise the same objective (`score.compare_data`):

```
score = Σ over types  (1 / N_type) · Σ over points  w² · (reference − calculated)²
```

* the weight is **squared**, so weight 0 removes a point entirely
* each type is divided by its own point count, so a 30 000-element Hessian
  cannot swamp a handful of bond lengths by sheer count
* per-type weights come from `constants.WEIGHTS` (`b` 100, `a` 2, `t` 1,
  `h` 0.031 …) and can be overridden with `WGHT`
* Hessian elements get an additional per-element weight by topology:
  diagonal `0`, 1-2 and 1-3 `0.031`, 1-4 `0.31`, longer range `0.031`
* eigenmatrix elements use `eig_d_low` / `eig_d_high` (`0.1`) on the
  diagonal, `eig_o` (`0.05`) off it, and `eig_i` (`0`) for an un-inverted
  transition-state mode (§4)

The printed `Score` column is the per-element contribution **after** the
`1/N_type` division, shown to 4 decimals. Most Hessian rows therefore print
`0.0000` although their true values are small-but-nonzero and *are* summed. A
row is genuinely excluded only when its **Weight** column reads `0.00`.

---

## 8. Running

```bash
cd $WORK
python $SRC/loop.py loop.in
```

On the cluster, wrap that in a submit script and request the cores you gave to
`n_processes`:

```bash
#!/bin/bash
#$ -N q2mm_cycle1
#$ -pe smp 24
#$ -q long

conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh

python $SRC/loop.py loop.in
```

> **Clean before re-running in the same directory.** Delete `swarm_particles/`
> and `calc/` from the previous attempt; stale worker directories cause
> `FileExistsError`. Note also that `FFLD read` keeps a one-time
> `MOL.frcmod.orig` backup and restores from it at the start of every run, so
> each run starts from identical parameters. Delete the `.orig` only when you
> have deliberately edited the starting FF.

---

## 9. Outputs

| file | contents |
|---|---|
| `frcmod.gaff.01` | the optimised force field |
| `start.txt` / `opt.txt` | full data comparison before / after |
| `ff_001.frcmod`, `ff_002.frcmod`, … | the best force field after each `LOOP` cycle; numbering continues from files already present |
| `root.log` | complete run log, including per-iteration best scores |
| `calc/` | Amber scratch: `prmtop`, `inpcrd`, `.ene`, `.lst` (topology listing; sets the Hessian weights), `.geo`, `.hes` |
| `swarm_particles/` | per-particle scratch (parallel `HYBR` only) |

Compare `start.txt` with `opt.txt` — the `Total score` lines give the
improvement, and the per-type breakdown underneath shows where it came from.
The log carries `INIT FF SCORE` and, for `HYBR`, a per-iteration
`Iter: k, Best fit: … at [params]` trace.

---

## 10. Staged cycles

Fitting every parameter at once from rough seeds is ill-conditioned. The
established approach is to fit subsets in order of how well determined they
are, feeding each cycle's `frcmod` into the next and tightening convergence as
you go:

| cycle | parameters | reference data |
|---|---|---|
| 1 | bond + angle **force constants** | Hessian |
| 2 | bond + angle **equilibria** | geometry |
| 3 | both, together | geometry + Hessian |
| 4 | **dihedral** barriers | torsions + Hessian |
| 5 | angles + dihedrals | angles + torsions + Hessian |
| 6 | everything | all of the above |

Force constants come first because the Hessian determines them directly;
torsions come last because they are soft and only weakly coupled to the stiff
terms.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `OverflowError: Range exceeds valid bounds` | a `HYBR` parameter has an infinite bound. Put finite `lower upper` on every line of `params.txt`. |
| `AssertionError` on `size_pop` | `pop_size` must be an **even** integer. |
| `Trimmed number of parameters down to 0` | the `frcmod` header is wrong — `# Q2MM` and `# OPT` must be on **separate** lines, in that order, before `MASS` (§2). |
| `Total Num. data points: 0`, every score `0.0` | no data was produced — almost always `tleap` failing. Check `calc/prmtop` is non-zero and read `leap.log`. |
| `could not find vdW parameters for type (Du)` | the `mol2` still carries `DU`/SYBYL types. Re-run the typing workflow in [`tools/README.md`](tools/README.md). |
| `Hessian file missing: …hes` | `nab`/`nmode` produced no `hessian.mat`. Either the earlier steps failed (check `prmtop`), or AmberTools is not patched (§0). |
| `FileExistsError: … swarm_particles/p_000` | leftovers from a previous run. Delete `swarm_particles/` first. |
| score barely moves while parameters change a lot | the objective is dominated by residuals your selected parameters cannot affect — typically a forming/breaking contact left **unbonded** in the `mol2`, which Amber then scores as a nonbonded clash with a huge Hessian. Fix the topology, not the optimiser. |
| `HYBR` used fewer workers than requested | `n_processes` is silently capped at `pop_size` and at the available cores. Speed only; results are unaffected. |
| two runs give different answers | expected — `HYBR` is stochastic with no fixed seed. |
| `HYBR: unrecognised option(s)` | options are `--name value` pairs (`--max_iter 200`); the old `name=value` form and `precision` are gone — the `LOOP` convergence is the precision now. |
