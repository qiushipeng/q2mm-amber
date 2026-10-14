# Running the Q2MM fit: gradient and hybrid optimizers

This guide continues from [`tools/README.md`](tools/README.md). That guide
ends with these files from `build_tsff.py`:

* a re-typed mol2 and a frcmod that pass `tleap`
* `MOL_ts_fit_report.csv`, which lists the TS terms to select for the fit
* `MOL_ts_seed_report.csv`, which lists the seed terms that the builder
  generated. The frcmod marks these terms `GENERATED_TS_SEED`.

This guide fits those parameters to the QM data.

The three optimizers minimize the same objective. Only their search methods
are different:

| command | method | use it for |
|---|---|---|
| `GRAD` | gradient least squares (local) | parameters that are already almost correct |
| `SIMP` | simplex (local) | a small final polish of a few parameters |
| `HYBR` | hybrid particle swarm + differential evolution (global) | seeds that are far from correct, or a rough start |

---

## Conventions

| placeholder | meaning |
|---|---|
| `$SRC` | `/path/to/q2mm-amber/src` |
| `$WORK` | your run directory (absolute path) |
| `$AMBERHOME` | your AmberTools installation |
| `$AMBERCLASSICHOME` | your AmberClassic installation (it supplies `nab`) |
| `MOL` | the stem of your molecule files, for example `Thio` |

Set the Amber variables for your site. The paths are different on each
installation.

---

## 0. Prerequisite: patch AmberTools to write the Hessian

**Do this step only if you build AmberTools yourself.** The `-ah` (Amber
Hessian) and `-ageig` flags need a `nab`/`nmode` build that writes the
Hessian to a file. Standard AmberTools does not write it. The upstream
project (<https://github.com/Q2MM/q2mm/tree/master>) specifies this change.
Edit this file:

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

Put this code **immediately after** that block:

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

Then build AmberTools again, as usual for your installation. The new code
must write the Hessian *after* the mass weighting. Q2MM expects a
**mass-weighted** Hessian. It does not apply mass weights to the Amber
Hessian again.

> **Check before you patch.** A shared AmberTools or AmberClassic build may
> already have this change. In that case, skip this section. Ask the person
> who maintains your installation.

**Check the result.** After an Amber run with `-ah`, this file must exist,
and its size must not be zero:

```bash
ls -l calc/amber.MOL.hes
```

---

## 1. Environment

The AMBER path uses the `q2mm` conda environment and AmberTools. It does
**not** use Schrödinger.

```bash
conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh
```

---

## 2. Run directory

Put all files in one directory. Set `DIR` to the **absolute path** of this
directory, and start `loop.py` from it (§8).

| file | role |
|---|---|
| `MOL.frcmod` | the force field to fit (from `build_tsff.py`, with the Q2MM header below) |
| `MOL.mol2` | the re-typed structure at the TS geometry |
| `MOL.in` | the tleap script. Q2MM runs it again for each force-field evaluation. |
| `MOL.log` | the QM reference (Gaussian frequency job) |
| `params.txt` | the parameters to fit (§3) |
| `loop.in` | the command file (§4) |
| `fixedatoms.txt` | *optional*: the atoms to exclude from the Hessian fit |

A minimal `MOL.in`:

```
source leaprc.gaff2
source leaprc.protein.ff19SB
loadamberparams MOL.frcmod
mol = loadmol2 MOL.mol2
saveamberparm mol calc/prmtop calc/inpcrd
quit
```

Obey these rules for `MOL.in`:

* Save the topology to `calc/prmtop` and `calc/inpcrd`. Q2MM reads these two
  files.
* Give the frcmod and the mol2 as relative file names in the run directory.
  A parallel `HYBR` copies only the files with relative names into each
  particle directory. With an absolute frcmod path, the particles do not
  read their own trial force fields.
* Do not use `MOL.leap.in` from `build_tsff.py` without changes. It uses
  absolute paths, and it saves the topology to `MOL.prmtop`.
* If `build_tsff.py` wrote `MOL.leap.mol2`, the TS has a hydrogen with two
  bonds. Copy the `loadmol2` line from `MOL.leap.in`, and the `set … pert`
  and `bond` lines after it. Use relative file names. Without these lines,
  the topology does not contain the forming bond.
* Optional: put `source MOL_atomtypes.leap` after the two `source leaprc`
  lines. Without this line, the custom atom types get no element or a wrong
  element in the prmtop. The fit does not use the elements. GB, QM/MM and
  element-based tools use them.

### The `frcmod` header: required

Q2MM reads parameters only from a **marked region** of the frcmod. Two
comment lines are necessary. They must be in this order, **on separate
lines**, before the `MASS` section:

```
# Q2MM
# OPT
MASS
CT 12.010
...
```

* A line that contains **`# Q2MM`** starts the Q2MM region.
* A **later** comment line that contains **`OPT`** starts the parameter
  collection.

Each line can also contain free text. For example,
`# Q2MM  estimated input TSFF ...` is correct. A title line can come before
the two lines.

**`build_tsff.py` does not write this header.** Add the two lines after its
title line and before `MASS`.

> **Q2MM gives no error for a wrong header.** With a wrong header,
> `read_frcmod` returns **zero parameters**. `PARM` then reports
> `Trimmed number of parameters down to 0`. The optimizer runs to the end,
> but it fits nothing. Each of these headers gives 0 parameters:
>
> | header | result |
> |---|---|
> | `# Q2MM` then `# OPT` | ✅ parameters read |
> | `# OPT` alone, no `# Q2MM` | ❌ 0 |
> | `# Q2MM OPT` on one line | ❌ 0 |
> | `# Q2MM` with no later `OPT` line | ❌ 0 |
>
> Before you start a run, check the header with
> `grep -n -m3 '^#' MOL.frcmod`.

`tleap` prints `Unknown keyword (# OPT)` when it reads the file. This message
is harmless. `tleap` ignores the line.

> **Check tleap before you start a run.** Make the `calc/` directory
> (`mkdir -p calc`). Run `tleap -f MOL.in`. Then check that the size of
> `calc/prmtop` is **not zero**. If tleap fails, Q2MM gets no data. Each
> score is then `inf`, and the log shows `Calculation failed` and
> `the calculation is incomplete`.

---

## 3. The parameter file (`PARM`)

Write one parameter on each line:

```
ff_row  ff_col  [lower upper | neg | pos | both]
```

* `ff_row`: the line number in the frcmod
* `ff_col`: **1 = force constant, 2 = equilibrium value**
* bounds: a finite `lower upper` pair, or one keyword. `neg` is −∞ to 0,
  `pos` is 0 to ∞, and `both` is −∞ to ∞.

A line without bounds gets the default range of its parameter type: 0 to ∞,
or −∞ to ∞ for `df` and `q`.

```
# bond force constants
49 1 10 1500
50 1 10 1500
# angle force constants
104 1 1 250
105 1 1 250
```

**Select each term in `MOL_ts_fit_report.csv` and in
`MOL_ts_seed_report.csv`.** The fit exists to find these values. If you omit
a term, its seed value stays in the final TSFF.

> **`HYBR` needs finite bounds.** `HYBR` makes the initial particles with
> `np.random.uniform(lower, upper)`. An infinite bound gives
> `OverflowError: Range exceeds valid bounds`. The `pos`, `neg` and `both`
> keywords and the default ranges all have an infinite bound. Infinite bounds
> do not cause a problem for `GRAD` and `SIMP`. The bounds must also contain
> the current values, because the swarm keeps some particles near them.

To make a first list, run `parameters.py` with `-pp`. Then replace each
infinite bound:

```bash
python $SRC/parameters.py -f MOL.frcmod -pt bf af -pp
```

`-pp` prints one line for each selected parameter: `ff_row ff_col lower
upper`. Without `-pp`, the command prints no list.

---

## 4. `loop.in`: the command file

Q2MM runs the commands in order, from top to bottom.

| command | meaning |
|---|---|
| `DIR <path>` | The run directory. **Use an absolute path.** `FFLD`, `PARM`, `FXATM` and `COMP -o` find their files in it. The run outputs also go to it. |
| `FFLD read <file>` | Read the initial frcmod. |
| `FFLD write <file>` | Write the optimized frcmod. |
| `PARM <file>` | Select the parameters to fit. |
| `RDAT <args>` | Read the **reference** (QM) data. |
| `CDAT <args>` | Calculate the **force-field** (Amber) data. |
| `COMP [-o out] [-p]` | Compare the reference data with the calculated data, and score them. |
| `LOOP <conv> … END` | Run the block again until the relative score change is less than `conv`. |
| `GRAD [opts]` | Gradient optimizer (§5). |
| `SIMP [max_params=N]` | Simplex optimizer. |
| `HYBR [opts]` | Hybrid optimizer (§6), inside `LOOP … END`. |
| `WGHT <typ> <w>` | Change the weight of one data type. |
| `STEP <ptype> <s>` | Change one differentiation step. |
| `FXATM <file>` | Exclude fixed atoms from the Hessian fit. |

`RDAT` and `CDAT` do not use `DIR`. They find their files relative to the
directory where you start `loop.py`, or relative to their `-d` option. Q2MM
ignores the text after `#` on a `loop.in` line.

### Data-type flags

`RDAT` takes the `g*` (Gaussian reference) flags. `CDAT` takes the related
`a*` (Amber calculated) flags. **Always use them in pairs.**

| property | reference (`RDAT`) | calculated (`CDAT`) |
|---|---|---|
| Hessian | `-gh MOL.log` | `-ah MOL.in` |
| eigenmatrix (normal modes) | `-geigz MOL.log` | `-ageig MOL.in,MOL.log` |
| bond lengths | `-gabo MOL.log` | `-abo MOL.in` |
| angles | `-gaao MOL.log` | `-aao MOL.in` |
| torsions | `-gato MOL.log` | `-ato MOL.in` |
| energies | `-ge / -ge1 / -geo / -ge1o` | `-ae / -ae1 / -aeo / -ae1o` |

### `-i`: the transition-state flip

```
RDAT -gh MOL.log -i 1
```

A transition state has one **negative** Hessian eigenvalue. This is the
imaginary frequency along the reaction coordinate. An MM force field can give
only positive curvature. `-i <value>` replaces the most negative eigenvalue
with `<value>`. This changes the saddle point into a minimum that the force
field can fit. Use `-i` in each Hessian-fitting cycle of a TSFF.

On the `RDAT` line, `-i` changes **only the reference**. On a `CDAT` line,
`-i` also changes the force-field Hessian.

### Hessian fitting and eigenmode fitting

Both methods fit the force-field curvature at the QM TS geometry to the QM
curvature. They use different bases.

* **Hessian fitting** (`-gh` / `-ah`) compares the two mass-weighted
  Cartesian Hessians element by element. The weight of each element depends
  on the number of bonds between its two atoms (§7).
* **Eigenmode fitting** (`-geigz` / `-ageig`) is the original Q2MM
  objective. This port comes from the upstream hybrid-opt branch.
  * The reference is the diagonal matrix of the QM eigenvalues, one for each
    normal mode, in kJ/mol/Å²·amu.
  * The calculated side is the force-field Hessian projected onto the QM
    eigenvectors, `V·H·Vᵀ`. Its diagonal is the force-field curvature along
    each QM mode. Its off-diagonal elements are the couplings between modes.
    In the reference, these couplings are zero.

The eigenmatrix weights are:

* `eig_d_low` and `eig_d_high`: diagonal elements below and above 1100
* `eig_o`: off-diagonal elements
* `eig_i`: the TS mode, *while its eigenvalue is still negative*. With `-i`,
  this element becomes positive and gets the weight of a usual diagonal
  element. `eig_i` is 0 by default. Thus, without `-i`, the fit ignores the
  imaginary mode.

`-ageig` takes the leap input and the log together. Separate them with a
comma and no space. `-ageig` can be on the same line as the Hessian flags:

```
RDAT -gh MOL.log -geigz MOL.log -i 1
CDAT -ah MOL.in -ageig MOL.in,MOL.log
```

Two requirements are especially important here:

* With the default `--modes printed` (below), run the Gaussian frequency job
  with **`nosymm`** and **`freq=hpmodes`**. Q2MM then reads the eigenvectors
  from the printed normal coordinates.
  * Without `hpmodes`, the eigenvectors have two decimals. They are then
    orthonormal only to a few percent.
  * Without `nosymm`, Gaussian prints them in its rotated *standard
    orientation*. The archive geometry and the archive Hessian (which `-gh`
    reads) stay in the input orientation.
* The log and the structure must have the same atoms, in the same order and
  in the same Cartesian frame. The structure is the mol2 that the leap input
  loads **and** the `MOL.pdb` at which nab calculates the Hessian. This rule
  also applies to `-gh` / `-ah`. Q2MM makes `MOL.pdb` from the mol2 only when
  `MOL.pdb` does not exist. Thus delete `MOL.pdb` each time you replace the
  mol2.

### `--modes`: the source of the normal modes

| `--modes` | source of the `-geigz` eigenvalues and the `-ageig` modes | precision | frame that the mol2 must use |
|---|---|---|---|
| `printed` (default) | The frequency section of the log (`Frequencies --`, force constants, normal coordinates), as in upstream Q2MM. | 5 decimals with `hpmodes`, 2 without | The frame of the last orientation table in the log: the input orientation with `nosymm`, the standard orientation without it. |
| `archive` | The archive Hessian (the Hessian that `-gh` reads). Q2MM projects the Hessian onto the space without rigid translation and rotation, then diagonalizes it. Gaussian does the same for the modes that it prints. | full | The archive geometry (input orientation). This is the same frame as for `-gh`. |

Use the same value on the `RDAT` and `CDAT` lines. `-ageig` stops if its
value is different from the value that `-geigz` used for that log:

```
RDAT -geigz MOL.log -i 1 --modes archive
CDAT -ageig MOL.in,MOL.log --modes archive
```

`archive` works with each frequency log that has an archive entry. This
includes jobs that ran without `nosymm` or `hpmodes`. `archive` also keeps
`-gh` and `-ageig` in one frame, so they can use the same mol2. For a correct
log, the two sources agree to rounding. On a 97-atom ωB97X-D/def2-SVP
transition state, the archive modes give the printed frequencies to
0.4 cm⁻¹. The TS mode is 637.1i with both sources.

### What `-ah` checks

* **Frame: the run stops.** `CDAT` compares two structures with the archive
  geometry of the `-gh` log: the mol2, and the pdb that nab reads.
  * Q2MM pairs the Hessian data by position. Thus the k-th `-ah` input goes
    with the k-th log on the `RDAT -gh` line. The names of the files have no
    effect.
  * If the difference is more than 0.01 Å after centering, Q2MM would
    compare each Hessian element with the wrong motion. `CDAT` then stops
    with an error that names the file.
  * To change the error into a warning, add `--allow-frame-mismatch` to the
    `CDAT` line.

### What `-ageig` checks and reports

* **Frame: the run stops.** Q2MM compares the mol2 and the pdb that nab reads
  with the geometry that sets the frame of the modes. For `printed`, this is
  the last orientation table in the log. For `archive`, it is the archive
  geometry. If the difference is more than 0.01 Å after centering, Q2MM
  would test each mode with the wrong motion. `CDAT` then stops with an
  error that names the probable cause and the solutions.
  `--allow-frame-mismatch` changes the error into a warning.
* **`nosymm`: warning** (`printed`). The printed frame and the archive frame
  are different. This means that the job ran without `nosymm`. If the mol2
  agrees with the printed standard orientation, the eigenmode fit is
  correct. But `-gh` cannot use that mol2, because `-gh` needs the input
  orientation.
* **`hpmodes`: warning** (`printed`). The modes are orthonormal only to more
  than 1%. This is typical of two-decimal modes from a job without
  `freq=hpmodes`.
* **Self-check: information line.** With `printed`, Q2MM rotates the archive
  QM Hessian into the frame of the modes and projects it onto them. The line
  gives three values:
  * the largest frequency error
  * the RMS coupling
  * the score of a force field that gives the QM Hessian *exactly*. This is
    the lowest score that the modes permit (0 for exact modes).

  With `archive`, the line compares the frequencies of the modes with the
  printed frequencies.

For the 97-atom log above, from a job without `nosymm` and `hpmodes`:

| setup | result |
|---|---|
| `printed`, mol2 in the archive frame | Stops: "differ by up to 14.795 A … The frequency job ran without nosymm …" |
| `printed`, mol2 in the standard orientation | Runs, with the `hpmodes` (3.2%) and `nosymm` warnings. Self-check floor: 0.082. |
| `archive`, mol2 in the archive frame | Runs with no warnings. The initial score is 6.13, against 6.24 for the row above. |

### `FXATM`: fixed atoms

```
FXATM <file>
```

`FXATM` excludes atoms that were **frozen in the reference QM calculation**.
Usually these are the boundary atoms of a truncated cluster, held at their
crystal or enzyme positions. The geometric constraint causes their Hessian
curvature, not real molecular stiffness. If you fit force constants to this
curvature, the result has a bias.

The file lists **one 1-based atom index on each line** (mol2 numbering).
Q2MM reads only the lines that contain exactly one number. An example
`fixedatoms.txt`:

```
18
20
32
35
38
41
43
```

This example uses `FXATM` with an `-ah` / `-gh` cycle, because `FXATM`
applies to the Hessian:

```
DIR $WORK
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

The effect on the weights:

| Hessian element | weight |
|---|---|
| long-range coupling (more than 3 bonds apart) that involves a fixed atom | **0**: the fit ignores it |
| bonded term (1-2, 1-3, 1-4) of a fixed atom | no change: the fit includes it |
| all other elements | no change |

Upstream Q2MM uses the same weighting. The local bond and angle curvature of
a constrained atom is still useful information. The constraint contaminates
mainly the long-range cross-terms of the atom.

Two practical notes:

* **Position in `loop.in`.** Q2MM applies the exclusion when it calculates a
  score. Thus `FXATM` can come before or after `CDAT`. It applies to each
  score that Q2MM calculates after the `FXATM` line. (An earlier version
  applied it when it built the data. The weights of a `COMP` before and
  after the command were then different.)
* **It is optional.** Without the `FXATM` line, Q2MM excludes no atoms. A
  `fixedatoms.txt` file in the directory has no effect by itself. Upstream
  Q2MM reads each file with that name automatically.

> Use `FXATM` only for atoms that were frozen in the QM calculation. Do not
> use it for a reaction-center contact that makes the Amber Hessian very
> large. That problem is a missing bond in the mol2. If you exclude the
> atoms, you hide the problem, but you do not solve it.

---

## 5. Gradient optimizer (`GRAD`)

`GRAD` is a local least-squares fit. It differentiates each selected
parameter and makes a Jacobian. Then it calculates the step that minimizes
the objective. It is fast and precise near a good starting point. With poor
seeds, it can stay in a local minimum.

**Put `GRAD` inside a `LOOP … END` block.** The block repeats `GRAD` until
convergence.

```
LOOP 0.01
GRAD
END
```

### Options

Each option has the form `method=True|False`. Settings for that method can
follow, separated by commas. Put value lists in brackets, with `/` between
the values:

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

* All methods accept `radii=[…]` and `cutoffs=[…]`. `cutoffs` takes exactly
  two values, or `[None]`.
* `lagrange`, `levenberg` and `svd` also accept `factor=[…]`.
* **Include `True` to enable a method.** An option without `True` disables
  the method. For example, `svd=factor=[0.1]` disables SVD.

Each enabled method proposes a trial force field. The trial with the best
score wins the cycle.

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

## 6. Hybrid optimizer (`HYBR`)

A population of trial force fields ("particles") searches the parameter
space. The search uses particle-swarm steps and differential-evolution steps.
It is a **global** search. It is much slower than `GRAD`, but it can leave a
local minimum. It also tolerates seeds that are very wrong. Very wrong seeds
are typical for the forming and breaking TS coordinates.

`HYBR` is the AMBER version of the `HYBR` command on the upstream
[`hybrid-opt`](https://github.com/Q2MM/q2mm/tree/hybrid-opt) branch. It keeps
the same logic. Source: `src/opt.py::SwarmOptimizer` →
`src/hybrid_optimizer.py::PSO_DE`. `src/calculators.py::AmberCalculator`
evaluates each particle.

### Synopsis

```
LOOP <conv>
HYBR [--max_iter N] [--pop_size N] [--tight T|F] [--n_processes N]
END
```

* Options are `--name value` pairs, separated by spaces, **in any order**.
  An unknown option causes an error.
* `HYBR` with no options is correct. It uses all the default values.
* **Put `HYBR` inside a `LOOP … END` block**, as for `GRAD`. The convergence
  value of the block controls the swarm (next section).

### Options

| key | type | default | meaning |
|---|---|---|---|
| `--max_iter` | int | **200** | PSO/DE iterations in each `LOOP` cycle (hard limit) |
| `--pop_size` | int | **24** | number of particles. **Must be even.** |
| `--tight` | T/F | **true** | tight (refine) or global (explore) initial spread |
| `--n_processes` | int | **1** | number of parallel Amber evaluations. `1` = serial. |

Q2MM reads the options when it makes the swarm, at the first `HYBR` of a
`LOOP` block. The later cycles of the block use the same options.

### How the `LOOP` controls `HYBR`

The `LOOP <conv>` value has two functions, as in upstream Q2MM:

1. **Between cycles**, it is the usual loop test. After each `HYBR` cycle,
   the block runs again until the relative score change is less than
   `conv`, or until the score stops changing.
2. **In a cycle**, it is the early-stop tolerance of the swarm. The cycle
   stops before `max_iter` when *each* particle stays within `conv` of the
   best particle, in *each* parameter, for **20 consecutive iterations**.
   This distance is absolute, in the units of each parameter. For force
   constants from 1 to 1500, a `conv` of `0.001` almost never stops the
   cycle. `max_iter` then sets the cycle length. A larger value (for
   example `1`) lets the early stop occur.

The swarm **continues across the cycles of one `LOOP` block**:

* The second `HYBR` continues with the same particles from the point where
  the first stopped, around the best force field so far.
* The first cycle changes the PSO/DE hyperparameters gradually from the
  exploring end to the exploiting end. The later cycles keep them at their
  final values.
* A new `LOOP` block starts a new swarm around the current force field.
* A cycle never returns a force field that is worse than the force field
  that it received. Thus a cycle without improvement ends the block.

### How a particle gets its score

Each **particle is one complete parameter vector**. To score a particle, the
optimizer does these steps:

1. It writes the candidate parameters into the frcmod.
2. It builds the Amber topology again (`tleap` with the `.in` file) and runs
   the calculation.
3. It compares the calculated data with the reference data
   (`score.score_data`).

The swarm **minimizes** that score (§7). If an evaluation fails, the particle
gets the score `inf`. Thus it cannot become the best particle. `HYBR` uses
the **same objective** as `GRAD` and `SIMP`. Only the search is different.

### Option details

**`--pop_size`**: the number of selected parameters is the dimension of the
search. A larger population covers the space better. The cost of each
iteration increases linearly with the population. As a general rule, use
approximately 6 × the number of parameters. A global search needs more
particles than a tight search. The value **must be even**, because the DE
step asserts it.

**`--max_iter`**: this option has the largest effect on the cost:

```
wall-time per cycle ≈ pop_size × max_iter × (Amber eval time) / n_processes
```

The real cost is a little lower, because the DE steps decrease during the
run. `max_iter` is an upper limit for each cycle. The `LOOP` convergence can
stop a cycle earlier. The block runs as many cycles as the convergence
needs.

**`--tight`**: this option sets only the **initial spread**. It does not
change the core PSO/DE hyperparameters. One `DEFAULT_CONFIG` sets them for
both values: inertia 0.9→0.4, cognitive 2.5→0.5, social 0.5→2.5,
`DE/best/1`, differential weight 0.4→0.1, taper-GA on. `--tight` changes
these values:

| | `--tight true` (refine) | `--tight false` (global) |
|---|---|---|
| particles tethered near the current values | 70 % | 30 % |
| initial spread of `af`/`bf` parameters | 0.125 | 1.0 |

The initial deviation for each parameter type is:

* `af` and `bf`: 0.125 or 1.0 (from `--tight`)
* `ae`: 15
* `be`: 0.5
* `df`: 5
* all other types: 1.0

Q2MM accepts these true values, in uppercase or lowercase: `t`, `true`, `1`,
`yes`. All other values are false. Thus `--tight TRUE` and `--tight T` both
give tight. `--tight G`, `--tight false` and `--tight tight` all give
global, because `tight` is not in the list.

**`--n_processes`**: this option changes only the speed, never the result.

* `1` (default): **serial**. Q2MM does one Amber evaluation at a time. It
  makes no extra directories.
* `>1`: **parallel**. Q2MM makes `swarm_particles/p_000 … p_{pop_size-1}`
  in the `DIR` directory. Into each directory, it copies the `.in` file, each
  input file that the `.in` file names with a relative path (frcmod,
  mol2, …), and the pdb. Thus simultaneous `tleap` and Amber jobs cannot
  interfere. It then runs `n_processes` worker processes.

Q2MM uses at most `pop_size` workers. It does not limit the workers to the
number of available cores. Thus set `n_processes` to the number of cores that
you reserved, or fewer.

### Parameters and bounds

The bounds come from columns 3 and 4 of the parameter file, through
`p.allowed_range`. `HYBR` accepts all parameter types. Each parameter gets
its bounds from the file and its initial deviation from its type (above).
The swarm always **prefers the current FF values**, because they are the
center of the tethered particles. A **reflective** boundary sends back each
particle that goes out of bounds. Finite bounds are mandatory (§3).

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
| smoke test (does it run?) | `HYBR --max_iter 2 --pop_size 4 --n_processes 4` |
| refine a force field that is already good | `HYBR --max_iter 300 --pop_size 24 --tight true --n_processes 8` |
| global search from seeds | `HYBR --max_iter 1000 --pop_size 48 --tight false --n_processes 24` |

`HYBR` is **stochastic**. It uses no fixed random seed, so two runs give
different results. Use sufficient `pop_size` and `max_iter` for a stable
result. Consider a second production run to compare the results.

---

## 7. The scoring function

All three optimizers minimize the same objective (`score.compare_data`):

```
score = Σ over types  (1 / N_type) · Σ over points  w² · (reference − calculated)²
```

* The weight is **squared**. Thus a weight of 0 excludes a point completely.
* Q2MM divides the sum of each type by the number of points of that type.
  Thus a 30 000-element Hessian cannot control the score only because of its
  size. The energy types share one divisor: the total number of energies.
  The Hessian uses the number of calculated elements.
* The weights for each type come from `constants.WEIGHTS` (`b` 100, `a` 2,
  `t` 1, `h` 0.031, …). `WGHT` can change them.
* Each Hessian element also gets a weight from the topology:
  * 0 for the 3×3 block of one atom
  * 0.031 for 1-2 and 1-3
  * 0.31 for 1-4
  * 0.031 for longer range
* Eigenmatrix elements use `eig_d_low` / `eig_d_high` (`0.1`) on the
  diagonal and `eig_o` (`0.05`) off the diagonal. An un-inverted TS mode
  uses `eig_i` (`0`) (§4).

If the calculated data and the reference data of a type have a different
number of points, the calculation is incomplete. The score is then `inf`,
and `COMP` writes no table.

The `Score` column in the output is the contribution of each element
**after** the division by `N_type`, with 4 decimals. Thus most Hessian rows
show `0.0000`. Their true values are small but not zero, and Q2MM adds them
to the total. Q2MM excludes a row only when its **Weight** column shows
`0.00`.

---

## 8. Running

Start `loop.py` from the run directory. `RDAT` and `CDAT` find their files
relative to this directory. `root.log` also goes to this directory.

```bash
cd $WORK
python $SRC/loop.py loop.in
```

On the cluster, put these commands in a submit script. Request the number of
cores that you gave to `n_processes`:

```bash
#!/bin/bash
#$ -N q2mm_cycle1
#$ -pe smp 24
#$ -q long

conda activate q2mm
source $AMBERCLASSICHOME/AmberClassic.sh
source $AMBERHOME/amber.sh

cd $WORK
python $SRC/loop.py loop.in
```

> **Before you run again in the same directory:**
>
> * You do not have to delete `swarm_particles/` or `calc/`. Q2MM writes
>   over the files in old particle directories. It also deletes the old
>   Amber outputs before each step.
> * The `ff_NNN.frcmod` numbers continue from the files that are already in
>   the directory.
> * `FFLD read` keeps a one-time `MOL.frcmod.orig` backup. At the start of
>   each run, it restores the frcmod from this backup. Thus each run starts
>   from the same parameters. Delete the `.orig` file only when you changed
>   the initial force field on purpose.

---

## 9. Outputs

| file | contents |
|---|---|
| `frcmod.gaff.01` | the optimized force field (the name that you give to `FFLD write`) |
| `start.txt` / `opt.txt` | the full data comparison before and after the fit |
| `ff_001.frcmod`, `ff_002.frcmod`, … | the best force field after each `LOOP` cycle. The numbers continue from the files that are already in the directory. |
| `MOL.frcmod.orig` | the backup of the initial frcmod (from `FFLD read`) |
| `hybrid_opt_history.bin` | the `HYBR` swarm history: the position, velocity and score of each particle at each iteration (a Python pickle of `X`, `V` and `Y`). Q2MM writes it at the end of each `HYBR` cycle, also when the cycle fails. |
| `root.log` | the complete run log, with the best score of each iteration. It goes to the directory where you start `loop.py`. |
| `amber.MOL.log` | the output of the Amber programs for the last evaluation |
| `calc/` | Amber scratch files: `prmtop`, `inpcrd`, `.ene`, `.lst` (the topology listing that sets the Hessian weights), `.geo`, `.hes` |
| `swarm_particles/` | a scratch directory for each particle (parallel `HYBR` only) |

Compare `start.txt` with `opt.txt`. The `Total score` lines show the
improvement. The breakdown for each type below them shows the source of the
improvement. The log contains `INIT FF SCORE`. For `HYBR`, the log also
contains the line `Iter: k, Best fit: … at [params]` for each iteration.
[`tools/extract_swarm.py`](tools/README.md) reads these lines.

---

## 10. Staged cycles

A fit of all parameters at the same time from rough seeds is
ill-conditioned. The usual method is to fit subsets, in the order of how well
the data determine them. Use the frcmod of each cycle as the input of the
next cycle. Make the convergence tighter at each cycle:

| cycle | parameters | reference data |
|---|---|---|
| 1 | bond + angle **force constants** | Hessian |
| 2 | bond + angle **equilibria** | geometry |
| 3 | both, together | geometry + Hessian |
| 4 | **dihedral** barriers | torsions + Hessian |
| 5 | angles + dihedrals | angles + torsions + Hessian |
| 6 | everything | all of the above |

Fit the force constants first, because the Hessian determines them directly.
Fit the torsions last. They are soft, and their coupling to the stiff terms
is weak.

---

## Troubleshooting

| symptom | cause and solution |
|---|---|
| `OverflowError: Range exceeds valid bounds` | A `HYBR` parameter has an infinite bound. Put a finite `lower upper` pair on each line of `params.txt` (§3). |
| `AssertionError: size_pop must be an even integer for GA` | `pop_size` must be an **even** integer. |
| `Trimmed number of parameters down to 0` | The frcmod header is wrong or missing. `# Q2MM` and `# OPT` must be on **separate** lines, in that order, before `MASS`. `build_tsff.py` does not write this header (§2). |
| The score is `inf`. The log shows `Calculation failed in …` and `the calculation is incomplete. Scoring as inf.` | An Amber step failed, usually `tleap`. Check that `calc/prmtop` exists and that its size is not zero. Read `amber.MOL.log` and `leap.log`. |
| `could not find vdW parameters for type (Du)` | The mol2 still has `DU` or SYBYL types. Build the TSFF again with `build_tsff.py` ([`tools/README.md`](tools/README.md)). |
| `Hessian file missing: …hes`, or `nmode wrote no calc/hessian.mat` | `nab`/`nmode` wrote no `hessian.mat`. An earlier step failed (check `prmtop`), or AmberTools does not have the patch (§0). |
| The score changes very little, but the parameters change a lot | Residuals that your selected parameters cannot change control the objective. The usual cause is a forming or breaking contact that is **not bonded** in the mol2. Amber then scores the contact as a nonbonded clash with a very large Hessian. Correct the topology, not the optimizer. |
| `HYBR` uses fewer workers than you requested | `n_processes` is more than `pop_size`. Q2MM uses at most `pop_size` workers. This changes only the speed. |
| `HYBR` overloads the node | `n_processes` is more than the number of reserved cores. Q2MM does not limit the workers to the available cores. |
| Two runs give different results | This result is normal. `HYBR` is stochastic and has no fixed seed. |
| `HYBR: unrecognised option(s)` | Options are `--name value` pairs (`--max_iter 200`). The old `name=value` form and `precision` are not available. The `LOOP` convergence is the precision now. |
| `-ah …: the archive geometry of … and … differ by up to … the QM and force-field frames do not match` | The mol2 or the pdb is not in the frame of the `-gh` log. Usually it is a file from an earlier frequency job with a different input orientation. Build the mol2 from the archive geometry of that log. Delete the old pdb, because Q2MM makes the pdb again only when it does not exist (§4). |
| `-ageig …: … differ by up to … the QM and force-field frames do not match` | The mol2 or the pdb is not in the frame of the normal modes. Usually the frequency job ran without `nosymm`. You then built the mol2 from the archive geometry. There are three solutions (§4, `--modes`). Use `--modes archive` on both lines. Or build the mol2 in the standard orientation of the log. Or run the frequency job again with `nosymm freq=hpmodes`. A pdb from an older mol2 also causes this error. Delete that pdb. |
| `-ageig … uses --modes …, but -geigz read … with --modes …` | The `RDAT` and `CDAT` lines use different mode sources. Give the same `--modes` on both lines. |
