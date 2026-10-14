# `tools/`

This directory has two scripts:

| script | purpose |
|---|---|
| [`build_tsff.py`](#build_tsffpy-source-resolved-hybrid-force-fields) | Builds the initial TSFF: the re-typed mol2, the frcmod, the LEaP files, and CSV reports of the TS terms to fit. |
| [`extract_swarm.py`](#extract_swarmpy-read-a-hybr-history) | Reads the `HYBR` history of a run from `root.log` and from the particle directories. |

---

# `build_tsff.py`: source-resolved hybrid force fields

`build_tsff.py` makes one topology-specific AMBER frcmod from these inputs:

* a RESP mol2
* a published parameter set
* ff19SB
* GAFF2
* optional custom parameters, inline or in a custom frcmod

The THI example uses published NADH parameters. The builder does not change
the input coordinates, atom names, connectivity or RESP charges.

## Usage

```bash
source $AMBERHOME/amber.sh
python build_tsff.py atom_mapping.txt -o output_prefix [options]
```

The script finds `tleap` and `parmchk2` in `$AMBERHOME/bin`, or on `PATH`.

| argument | meaning |
|---|---|
| `atom_mapping.txt` (required) | the mapping file (refer to "Mapping file") |
| `-o`, `--out` (required) | the output prefix. `-o MOL` writes `MOL.mol2`, `MOL.frcmod`, and the other outputs. |
| `--no-verify` | write the outputs, but do not run `tleap` |
| `--allow-ts-seeds` | make flagged initial angle and torsion seeds when a custom TS term has no source parameter (refer to "TS seeds") |
| `--ts-equilibria` | take the equilibrium values of the reaction-center bonds and angles from the input TS geometry (refer to "TS equilibria") |

## Resolution rules

Atom type and parameter source are two different concepts. Each atom has:

* one **final type**, which the builder writes to the output mol2
* one **category**, which describes the region of the atom in the reports
* one or more **source aliases**, which the builder uses only to find
  parameters

The resolver tries the sources in the order of `[PRIORITY]`. `custom` is
always first. Sources that `[PRIORITY]` does not list go to the end. The
example order is `custom > published > ff19sb > gaff2`.

A bonded term can use a source only when each atom in that term has an alias
for that source. The resolver never makes one term from types of different
sources.

The `published` source is the published frcmod overlay from the mapping
file, plus its ff19SB dependency. In the THI example, `frcmod.NADH` wins for
the published nicotinamide terms and for the nicotinamide/ribose boundary
terms. For the usual ribose terms, the CSV report shows ff19SB as the source
of the numbers.

The script rejects these terms:

* unresolved bonds, angles and proper torsions
* `parmchk2` rows marked `ATTN, need revision`. Those zero values are
  diagnostics, not parameters.

The builder includes an improper torsion only when a source matches it. It
does not infer an improper only because an atom has three neighbours.

Without `--allow-ts-seeds`, each unsupported term is an error, and the
builder makes no new terms from the TS geometry. With that option, the
builder can make a small set of reported initial seeds for the Q2MM fit
(refer to "TS seeds").

The builder copies all Fourier rows of a matched proper torsion. Thus a
multi-term torsion stays multi-term.

## Mapping file

The input is a plain text file with these sections:

```text
[FILES]
MOL2 molecule_resp.mol2
PUBLISHED_LIB NADH.lib
PUBLISHED_FRCMOD frcmod.NADH
FF19SB leaprc.protein.ff19SB
GAFF2 leaprc.gaff2

[PRIORITY]
custom
published
ff19sb
gaff2

[FF19SB_TEMPLATES]
# library       residue unit  target anchor
amino19.lib     HIP           C9
amino19.lib     GLU           C22
amino19.lib     LYS           N6
amino19.lib     ASP           C25

[TS_BONDS]
# target atom names; both bonds remain covalent
O4   H19   breaking
H19  O7    forming

[CUSTOM_PARAMETERS]
# section  exact final-type key  numerical values
ANGLE AO-BH-OB 50.0000 171.8433

[ATOMS]
# name category final_type source:type [source:type ...]
C1  published    CF  published:CF
C2  published    DC  published:CH
C6  ff19sb  auto  published:CT ff19sb:auto
C9  ff19sb  auto  ff19sb:auto
C16 custom   QC  gaff2:c3
C13 gaff2    original gaff2:original
```

### `[FILES]`

The paths in `[FILES]` are relative to the mapping file. You can give a
leaprc by name or by path.

`[FILES]` can also have an optional `CUSTOM_FRCMOD <file>` line. The builder
adds the terms of that file to the `custom` source. A custom frcmod is not
necessary. You can write custom values directly in `[CUSTOM_PARAMETERS]`.

### `[ATOMS]`

Each mol2 atom must occur exactly once in `[ATOMS]`, with its unique atom
name. The four categories are `custom`, `published`, `ff19sb` and `gaff2`.
They describe the atom in the figure. They do not force each term that
touches the atom to use that source.

The third column in `[ATOMS]` is an AMBER **atom type**, not a mol2 atom
name. A custom final type changes the type label of the atom. The terms of
that type then become independent parameters that Q2MM can optimize. By
default, all initial parameters come from the declared aliases. For example,
`O4 custom AO gaff2:oh` writes `AO` to the output mol2. Where no exact custom
term exists, the builder uses the GAFF2 `oh` values. The atom name `O4` and
the RESP charge do not change.

You can use `original` as a final type or as an alias. It expands to the
type in the input mol2. `auto` for an ff19SB atom first uses a matched
residue template. If the atom is outside those templates, the builder uses
conservative rules for the saturated C, O and H environments that the ribose
needs.

Aliases must be chemically intentional. These are examples from the THI
mapping:

* `DC` has `published:CH`. Thus the DC label in the figure is an exact alias
  of the published CH type.
* Ribose `C6` has `published:CT` and `ff19sb:auto`. Automatic environment
  typing gives `CT` as the final type and as the ff19SB type. The published
  alias lets the NADH boundary terms win.
* Custom `QC` has `gaff2:c3`. This alias gives a parent for the initial
  values where no exact custom QC term exists.
* Custom `BC` and `OB` use `ff19sb:auto`. The GLU template match sets their
  parent aliases to `CO` and `O2`. Their final custom names do not change.

The builder graph-matches `NADH.lib` against the mol2 and checks each
declared `published:` alias. This check finds a published type on the wrong
atom. It does not change the RESP charges in the mol2.

### `[CUSTOM_PARAMETERS]`

This section is optional. Use it to replace an inherited value for an exact
combination of final atom types. You do not need a separate frcmod for this.
The supported section names and value columns are:

| Section | Row after the section name |
|---|---|
| `MASS` | `TYPE mass [polarizability]` |
| `NONBON` | `TYPE radius epsilon [screen]` |
| `BOND` | `TYPE-TYPE force_constant equilibrium_distance` |
| `ANGLE` | `TYPE-TYPE-TYPE force_constant equilibrium_angle` |
| `DIHE` | `TYPE-TYPE-TYPE-TYPE idivf barrier phase periodicity` |
| `IMPROPER` | `TYPE-TYPE-TYPE-TYPE barrier phase periodicity` |

The O-H-O example above replaces only `AO-BH-OB`. Other terms that contain
AO, BH or OB still get their parameters through their aliases. To give more
than one Fourier term, write more than one `DIHE` row. If an inline custom
bonded term spans a declared `[TS_BONDS]` pair, the TS fit report includes
it.

### `[FF19SB_TEMPLATES]`

Each row gives an Amber library, a residue unit, and one unique atom name in
the target mol2. The anchor selects the correct fragment. The graph matcher
assigns the atom types of the residue. The atom names and the atom numbers do
not have to agree. For example, `HIP C9` assigns
`CC/CW/CR/NA/H/H4/H5/CT/HC` across the truncated His 381 fragment. The Glu,
Lys and Asp rows work in the same way. The run log reports, for each residue
template, the number of heavy atoms and the total number of atoms that it
matched.

### `[TS_BONDS]`

This section lists the reaction-coordinate bonds that stay covalent in the
TS topology. The builder keeps their available GAFF2 parameters as initial
values. The parameter report labels them `ts_initial_breaking` or
`ts_initial_forming`, so that you include them in the Q2MM fit.

In some TS topologies, a hydrogen has bonds to both the donor and the
acceptor. For this case, the builder writes a `.leap.mol2` loading copy. The
LEaP input then adds the forming bond again under the temporary perturbation
override of LEaP. After LEaP saves the prmtop, the builder compares the bond
graph of the prmtop with the complete output mol2. This check prevents a
silent loss of the bond to a two-coordinate hydride.

## Outputs

For `-o THI.figure`, the script writes these files:

| File | Purpose |
|---|---|
| `THI.figure.mol2` | the molecule with the final atom types and the unchanged RESP charges |
| `THI.figure.leap.mol2` | a LEaP loading copy. The builder writes it when a transferring H with two bonds needs its forming bond back after loading. |
| `THI.figure.frcmod` | all parameters that this topology needs. It has no Q2MM header (refer to "Next step"). |
| `THI.figure_atomtypes.leap` | LEaP definitions (`addAtomTypes`) for each final type in the topology. The generated LEaP inputs do not load this file (refer to the note below). |
| `THI.figure_parameter_report.csv` | the source, source file, aliases, values, region and atom instances of each parameter in the frcmod |
| `THI.figure_ts_seed_report.csv` | the generated TS seed terms. Q2MM must optimize them. |
| `THI.figure_ts_fit_report.csv` | the breaking and forming bonds, the inline reaction-center overrides, and the coupled seed terms. Select these terms for the Q2MM fit. |
| `THI.figure_atom_report.csv` | the input type, final type, category, aliases, assignment method and charge of each atom |
| `THI.figure.leap.in` | the production LEaP input. It loads GAFF2, ff19SB and the generated frcmod, with absolute paths. |
| `THI.figure.standalone.leap.in` | the completeness check. It loads only the generated frcmod, without GAFF2 or ff19SB. |
| `THI.figure.prmtop`, `THI.figure.inpcrd` | the topology and coordinates from the production LEaP check |
| `THI.figure.standalone.prmtop`, `THI.figure.standalone.inpcrd` | the topology and coordinates from the standalone check |

If you do not give `--no-verify`, both LEaP inputs must finish with zero
errors. The standalone run shows that the frcmod contains each numerical
parameter that this mol2 uses. LEaP can show a close-contact warning for a
TS geometry. This warning can be physically correct.

> **Note: the atom-types file.** The two generated LEaP inputs do not load
> `THI.figure_atomtypes.leap`. Thus tleap does not know the elements of the
> custom atom types. It gives them no element, or a wrong element, in the
> prmtop. For example, the THI model gets boron for `BC` and `BH`. tleap sets
> the GB radius and the GB screening factor of each atom from its element. The
> Q2MM fit uses these values in its GB electrostatics. For a QM reference in a
> solvent (ε > 1), wrong elements give wrong Born radii. QM/MM and other
> element-based tools also use the elements. To get the correct elements, put
> `source THI.figure_atomtypes.leap` after the two `source leaprc` lines.
> Refer to [`OPTIMIZATION.md`](../OPTIMIZATION.md) §2 and §4.

## TS seeds (`--allow-ts-seeds`)

Use `--allow-ts-seeds` for an initial TSFF with unsupported terms that
involve custom atoms.

* Terms with a GAFF2 parent still get the GAFF2 values. For the THI model,
  `AO-BH` and `BH-OB` both get the `ho-oh` bond values.
* If an angle has no source parameter, its equilibrium angle comes from the
  input TS geometry. Its initial force constant is 50 kcal/mol/rad².
* An unsupported proper torsion gets a zero barrier.

The frcmod marks each such term `GENERATED_TS_SEED`, and
`_ts_seed_report.csv` lists it. Q2MM must fit these terms. They are not final
parameters. A missing standard term still stops the build.

## TS equilibria (`--ts-equilibria`)

`--ts-equilibria` keeps each resolved force constant. It replaces the
equilibrium length or angle of each reaction-center bond and angle with the
value measured in the input TS geometry.

* A term is a reaction-center term when each instance of its final-type key
  involves a `custom` atom. Thus the keys of the host region never change.
* If a key has more than one instance, the builder uses their mean.
* The frcmod comment (`TS_GEOMETRY_EQ was ... n=...`) and the `equilibrium`
  column of the report record the old value, the number of instances and the
  range.
* The builder does not change explicit `[CUSTOM_PARAMETERS]` values or
  generated seeds.

Use this option when a Q2MM step fits only force constants to the TS
Hessian. A ground-state reference value puts strain on its term at the TS.
The bond tension in the MM Hessian then goes into distorted force constants,
because the fit has no other way to absorb it.

## Next step: prepare the fit

`loop.py` cannot use the builder outputs without changes. Do these steps.
[`OPTIMIZATION.md`](../OPTIMIZATION.md) §2 and §3 give the details.

1. Add the Q2MM header to the frcmod. Put `# Q2MM` and then `# OPT`, each
   on its own line, after the title line and before `MASS`. Without the
   header, Q2MM reads 0 parameters and gives no error.
2. Write the tleap script `MOL.in` for the fit. It must save `calc/prmtop`
   and `calc/inpcrd`, and it must use relative file names. Do not use
   `THI.figure.leap.in` without changes, because it uses absolute paths.
   Put `source THI.figure_atomtypes.leap` after the two `source leaprc`
   lines (refer to the note above).
3. If the builder wrote a `.leap.mol2`, copy its `loadmol2` line into
   `MOL.in`. Also copy the `set … pert` and `bond` lines after it.
4. In the parameter file, select the terms in `_ts_fit_report.csv` and in
   `_ts_seed_report.csv`.

## Diagnosing a failure

An unresolved-term error prints the final types, the atom names, the declared
aliases, and each source that the builder tried. To resolve the term, do one of
these:

* Add a chemically justified alias.
* Add the exact final-type term as a custom parameter, in
  `[CUSTOM_PARAMETERS]` or in the custom frcmod.

Do not add an alias only to stop the error. An alias states that the atom
really has that identity in that parameter source.

If LEaP reports a missing term that the CSV lists, examine the rendered key
in the frcmod. The builder keeps the numerical parameter text. It moves the
numerical citation fields of Amber `.dat` files into comments, so that cloned
keys stay valid.

---

# `extract_swarm.py`: read a HYBR history

`extract_swarm.py` gets swarm data from the log and the particle directories
of a `HYBR` run.

```bash
python3 extract_swarm.py <run_dir> [-o out.npz]
```

The script prints a summary. With `-o`, it also writes the data to an `.npz`
file.

It reads three sources:

* **The gbest trajectory**: the `Iter: N, Best fit: Y at [X]` lines in
  `root.log`.
* **The taper schedule**: the `cp: ... w: ... cg ...` lines in `root.log`.
* **The last position of each particle**:
  `swarm_particles/p_NNN/<ff>.frcmod` (parallel runs only). Each file
  contains the last position that the particle evaluated.

Use the script in these two cases:

1. **Runs from before the history dump.** `loop.py` now writes
   `hybrid_opt_history.bin` at the end of each `HYBR` cycle. Older runs do
   not have this file.
2. **The gbest trajectory.** `hybrid_opt_history.bin` keeps only the final
   best position. The log records the best position at each iteration.

The per-iteration positions and scores of the whole swarm are only in
`hybrid_opt_history.bin`. Q2MM never records pbest.

The script reads the frcmod files that q2mm-amber wrote. Their columns are
wider than in a typical hand-made frcmod. `parse_frcmod` uses constant column
offsets. They are correct for that layout, but not for arbitrary input.
