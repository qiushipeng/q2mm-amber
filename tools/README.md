# `build_tsff.py`: source-resolved hybrid force fields

`build_tsff.py` constructs one topology-specific AMBER frcmod from a RESP
mol2, a published parameter set, ff19SB, GAFF2, and optional inline custom
parameters. The THI example uses published NADH parameters. The builder keeps
the input coordinates, atom names, connectivity, and RESP charges unchanged.

The command has two arguments:

```bash
python build_tsff.py atom_mapping.txt -o output_prefix
```

For the THI model in this workspace:

```bash
cd /groups/owiest/Qiushi/test/thio/prep
./build_figure.sh
```

The wrapper sources AmberTools from `$AMBERHOME`, or from
`/groups/owiest/Q2MM/tools/ambertools26` when `AMBERHOME` is unset.

## Resolution rules

Atom type and parameter source are separate concepts. Every atom has:

- one **final type**, written to the output mol2;
- one **category**, used to describe the region in the reports;
- one or more **source aliases**, used only for parameter lookup.

The resolver checks `custom > published > ff19sb > gaff2`. A bonded term can use a
source only when every atom in that interaction declares an alias for the same
source. It never assembles one term from types belonging to different sources.

The `published` source is the configured published frcmod overlay plus its
ff19SB dependency. In the THI example, `frcmod.NADH` wins for published
nicotinamide and nicotinamide/ribose boundary terms. Ordinary ribose terms
retain their ff19SB numerical provenance in the CSV report.

The script rejects unresolved bonds, angles, and proper torsions. It also
rejects `parmchk2` rows marked `ATTN, need revision`; those zero values are
diagnostics rather than parameters. Improper torsions are included only when a
source actually matches one. No improper is inferred merely because an atom
has three neighbours.

Without `--allow-ts-seeds`, the TS geometry is not used to create parameters
and every unsupported interaction is an error. With that explicit option, the
builder can make the small set of reported initial seeds described below for
subsequent Q2MM fitting.

All Fourier rows of a matched proper torsion are copied, so a multi-term
torsion remains multi-term.

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

Paths in `[FILES]` are resolved relative to the mapping file. A leaprc may be
given by name or by path. No separate custom frcmod is needed; custom values
can be written directly in `[CUSTOM_PARAMETERS]`.

Every mol2 atom must occur exactly once in `[ATOMS]`, addressed by its unique
atom name. The four categories are `custom`, `published`, `ff19sb`, and `gaff2`.
They describe the atom in the figure; they do not force every interaction
touching that atom to use that source.

The third column in `[ATOMS]` is an AMBER **atom type**, not a mol2 atom name.
Giving an atom a custom final type changes that type label and makes its terms
independently optimizable. By default, all starting parameters are inherited
from the declared aliases. For example, `O4 custom AO gaff2:oh` writes `AO` to
the output mol2 while using GAFF2 `oh` values wherever no exact custom term is
present. Atom names such as `O4` and RESP charges are unchanged.

`[CUSTOM_PARAMETERS]` is optional. Use it to replace an inherited value for an
exact combination of final atom types without maintaining a separate frcmod.
The supported section names and value columns are:

| Section | Row after section name |
|---|---|
| `MASS` | `TYPE mass [polarizability]` |
| `NONBON` | `TYPE radius epsilon [screen]` |
| `BOND` | `TYPE-TYPE force_constant equilibrium_distance` |
| `ANGLE` | `TYPE-TYPE-TYPE force_constant equilibrium_angle` |
| `DIHE` | `TYPE-TYPE-TYPE-TYPE idivf barrier phase periodicity` |
| `IMPROPER` | `TYPE-TYPE-TYPE-TYPE barrier phase periodicity` |

The O-H-O example above overrides only `AO-BH-OB`. Other terms containing AO,
BH, or OB still inherit parameters through their aliases. Repeat a `DIHE` row
to provide multiple Fourier terms. An inline custom bonded term spanning a
declared `[TS_BONDS]` pair is included in the TS fitting report.

`original` is allowed for a final type or an alias and expands to the type in
the input mol2. `auto` for an ff19SB atom first uses a matched residue template.
When the atom is outside those templates, the builder handles the conservative
saturated C/O/H environments needed by the ribose.

Each `[FF19SB_TEMPLATES]` row gives an Amber library, a residue unit, and one
unique atom name in the target mol2. The anchor selects the intended fragment;
the graph matcher assigns the residue's atom types without relying on matching
atom names or numbering. For example, `HIP C9` automatically assigns
`CC/CW/CR/NA/H/H4/H5/CT/HC` across the truncated His 381 fragment. The Glu,
Lys, and Asp rows work the same way. The run log reports how many heavy atoms
and total atoms each residue template matched.

`[TS_BONDS]` records the reaction-coordinate bonds that remain covalent in the
TS topology. Their available GAFF2 parameters are retained as initial values,
but the parameter report labels them `ts_initial_breaking` or
`ts_initial_forming` so they are included in the Q2MM fitting list.
For a hydrogen covalently connected to both donor and acceptor, the builder
writes a `.leap.mol2` loading copy and restores the forming bond under LEaP's
temporary perturbation override. It then checks the saved prmtop bond graph
against the complete output mol2, preventing LEaP from silently dropping a
two-coordinate hydride bond.

Aliases must be chemically intentional. Examples from the THI mapping are:

- `DC` has `published:CH`, making the figure's DC label an exact published CH alias.
- Ribose `C6` has `published:CT` and `ff19sb:auto`; automatic environment typing
  resolves the final and ff19SB type to `CT`, while the published alias lets
  NADH boundary terms win.
- Custom `QC` has `gaff2:c3`, which supplies a starting parent where the custom
  frcmod has no exact QC term.
- Custom `BC` and `OB` use `ff19sb:auto`; matching the GLU template determines
  their parent aliases as `CO` and `O2` while their final custom names remain.

The builder graph-matches `NADH.lib` against the mol2 and checks every declared
`published:` alias. This catches a published type assigned to the wrong atom while
leaving the mol2 RESP charges untouched.

## Outputs

For `-o THI.figure`, the script writes:

| File | Purpose |
|---|---|
| `THI.figure.mol2` | molecule with final atom types and unchanged RESP charges |
| `THI.figure.leap.mol2` | LEaP loading copy used when a two-coordinate transferring H requires its forming bond to be restored after loading |
| `THI.figure.frcmod` | all parameters required by this topology |
| `THI.figure_atomtypes.leap` | LEaP definitions for every final type in the topology |
| `THI.figure_parameter_report.csv` | source, source file, aliases, values, region, and atom instances for every emitted parameter |
| `THI.figure_ts_seed_report.csv` | generated transition-state seed terms that must be optimized with Q2MM |
| `THI.figure_ts_fit_report.csv` | breaking/forming bonds, inline reaction-center overrides, and coupled seed terms to select for Q2MM fitting |
| `THI.figure_atom_report.csv` | input/final type, category, aliases, assignment method, and charge for every atom |
| `THI.figure.leap.in` | reproducible production LEaP input using GAFF2 and ff19SB |
| `THI.figure.standalone.leap.in` | completeness check using only the generated atom-type file and frcmod |
| `THI.figure.prmtop`, `THI.figure.inpcrd` | topology and coordinates from the production LEaP check |
| `THI.figure.standalone.prmtop`, `THI.figure.standalone.inpcrd` | topology and coordinates from the standalone check |

Unless `--no-verify` is given, both LEaP inputs must finish with zero errors.
The standalone run proves the frcmod contains every numerical parameter used
by this particular mol2. A close-contact warning can still be physically
expected for a transition-state geometry.

For an initial TSFF with unsupported interactions involving custom atoms, pass
`--allow-ts-seeds`. Terms with a GAFF2 parent still inherit the GAFF2 values;
for the THI model this gives both `AO-BH` and `BH-OB` the `ho-oh` bond values.
If an angle has no source parameter, its equilibrium angle comes from the input
TS geometry with an initial force constant of 50 kcal/mol/rad². Unsupported
proper torsions receive a zero barrier. Every such term is marked
`GENERATED_TS_SEED` in the frcmod and isolated in `_ts_seed_report.csv`; these
terms are intended for Q2MM fitting, not as final parameters. Missing standard
interactions still stop the build.

`--ts-equilibria` keeps every resolved force constant but replaces the
equilibrium length or angle of each reaction-center bond and angle with the
value measured in the input TS geometry. A term is reaction center when every
instance of its final-type key involves a `custom` atom, so host-region keys
never change; a key with several instances takes their mean. The frcmod
comment (`TS_GEOMETRY_EQ was ... n=...`) and the report's `equilibrium` column
record the replaced value, the instance count and the range. Explicit
`[CUSTOM_PARAMETERS]` values and generated seeds are left unchanged. Use it
when a Q2MM step fits only force constants to the TS Hessian: a ground-state
reference value strains its term at the TS, and the resulting bond tension in
the MM Hessian can only be absorbed by distorting force constants.

## Diagnosing a failure

An unresolved-term error prints the final types, atom names, declared aliases,
and every source attempted. Resolve it by either adding a chemically justified
alias or adding the exact final-type term to the custom frcmod. Do not add an
alias merely to silence the error: it states that the atom really has that
identity in that parameter source.

If LEaP reports a missing term even though it is listed in the CSV, inspect the
rendered key in the frcmod. This builder preserves numeric parameter text and
moves numeric citation fields from Amber `.dat` files into comments so cloned
keys remain valid.
