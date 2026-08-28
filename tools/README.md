# build_tsff.py — transition-state force fields from a RESP mol2

One script, one command:

```
MOL_resp.mol2  ->  MOL.mol2  +  MOL.frcmod
```

A re-typed structure and a **single self-contained frcmod**. Nothing else to
load, nothing else to remember.

```bash
python $TOOLS/build_tsff.py MOL_resp.mol2 --sel sel.txt \
    --published COFACTOR.lib frcmod.COFACTOR -o MOL
```

It uses only the Python standard library and shells out to `parmchk2` and
`tleap`, so it can be run directly from this directory.

---

## Placeholders used below

| placeholder | meaning |
|---|---|
| `$TOOLS` | this directory |
| `$AMBERHOME` | your AmberTools installation |

```bash
export AMBERHOME=/path/to/ambertools
export PATH=$AMBERHOME/bin:$PATH
export TOOLS=/path/to/q2mm-amber/tools
```

`gaff2.dat` is found automatically under `$AMBERHOME`; override with `--gaff`.

---

## Inputs

| file | required | description |
|---|---|---|
| `MOL_resp.mol2` | yes | your molecule: GAFF/GAFF2 types, RESP charges, **TS geometry** |
| `sel.txt` | for a TS | the reactive atoms to give custom types |
| `COFACTOR.lib` + `frcmod.COFACTOR` | optional | a published parameter set to adopt |

### `sel.txt`

One atom per line: **mol2 index, new type, parent type**.

```
# atom_index   new_type   parent_type
39   AC   c3
40   AH   h2
43   AO   oh
44   BH   ho     # transferring proton -- was typed DU
45   AS   ss
```

* `new_type` — any 1–2 character label, as long as it is not a real GAFF type.
  It is only a lookup key; giving an atom a unique type is what makes its
  parameters privately fittable instead of shared with every similar atom.
* `parent_type` — the existing type whose parameters are cloned. Optional when
  the atom already carries a usable type; **required** for `DU` atoms.

> **Why `DU` appears.** `antechamber` types an atom `DU` when it cannot assign
> one — typically the in-flight proton of a transition state, bonded to two
> heavy atoms at once. `DU` has no mass, radius or parameters, so you must name
> a real parent yourself (`ho` for a transferring hydroxyl proton). The script
> stops with an explicit message if you forget.

---

## What it does

```
1. match each --published template against your structure, adopt its types
2. apply --sel on top (always wins)
3. run parmchk2 on an all-GAFF copy -> gap-fills
4. generate every term the new types touch
5. write ONE frcmod + the re-typed mol2, verify with tleap
```

### 1. Published types, by structure matching

The template and your molecule are compared as **graphs** (elements + bonds)
and the largest common substructure is found; every matched atom inherits the
template's type. Atom numbering, naming and ordering are irrelevant, so the
same command works on any structure of the same fragment.

Matching (rather than exact isomorphism) is necessary because your model is
usually a **truncation** of the published molecule — you keep one ring, the
published library holds the whole cofactor.

**Where the transfer stops:** only types the published frcmod *defines* (has a
`MASS` entry for) are adopted. Published sets routinely reference types they do
not define, because those belong to a protein force field; this rule ends the
transfer at the parameterised region and leaves the junction on GAFF.

To adopt those host-force-field types as well, source the force field that
defines them when verifying:

```bash
--leaprc leaprc.protein.ff19SB
```

### 2. Custom types on top

`--sel` is applied last and wins. A reactive atom inside a published fragment
takes the **published type as its parent**, so its parameters are cloned from
the published values rather than from GAFF — isolation without losing the
published physics.

### 3. GAFF gaps, handled silently

**This is the step that used to bite.** GAFF does not cover everything. The
subtypes `c5`/`c6` (sp3 carbons in 5- and 6-membered rings) have **no wildcard
torsions** — there is no `X-c5-c5-X` the way there is `X-c3-c3-X` — so any
sugar or saturated ring produces a pile of

```
** No torsion terms for atom types: h1-c5-c5-oh
```

errors in tleap, an empty prmtop, and then empty Hessians and `inf` scores in
the optimiser, far from the real cause.

`parmchk2` fills those gaps by falling back to the generic type
(`same as X-c3-c3-X, penalty score= 0.0`). The script runs it internally and
folds the result into the output, so the problem never reaches you and there is
only one file to carry.

> Note the physics: the ring-strain benefit of `c5` survives in bonds and
> angles, but its torsions revert to acyclic `c3` values. If ring puckering
> matters for your reaction, those are the weakest terms in the force field.

### 4. Parameter sources, in priority order

1. **published frcmod** — used verbatim, no geometry override
2. **parmchk2 gap-fills** — including junction terms GAFF has no analog for
3. **gaff2.dat** — everything else, cloned via each atom's parent type

Terms straddling two schemes (a published type bonded to a custom type whose
parent is published) are found by trying every combination of each atom's
effective and parent type.

Independently: if an analog exists but your geometry deviates from it
(> 0.08 Å, > 12°), the equilibrium value is taken from **your TS geometry** and
the force constant is demoted to a seed. That is how a partial bond keeps its
stretched length.

### 5. Output

| file | contents |
|---|---|
| `MOL.mol2` | re-typed structure — coordinates, names and charges byte-identical to the input |
| `MOL.frcmod` | the complete force field, self-contained |
| `MOL.frcmod.review` | every seeded term, for the optimiser's parameter selection |

tleap then runs automatically and must report `Errors = 0`. Use `--no-verify`
to skip it.

---

## Reading the review file

```
BOND  AO-BH: reactive, req=1.2705 from geometry (GAFF 0.9725); k=535.51 is a seed
ANGLE AO-BH-OB: NO GAFF analog; theta=171.84 from geometry, k=50.0 is a seed
DIHE  AC-AO-BH-OB: NO GAFF analog; zero-barrier placeholder
```

**This is a checklist, not an error log.** Every line is a parameter that is a
*seed* rather than a derived value:

* **`reactive`** — an analog exists but your geometry is far from it. These are
  the forming/breaking coordinates. The equilibrium value is right (it comes
  from your QM geometry); the **force constant is not**.
* **`NO GAFF analog`** — the term does not exist in GAFF at all, because the
  arrangement is impossible in a ground state (a hydrogen bonded to two heavy
  atoms). Force constant and torsion barrier are both placeholders.

**Every term listed here must be selected in the optimiser's parameter file.**
These force constants are exactly what the fit determines; unselected, the
placeholders survive into the final force field.

Equally important: **nothing unexpected should appear**. Entries should cluster
around the reacting atoms. A seeded term elsewhere means a parameter found no
analog — investigate before optimising.

---

## Options

| flag | meaning |
|---|---|
| `-o, --out` | output prefix → `<out>.mol2`, `<out>.frcmod` |
| `--sel` | custom/reactive atom types |
| `--published TEMPLATE FRCMOD` | a published set; repeatable |
| `--gaff` | `gaff2.dat` (default: found under `$AMBERHOME`) |
| `--leaprc` | extra leaprc to source when verifying; repeatable |
| `--no-verify` | skip the tleap check |

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `atom N is typed DU but is not selected` | a `DU` atom has nothing to inherit. Add it to `sel.txt` with a real parent (`ho` for a transferring hydroxyl proton). |
| `atom N has parent 'DU'` | you wrote `DU` as the parent. Use a real GAFF type. |
| `did not match the structure` | the template is not the same fragment, or your model is missing part of it. Check the printed match count. |
| `adopted 0 published types` | the frcmod defines none of the matched types. Confirm it has `MASS` entries for them. |
| tleap reports a missing **improper** | impropers are not generated; add the rare one by hand. GAFF wild-cards most of them. |
| `cannot find parmchk2` | `AMBERHOME` is not set, or `$AMBERHOME/bin` is not on `PATH`. |
| all bonds show order 1 in a viewer | cosmetic only — AMBER reads parameters from atom types and ignores mol2 bond orders. `antechamber` gives up on bond perception when a TS geometry has an atom of impossible valence. |

---

## A note on row numbers

The optimiser's parameter selection (`BandA_FC.txt` and friends) indexes
**rows** in the frcmod. Because this script emits one merged file, its rows
differ from the older two-file layout — regenerate any selection file built
against the old split, or it will silently select the wrong parameters.
