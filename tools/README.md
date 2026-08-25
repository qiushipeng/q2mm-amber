# Atom-typing tools for building a transition-state force field

Two standalone scripts that turn a plain GAFF/GAFF2 `mol2` into a
tleap-ready transition-state force field (TSFF):

| script | what it does |
|---|---|
| `map_published.py` | transfers **published** atom types onto your molecule by structure matching, and merges your custom selection |
| `clone_atom_types.py` | re-types the selected atoms and writes the `frcmod` supplying every parameter those new types touch |

They only use the Python standard library and import nothing from the rest of
the package, so they can be run directly from this directory.

---

## Conventions used below

Replace these with your own paths:

| placeholder | meaning |
|---|---|
| `$TOOLS` | this directory |
| `$AMBERHOME` | your AmberTools installation |
| `$GAFF` | `$AMBERHOME/dat/leap/parm/gaff2.dat` |
| `$WORK` | the directory holding your molecule |

```bash
export AMBERHOME=/path/to/ambertools
export PATH=$AMBERHOME/bin:$PATH
export GAFF=$AMBERHOME/dat/leap/parm/gaff2.dat
export TOOLS=/path/to/q2mm-amber/tools
```

Use a Python 3 interpreter (a conda environment is fine).

---

## Inputs you need

| file | description |
|---|---|
| `MOL_resp.mol2` | your molecule: RESP charges, GAFF2 atom types, **TS geometry** |
| `sel.txt` | the reactive atoms you want custom types for |
| `PUB.lib` | published structure carrying the published atom types (`.lib`/`.off`/`.mol2`) |
| `frcmod.PUB` | published parameters |

The last two are only needed if part of your molecule has published
parameters (a cofactor, a substrate analogue, ...). Skip them otherwise.

### `sel.txt` format

One atom per line: **mol2 atom index, new type, parent type**.

```
# atom_index   new_type   parent_type
39   AC   c3
40   AH   h2
43   AO   oh
44   BH   ho     # transferring proton -- was typed DU
45   AS   ss
```

* `new_type` — any 1-2 character label you choose, must not collide with a
  real GAFF type.
* `parent_type` — the real GAFF type the new type is *derived from*.
  Parameters are cloned from it. It may be omitted when the atom already
  carries a usable GAFF type, but is **required** for atoms typed `DU`.

> **Why `DU` appears.** `antechamber` types an atom `DU` when it cannot assign
> it — typically the in-flight proton of a transition state, which is bonded
> to two heavy atoms at once. `DU` is a placeholder with no mass, radius or
> parameters, so you must name a real parent yourself (`ho` for a transferring
> hydroxyl proton).

---

## Step 1 — assign the atom types

### With published parameters for part of the molecule

```bash
python $TOOLS/map_published.py \
    --target MOL_resp.mol2 \
    --template PUB.lib --frcmod frcmod.PUB \
    --merge sel.txt \
    --out sel_full.txt
```

**How it works.** The published template and your molecule are compared as
*graphs* (elements + bonds), and the largest common substructure is found.
Every matched atom inherits the template's atom type. Nothing depends on atom
numbering, naming or ordering, so the same command works on any structure of
the same fragment.

Matching (rather than exact isomorphism) is necessary because your model is
usually a **truncation** of the published molecule — you keep one ring, the
published library holds the entire cofactor.

**Where the transfer stops.** `--frcmod` restricts adoption to types the
parameter file actually *defines* (has a `MASS` entry for). Published files
often reference types they do not define — those belong to a protein force
field, not to the fragment — so this rule makes the transfer end exactly at
the parameterised region and leaves the junction atom on GAFF.

**Layering.** `--merge sel.txt` is applied **last and wins**. If a reactive
atom lies inside the published fragment, it takes the **published type as its
parent** (unless you wrote an explicit parent), so its parameters are cloned
from published values rather than from GAFF.

Repeat `--template`/`--frcmod` for several published sets.

**Read the printed mapping before continuing** — it is the review step:

```
PUB.lib: matched 29 atoms, adopted 16 published types
25 atoms re-typed: 16 published, 9 custom
```

### Reactive atoms only (no published set)

Skip this step and use your `sel.txt` directly as the selection file in
step 3.

---

## Step 2 — fill the GAFF gaps

```bash
sed 's/ DU / ho /' MOL_resp.mol2 > MOL_gaff.mol2
```

```bash
parmchk2 -i MOL_gaff.mol2 -f mol2 -o base.frcmod -s gaff2
```

**Why a separate copy.** `parmchk2` only understands GAFF types. It fails on
your custom types, on published types, and on `DU`. So it is run on an
all-GAFF copy of the *same* molecule; the parameters it finds are keyed by
GAFF type and apply to the re-typed system just as well. `DU` and `ho` are
both two characters, so column alignment is preserved. `MOL_gaff.mol2` is
scratch — nothing else reads it.

This step must come **before** step 3, which uses `base.frcmod` as a
parameter source.

---

## Step 3 — re-type and build the frcmod

```bash
python $TOOLS/clone_atom_types.py \
    --mol2 MOL_resp.mol2 --sel sel_full.txt --gaff $GAFF \
    --params frcmod.PUB --params base.frcmod \
    --out-mol2 MOL_ts.mol2 --out-frcmod MOL_ts.frcmod
```

Outputs `MOL_ts.mol2` (re-typed) and `MOL_ts.frcmod` (every `MASS`, `BOND`,
`ANGLE`, `DIHE` and `NONBON` term the new types touch), plus a
`MOL_ts.frcmod.review` sidecar.

**Parameter sources, in priority order:**

1. `frcmod.PUB` — published values, used **verbatim** (no geometry override)
2. `base.frcmod` — `parmchk2` gap-fills; needed for **junction** terms between
   the published fragment and the GAFF remainder, which `gaff2.dat` has no
   analog for
3. `gaff2.dat` — everything else, cloned via each atom's parent type

**How each term is decided:**

* has an analog, geometry close to it → clone force constant *and* equilibrium
* has an analog, geometry **far** from it (bond > 0.08 Å, angle > 12°) → the
  coordinate is *reactive*: keep the equilibrium from your TS geometry, clone
  the force constant as a **seed**, and log it
* no analog at all → equilibrium from geometry, force constant is a default
  **seed**, torsions become zero-barrier placeholders, and it is logged

The `.mol2` is rewritten by substituting **only** the atom-type field, so
coordinates, names and charges keep their exact original columns.

> **A `parmchk2` trap.** `parmchk2` writes zero-valued placeholders marked
> `ATTN, need revision` for terms it cannot determine — including anything
> touching your in-flight proton. Used as a cloning source these would
> silently zero a reactive force constant, so `clone_atom_types.py` skips any
> line carrying that marker.

---

## Step 4 — verify with tleap

```bash
printf 'source leaprc.gaff2\nloadamberparams base.frcmod\nloadamberparams frcmod.PUB\nloadamberparams MOL_ts.frcmod\nmol = loadmol2 MOL_ts.mol2\nsaveamberparm mol prmtop inpcrd\nquit\n' > tleap.in
```

```bash
tleap -f tleap.in
```

Load order matters — general GAFF2 first, then the gap-fills, then the
published set, then your own frcmod (later definitions win).

**Required result:**

```
Exiting LEaP: Errors = 0; Warnings = 0; Notes = 0.
```

If tleap reports a missing **improper**, add it to `MOL_ts.frcmod` by hand —
impropers are not generated automatically, because GAFF wild-cards most of
them and the rare exception is easier to add than to predict.

---

## Step 5 — check the result

Confirm the geometry and charges came through untouched:

```bash
awk '/@<TRIPOS>ATOM/{a=1;next}/@<TRIPOS>BOND/{a=0}a{n++;s+=$9}END{printf "atoms=%d  sum(q)=%.4f\n",n,s}' MOL_ts.mol2
```

`sum(q)` must equal your net charge.

Then read the review file:

```bash
cat MOL_ts.frcmod.review
```

```
BOND  AO-BH: reactive, req=1.2705 from geometry (GAFF 0.9725); k=535.51 is a seed
BOND  BH-OB: reactive, req=1.1459 from geometry (GAFF 0.9725); k=535.51 is a seed
ANGLE AC-AO-BH: reactive, theta=120.11 from geometry (GAFF 107.39); k=65.13 is a seed
ANGLE AO-BH-OB: NO GAFF analog; theta=171.84 from geometry, k=50.0 is a seed
DIHE  AC-AO-BH-OB: NO GAFF analog; zero-barrier placeholder
```

**This file is a checklist, not an error log.** It lists every parameter that
is a *seed* — a placeholder rather than a derived value. Two kinds appear:

* **`reactive`** — a GAFF analog exists, but your geometry is far from it.
  These are the forming/breaking coordinates: a partial bond stretched well
  past a normal one, an angle opened toward linearity. The equilibrium value
  is right (it comes from your QM geometry); the **force constant is not**.
* **`NO GAFF analog`** — the term does not exist in GAFF at all, because the
  arrangement is impossible in a ground state (a hydrogen bonded to two heavy
  atoms). Both the force constant and, for torsions, the barrier are
  placeholders.

**Everything listed here must be selected in the optimizer's parameter file.**
These force constants are exactly what the fit is supposed to determine; if
they are not selected, the placeholder values survive into the final force
field and the TSFF is wrong.

Equally important: **nothing unexpected should appear in this list.** Entries
should cluster around the reacting atoms. A seeded term somewhere else means
a parameter failed to find an analog — investigate before optimizing.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `atom N has parent 'DU'` | `DU` is a placeholder, not a type. Put a real GAFF parent in `sel.txt` (`ho` for a transferring hydroxyl proton). |
| `no match for template` | the template is not the same fragment, or your molecule is missing part of it. Check the printed match count. |
| adopted 0 published types | `--frcmod` defines none of the matched types. Confirm the frcmod has `MASS` entries for them. |
| tleap: dozens of errors | `base.frcmod` missing or not loaded. Run step 2 and list it in `tleap.in`. |
| tleap: `Unknown keyword` | an frcmod has extra text in its header. Only line 1 is a free-text title. |
| many seeded junction terms | pass `--params base.frcmod` in step 3 as well. |
| `type 'X' has no mass` | the parent type is not in `gaff2.dat` or in any `--params` file. Use a real parent. |
| all bonds show order 1 in a viewer | cosmetic only — AMBER ignores mol2 bond orders and reads parameters from atom types. `antechamber` gives up on bond perception when a TS geometry has an atom with impossible valence. |
