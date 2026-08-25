#!/usr/bin/env python
"""
map_published.py
----------------
Transfer PUBLISHED atom types onto a target mol2 by *structure matching*, then
layer custom (reactive) types on top -- producing one selection file for
clone_atom_types.py.

This is the general form of "use molecule X's published atom types and
parameters": nothing is specific to NADH. Whatever published set you have --
as long as it ships a structure file carrying the types (.lib/.off/.mol2) and
a frcmod with the parameters -- can be mapped onto your molecule.

Why matching (and not a hand-written index list)
------------------------------------------------
The published template usually is not the same molecule as yours: your model
is typically a TRUNCATION of it (you keep the nicotinamide, NADH.lib holds the
whole dinucleotide). So the tool finds the maximum common substructure between
the template graph and your molecule graph -- element + connectivity -- and
copies the type of every matched template atom onto the atom it matched.
Atom ordering, naming and numbering are irrelevant.

Where the fragment stops
------------------------
With --frcmod, only types the published parameter file actually DEFINES (i.e.
that have a MASS entry) are adopted. That is what stops the transfer at the
edge of the parameterised fragment: frcmod.NADH mentions `CT` in a bond, but
does not define it, so the junction atom keeps its GAFF type and
clone_atom_types.py builds the junction terms from GAFF parents.

Layering with the custom/reactive selection
-------------------------------------------
--merge sel.txt is applied LAST and WINS. If a reactive atom lies inside the
published fragment, its parent type becomes the PUBLISHED type (unless you
wrote an explicit parent in sel.txt), so its parameters are cloned from the
published values rather than from GAFF.

Usage
-----
    # 1) published types only
    python map_published.py --target THI_resp.mol2 \\
        --template NADH.lib --frcmod frcmod.NADH --out sel_pub.txt

    # 2) published types + reactive overrides, in one selection file
    python map_published.py --target THI_resp.mol2 \\
        --template NADH.lib --frcmod frcmod.NADH \\
        --merge sel.txt --out sel_full.txt

    # several published sets at once
    python map_published.py --target X.mol2 \\
        --template NADH.lib --frcmod frcmod.NADH \\
        --template HEM.lib  --frcmod frcmod.hem  --out sel.txt

Then:
    python clone_atom_types.py --mol2 THI_resp.mol2 --sel sel_full.txt \\
        --gaff gaff2.dat --params frcmod.NADH \\
        --out-mol2 THI_ts.mol2 --out-frcmod THI_ts.frcmod
"""
from __future__ import print_function

import argparse
import os
import sys
from collections import OrderedDict

ATOMIC_NUMBER = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
                 17: "Cl", 35: "Br", 53: "I"}
ELEMENTS = ("Cl", "Br", "Si", "C", "N", "O", "S", "P", "H", "F", "I")


# ---------------------------------------------------------------------------
# structure readers
# ---------------------------------------------------------------------------
def element_of(name, atom_type):
    """Element from the atom type's first letter, falling back to the name."""
    t = (atom_type or "").strip()
    if t and t[0].islower():                 # gaff style: c3 n3 oh ss h1
        return t[0].upper()
    for e in ELEMENTS:                       # names: C1 N2 O10 H43 S1
        if name.upper().startswith(e.upper()):
            return e
    return name[0].upper() if name else "X"


def read_mol2(path):
    atoms, bonds, sec = OrderedDict(), [], None
    for line in open(path):
        s = line.strip()
        if s.startswith("@<TRIPOS>"):
            sec = s[9:].split()[0] if len(s) > 9 else None
            continue
        if not s:
            continue
        if sec == "ATOM":
            t = line.split()
            i = int(t[0])
            atoms[i] = {"name": t[1], "type": t[5],
                        "elem": element_of(t[1], t[5])}
        elif sec == "BOND":
            t = line.split()
            bonds.append((int(t[1]), int(t[2])))
    return atoms, adjacency(atoms, bonds)


def read_off(path):
    """AMBER OFF / .lib library (the format tleap's saveOff writes)."""
    lines = open(path).read().splitlines()
    atoms, bonds = OrderedDict(), []
    i, n = 0, len(lines)
    while i < n:
        head = lines[i]
        if head.startswith("!entry.") and ".unit.atoms table" in head:
            i += 1
            k = 1
            while i < n and not lines[i].startswith("!"):
                t = lines[i].split()
                if len(t) >= 7:
                    name = t[0].strip('"')
                    typ = t[1].strip('"')
                    elem = ATOMIC_NUMBER.get(int(t[6]),
                                             element_of(name, typ))
                    atoms[k] = {"name": name, "type": typ, "elem": elem}
                    k += 1
                i += 1
            continue
        if head.startswith("!entry.") and ".unit.connectivity table" in head:
            i += 1
            while i < n and not lines[i].startswith("!"):
                t = lines[i].split()
                if len(t) >= 2:
                    bonds.append((int(t[0]), int(t[1])))
                i += 1
            continue
        i += 1
    if not atoms:
        sys.exit("ERROR: no unit.atoms table found in {}".format(path))
    return atoms, adjacency(atoms, bonds)


def read_structure(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mol2":
        return read_mol2(path)
    if ext in (".lib", ".off"):
        return read_off(path)
    sys.exit("ERROR: unsupported template '{}' (use .mol2, .lib or .off)"
             .format(path))


def adjacency(atoms, bonds):
    adj = dict((i, []) for i in atoms)
    for a, b in bonds:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def frcmod_defined_types(paths):
    """Types the published parameter file DEFINES (has a MASS entry for)."""
    defined = set()
    for p in paths:
        sec = None
        for raw in open(p):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up in ("MASS", "BOND", "ANGLE", "ANGL", "DIHE", "IMPROPER",
                      "IMPRO", "IMPR", "NONBON", "NONB", "NONBONDED"):
                sec = up
                continue
            if sec == "MASS":
                defined.add(s.split()[0])
    return defined


# ---------------------------------------------------------------------------
# graph matching
# ---------------------------------------------------------------------------
def signature(i, atoms, adj, radius):
    """Canonical description of an atom's neighbourhood out to `radius` bonds."""
    shells = [atoms[i]["elem"]]
    seen, frontier = set([i]), [i]
    for _ in range(radius):
        nxt, labels = [], []
        for x in frontier:
            for y in adj[x]:
                if y in seen:
                    continue
                seen.add(y)
                nxt.append(y)
                labels.append(atoms[y]["elem"])
        shells.append("".join(sorted(labels)))
        frontier = nxt
    return "|".join(shells)


def similarity(sa, sb):
    """How many leading shells two signatures share."""
    a, b = sa.split("|"), sb.split("|")
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def grow(seed_t, seed_g, T, Tadj, G, Gadj, tsig, gsig, avail):
    """Expand a seed correspondence outwards, keeping bonds consistent."""
    mt, mg = {seed_t: seed_g}, {seed_g: seed_t}
    queue = [(seed_t, seed_g)]
    while queue:
        t, g = queue.pop(0)
        tn = [x for x in Tadj[t] if x not in mt and T[x]["elem"] != "H"]
        gn = [y for y in Gadj[g]
              if y not in mg and G[y]["elem"] != "H" and y in avail]
        cand = []
        for x in tn:
            for y in gn:
                if T[x]["elem"] != G[y]["elem"]:
                    continue
                cand.append((-similarity(tsig[x], gsig[y]), x, y))
        cand.sort()
        usedx, usedy = set(), set()
        for _score, x, y in cand:
            if x in usedx or y in usedy:
                continue
            # every already-mapped neighbour of x must be a neighbour of y
            ok = True
            for nx in Tadj[x]:
                if nx in mt and mt[nx] not in Gadj[y]:
                    ok = False
                    break
            if not ok:
                continue
            mt[x] = y
            mg[y] = x
            usedx.add(x)
            usedy.add(y)
            queue.append((x, y))
    return mt


def match_heavy(T, Tadj, G, Gadj, avail, radius=3, max_seeds=400):
    """Largest connected common substructure over heavy atoms."""
    tsig = dict((i, signature(i, T, Tadj, radius)) for i in T)
    gsig = dict((i, signature(i, G, Gadj, radius)) for i in G)
    best = {}
    for r in range(radius, 0, -1):
        seeds = []
        for t in T:
            if T[t]["elem"] == "H":
                continue
            st = "|".join(tsig[t].split("|")[:r + 1])
            for g in G:
                if g not in avail or G[g]["elem"] == "H":
                    continue
                if T[t]["elem"] != G[g]["elem"]:
                    continue
                if "|".join(gsig[g].split("|")[:r + 1]) == st:
                    seeds.append((t, g))
        for t, g in seeds[:max_seeds]:
            m = grow(t, g, T, Tadj, G, Gadj, tsig, gsig, avail)
            if len(m) > len(best):
                best = m
        if best:
            break
    return best


def add_hydrogens(mapping, T, Tadj, G, Gadj):
    """Map hydrogens through their (already matched) heavy atoms."""
    out = dict(mapping)
    used = set(mapping.values())
    for t, g in mapping.items():
        th = [x for x in Tadj[t] if T[x]["elem"] == "H" and x not in out]
        gh = [y for y in Gadj[g] if G[y]["elem"] == "H" and y not in used]
        for x, y in zip(th, gh):
            out[x] = y
            used.add(y)
    return out


def check_bonds(mapping, Tadj, Gadj):
    """Template bonds that did not survive the mapping (should be empty)."""
    bad = []
    for t, g in mapping.items():
        for nt in Tadj[t]:
            if nt in mapping and mapping[nt] not in Gadj[g]:
                bad.append((t, nt))
    return bad


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map published atom types onto a mol2 by structure "
                    "matching; optionally layer custom types on top.")
    ap.add_argument("--target", required=True, help="your all-GAFF mol2")
    ap.add_argument("--template", action="append", required=True,
                    help="published structure carrying the types "
                         "(.lib/.off/.mol2); repeatable")
    ap.add_argument("--frcmod", action="append", default=[],
                    help="published frcmod; restricts the transfer to types it "
                         "DEFINES (MASS entries). Repeatable.")
    ap.add_argument("--merge", help="custom selection file (reactive atoms); "
                                    "applied last and overrides published types")
    ap.add_argument("--out", required=True, help="selection file to write")
    ap.add_argument("--min-match", type=int, default=4,
                    help="fail if a template matches fewer atoms than this")
    opts = ap.parse_args(argv)

    G, Gadj = read_mol2(opts.target)
    allowed = frcmod_defined_types(opts.frcmod) if opts.frcmod else None

    published = OrderedDict()     # target idx -> published type
    origin = {}                   # target idx -> template file
    avail = set(i for i in G)

    for tpl in opts.template:
        T, Tadj = read_structure(tpl)
        heavy = match_heavy(T, Tadj, G, Gadj, avail)
        if not heavy:
            sys.stderr.write("WARNING: no match for template {}\n".format(tpl))
            continue
        full = add_hydrogens(heavy, T, Tadj, G, Gadj)
        bad = check_bonds(full, Tadj, Gadj)
        if bad:
            sys.stderr.write("WARNING: {} template bond(s) not reproduced in "
                             "the match for {}\n".format(len(bad), tpl))

        kept = 0
        for t, g in full.items():
            ttype = T[t]["type"]
            if allowed is not None and ttype not in allowed:
                continue          # type not defined by the frcmod -> leave GAFF
            published[g] = ttype
            origin[g] = os.path.basename(tpl)
            avail.discard(g)
            kept += 1
        sys.stderr.write("{}: matched {} atoms, adopted {} published types\n"
                         .format(os.path.basename(tpl), len(full), kept))
        if kept < opts.min_match:
            sys.exit("ERROR: template {} adopted only {} types (< --min-match "
                     "{}). Check that it really is the same fragment."
                     .format(tpl, kept, opts.min_match))

    # ---- layer the custom / reactive selection on top ----------------------
    custom = OrderedDict()        # idx -> (type, explicit parent or None)
    if opts.merge:
        for line in open(opts.merge):
            body = line.split("#")[0].strip()
            if not body:
                continue
            c = body.split()
            custom[int(c[0])] = (c[1], c[2] if len(c) > 2 else None)

    rows, overrides = [], []
    for idx in sorted(set(published) | set(custom)):
        if idx in custom:
            newtype, parent = custom[idx]
            if parent is None:
                # inside a published fragment -> clone from the PUBLISHED type
                parent = published.get(idx, G[idx]["type"])
            if idx in published:
                overrides.append((idx, published[idx], newtype, parent))
            note = "custom"
        else:
            newtype, parent = published[idx], G[idx]["type"]
            note = "published " + origin[idx]
        if parent.upper() in ("DU", "DUMMY", ""):
            sys.exit("ERROR: atom {} ({}) has parent '{}'; give an explicit "
                     "parent type in {}.".format(idx, G[idx]["name"], parent,
                                                 opts.merge))
        rows.append("{:<5}{:<5}{:<5}# {:<5} {}".format(
            idx, newtype, parent, G[idx]["name"], note))

    with open(opts.out, "w") as fh:
        fh.write("# atom_index  new_type  parent_type   "
                 "(map_published.py)\n")
        fh.write("\n".join(rows) + "\n")

    # ---- report ------------------------------------------------------------
    sys.stderr.write("\n{} atoms re-typed: {} published, {} custom\n".format(
        len(rows), len(published) - len(overrides), len(custom)))
    for idx, was, now, parent in overrides:
        sys.stderr.write("  override: atom {} ({}) published {} -> custom {} "
                         "(parent {})\n".format(idx, G[idx]["name"], was, now,
                                                parent))
    sys.stderr.write("wrote {}\n".format(opts.out))


if __name__ == "__main__":
    main()
