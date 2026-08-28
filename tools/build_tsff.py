#!/usr/bin/env python
"""
build_tsff.py
-------------
Build a transition-state force field from a RESP-charged GAFF/GAFF2 mol2.

    THI_resp.mol2  ->  THI.mol2  +  THI.frcmod     (one structure, one frcmod)

One command does the whole job: re-types the atoms you select, adopts published
types for any cofactor you point it at, fills every GAFF gap, and writes a
single self-contained frcmod that tleap can load on its own.

    python build_tsff.py THI_resp.mol2 --sel sel.txt \\
        --published NADH.lib frcmod.NADH -o THI

Why one frcmod
--------------
GAFF does not cover everything. Specialised subtypes such as `c5`/`c6` (sp3
carbons in 5- and 6-membered rings) have no wildcard torsions, so any sugar or
saturated ring in your model produces "No torsion terms for atom types" errors
in tleap -- dozens of them, which then cascade into empty Hessians and inf
scores in the optimiser. `parmchk2` fills those gaps, but that put the force
field in two files that BOTH had to be loaded, and forgetting one failed loudly
only much later. This tool runs parmchk2 itself and folds the result in, so the
gap problem never reaches you and there is only ever one file to carry.

What it does, in order
----------------------
1. matches each --published template against your structure (element +
   connectivity, so atom numbering is irrelevant) and adopts the published
   atom types it defines;
2. applies --sel on top, which always wins;
3. runs parmchk2 on an all-GAFF copy to obtain the gap-fills;
4. generates every bond/angle/dihedral term the new types touch, taking
   published values verbatim, cloning the rest, and reading reactive
   equilibrium values off your TS geometry;
5. writes ONE frcmod (gap-fills + published terms + generated terms) and the
   re-typed mol2, then verifies the pair with tleap.

Selection file (--sel)
----------------------
    # atom_index  new_type  parent_type
    39   AC   c3
    44   BH   ho          # required for a DU atom: it has no type to inherit

`parent_type` may be omitted unless the atom is typed DU; it is the existing
type whose parameters are cloned for the new one.

Published sets (--published TEMPLATE FRCMOD, repeatable)
--------------------------------------------------------
TEMPLATE is any .lib/.off/.mol2 carrying the published atom types; FRCMOD is
the matching parameter file. Only types the frcmod DEFINES (has a MASS entry
for) are adopted, so the transfer stops at the edge of the parameterised
fragment and leaves the junction on GAFF.
"""
from __future__ import print_function

import argparse
import glob
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict


# Equilibrium value is taken from geometry (not the GAFF analog) when the
# mol2 geometry deviates from the GAFF equilibrium by more than this much --
# i.e. the coordinate is stretched/bent, as at a transition state.
BOND_EQ_THRESH = 0.08     # Angstrom
ANGLE_EQ_THRESH = 12.0    # degrees

# Seeds used only when no GAFF analog exists at all.
DEFAULT_BOND_K = 300.0    # kcal/mol/A^2
DEFAULT_ANGLE_K = 50.0    # kcal/mol/rad^2
DEFAULT_DIHE = (1, 0.0, 0.0, 2.0)   # IDIVF, PK, phase, periodicity (zero barrier)


# ---------------------------------------------------------------------------
# mol2
# ---------------------------------------------------------------------------
def read_mol2(path):
    """Return (lines, atom_tokens, coords, types, bonds, atom_line_idx)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    section = None
    atom_tokens = {}      # idx -> list of raw tokens
    coords = {}           # idx -> (x, y, z)
    types = {}            # idx -> current atom type
    atom_line_idx = {}    # idx -> line number (for rewrite)
    bonds = []            # list of (a, b) 1-based
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("@<TRIPOS>"):
            section = s[9:].split()[0] if len(s) > 9 else None
            continue
        if not s:
            continue
        if section == "ATOM":
            tok = raw.split()
            idx = int(tok[0])
            atom_tokens[idx] = tok
            coords[idx] = (float(tok[2]), float(tok[3]), float(tok[4]))
            types[idx] = tok[5]
            atom_line_idx[idx] = i
        elif section == "BOND":
            tok = raw.split()
            bonds.append((int(tok[1]), int(tok[2])))
    return lines, atom_tokens, coords, types, bonds, atom_line_idx


_WS_RE = re.compile(r"\S+")


def set_mol2_type_inplace(raw, new_type):
    """Return the mol2 ATOM line `raw` with ONLY its atom-type field (the 6th
    whitespace-delimited token) replaced by `new_type`. Every other field keeps
    its exact original column position, so re-typed lines stay perfectly
    aligned with the untouched ones (we don't re-format the whole line)."""
    toks = list(_WS_RE.finditer(raw))
    if len(toks) < 6:
        return raw                          # not a well-formed atom line
    start = toks[5].start()
    next_start = toks[6].start() if len(toks) > 6 else len(raw)
    field_width = next_start - start        # old type token + its trailing spaces
    if len(new_type) < field_width:
        newfield = new_type.ljust(field_width)
    else:
        newfield = new_type + " "           # too long to fit the field: keep 1 space
    return raw[:start] + newfield + raw[next_start:]


def write_mol2(path, lines, atom_tokens, atom_line_idx, sel):
    out = list(lines)
    for idx, (new_type, _parent) in sel.items():
        li = atom_line_idx[idx]
        out[li] = set_mol2_type_inplace(out[li], new_type)
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# selection file
# ---------------------------------------------------------------------------
def read_selection(path, mol2_types):
    sel = OrderedDict()
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            cols = line.split()
            idx = int(cols[0])
            new_type = cols[1]
            parent = cols[2] if len(cols) > 2 else None
            if parent is None:
                cur = mol2_types.get(idx, "")
                if cur.upper() in ("DU", "DUMMY", ""):
                    sys.exit("ERROR: atom {} is typed '{}'; give an explicit "
                             "parent_type in the selection file "
                             "(e.g. '{} {} ho').".format(idx, cur, idx, new_type))
                parent = cur
            sel[idx] = (new_type, parent)
    return sel


# ---------------------------------------------------------------------------
# GAFF2 parameter file
# ---------------------------------------------------------------------------
def read_gaff(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    masses, bonds, angles, dihe, vdw = {}, {}, {}, {}, {}

    i = 1  # skip title
    # MASS block -> blank
    while i < len(lines) and lines[i].strip():
        tok = lines[i].split()
        masses[tok[0]] = float(tok[1])
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    # optional "hydrophilic atoms" line (no '-' in the type field)
    if i < len(lines) and (len(lines[i]) < 3 or lines[i][2] != "-"):
        i += 1
    # BOND block -> blank
    while i < len(lines) and lines[i].strip():
        L = lines[i]
        t1, t2 = L[0:2].strip(), L[3:5].strip()
        r = L[5:].split()
        bonds[(t1, t2)] = bonds[(t2, t1)] = (float(r[0]), float(r[1]))
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    # ANGLE block -> blank
    while i < len(lines) and lines[i].strip():
        L = lines[i]
        t1, t2, t3 = L[0:2].strip(), L[3:5].strip(), L[6:8].strip()
        r = L[8:].split()
        angles[(t1, t2, t3)] = angles[(t3, t2, t1)] = (float(r[0]), float(r[1]))
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    # DIHEDRAL block -> blank (multi-term: same key repeats, neg. periodicity)
    while i < len(lines) and lines[i].strip():
        L = lines[i]
        key = (L[0:2].strip(), L[3:5].strip(), L[6:8].strip(), L[9:11].strip())
        r = L[11:].split()
        dihe.setdefault(key, []).append(
            (int(r[0]), float(r[1]), float(r[2]), float(r[3])))
        i += 1

    # vdW: jump straight to the MOD4 section (robust to intervening sections)
    for j, L in enumerate(lines):
        if L.strip().startswith("MOD4"):
            k = j + 1
            while k < len(lines) and lines[k].strip() \
                    and not lines[k].strip().startswith("END"):
                tok = lines[k].split()
                if len(tok) >= 3:
                    vdw[tok[0]] = (float(tok[1]), float(tok[2]))
                k += 1
            break
    return masses, bonds, angles, dihe, vdw


def analog_keys(pairs):
    """Candidate lookup keys for a term.

    `pairs` gives (effective, parent) type for each atom of the term. An atom's
    EFFECTIVE type is its published type when it has one, else its GAFF parent.
    A term can straddle the two schemes -- e.g. a published CF bonded to a
    reactive atom whose parent is the published CH -- so every combination is
    tried, most-published first."""
    opts = []
    for eff, par in pairs:
        opts.append([eff] if par == eff else [eff, par])
    seen = []
    for combo in itertools.product(*opts):
        if combo not in seen:
            seen.append(combo)
    return seen


def find_analog(keys, gaff_dict, pub_dict):
    """First hit for any candidate key, GAFF preferred then published."""
    for k in keys:
        hit = gaff_dict.get(k) or pub_dict.get(k)
        if hit:
            return hit
    return None


def lookup_dihe(dihe, p):
    """Try exact then wild-carded (X-b-c-X ...) matches, both directions."""
    p1, p2, p3, p4 = p
    for key in (
        (p1, p2, p3, p4), (p4, p3, p2, p1),
        ("X", p2, p3, "X"), ("X", p3, p2, "X"),
        (p1, p2, p3, "X"), ("X", p2, p3, p4),
        (p4, p3, p2, "X"), ("X", p3, p2, p1),
    ):
        if key in dihe:
            return dihe[key]
    return None


def read_frcmod(path):
    """Parse a frcmod (MASS/BOND/ANGLE/DIHE/NONBON); frcmod-dialect headers OK.
    Returns dicts keyed by *type* tuples (published values, e.g. frcmod.NADH)."""
    HDR = {"MASS": "mass", "BOND": "bond", "ANGLE": "angle", "ANGL": "angle",
           "DIHE": "dihe", "DIHEDRAL": "dihe", "IMPROPER": "impro",
           "IMPRO": "impro", "IMPR": "impro", "NONBON": "nonb",
           "NONB": "nonb", "NONBONDED": "nonb"}
    masses, bonds, angles, vdw = {}, {}, {}, {}
    dihe = {}
    sec = None
    for raw in open(path):
        s = raw.rstrip("\n")
        st = s.strip()
        if not st or st.startswith("#"):
            continue
        # parmchk2 marks terms it could NOT determine with "ATTN, need revision"
        # and writes zeros for them. They are placeholders, not parameters, so
        # they must never be used as a cloning source.
        if "ATTN" in st and "need revision" in st:
            continue
        if st.upper() in HDR:
            sec = HDR[st.upper()]
            continue
        if sec == "mass":
            t = st.split()
            masses[t[0]] = float(t[1])
        elif sec == "bond":
            a, b = s[0:2].strip(), s[3:5].strip()
            r = s[5:].split()
            bonds[(a, b)] = bonds[(b, a)] = (float(r[0]), float(r[1]))
        elif sec == "angle":
            a, b, c = s[0:2].strip(), s[3:5].strip(), s[6:8].strip()
            r = s[8:].split()
            angles[(a, b, c)] = angles[(c, b, a)] = (float(r[0]), float(r[1]))
        elif sec == "dihe":
            k = (s[0:2].strip(), s[3:5].strip(), s[6:8].strip(), s[9:11].strip())
            r = s[11:].split()
            dihe.setdefault(k, []).append(
                (int(float(r[0])), float(r[1]), float(r[2]), float(r[3])))
        elif sec == "nonb":
            t = st.split()
            vdw[t[0]] = (float(t[1]), float(t[2]))
    return masses, bonds, angles, dihe, vdw


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def angle_deg(a, b, c):
    u = [a[i] - b[i] for i in range(3)]
    v = [c[i] - b[i] for i in range(3)]
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    cos = sum(u[i] * v[i] for i in range(3)) / (du * dv)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def mean(xs):
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# frcmod formatting (mirrors the run_10 template your tool already parses)
# ---------------------------------------------------------------------------
def t2(t):
    return "{:<2}".format(t)


def fmt_bond(t1, t2_, k, eq):
    return "{:<24}{:>10.4f}{:>12.4f}".format(t2(t1) + "-" + t2(t2_), k, eq)


def fmt_angle(t1, t2_, t3, k, eq):
    lbl = t2(t1) + "-" + t2(t2_) + "-" + t2(t3)
    return "{:<24}{:>10.4f}{:>12.3f}".format(lbl, k, eq)


def fmt_dihe(t1, t2_, t3, t4, idivf, pk, phase, per):
    lbl = t2(t1) + "-" + t2(t2_) + "-" + t2(t3) + "-" + t2(t4)
    return "{:<16}{:>2}{:>12.3f}{:>12.3f}{:>12.3f}".format(
        lbl, idivf, pk, phase, per)


def fmt_nonbon(t, rstar, eps):
    return "  {:<2}   {:>9.5f}   {:>9.5f}   0.00000".format(t, rstar, eps)

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


def read_mol2_graph(path):
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
        return read_mol2_graph(path)
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
# frcmod assembly
# ---------------------------------------------------------------------------
FRC_SECTIONS = ["MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"]
FRC_ALIASES = {"MASS": "MASS", "BOND": "BOND", "ANGLE": "ANGLE",
               "ANGL": "ANGLE", "DIHE": "DIHE", "DIHEDRAL": "DIHE",
               "IMPROPER": "IMPROPER", "IMPRO": "IMPROPER", "IMPR": "IMPROPER",
               "NONBON": "NONBON", "NONB": "NONBON", "NONBONDED": "NONBON"}
# width of the "a-b-c-d" type field, per section; None = key on the first token
KEY_WIDTH = {"MASS": None, "NONBON": None,
             "BOND": 5, "ANGLE": 8, "DIHE": 11, "IMPROPER": 11}


def frc_key(section, line):
    w = KEY_WIDTH[section]
    return line.split()[0] if w is None else line[:w]


def frc_types(section, line):
    """The atom types a frcmod line refers to."""
    if KEY_WIDTH[section] is None:
        return [line.split()[0]]
    return [t.strip() for t in frc_key(section, line).split("-")]


def frc_sections(text):
    """{section: OrderedDict(key -> [raw lines])}; the title line is dropped."""
    out = OrderedDict((s, OrderedDict()) for s in FRC_SECTIONS)
    section = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()
        if upper in FRC_ALIASES:
            section = FRC_ALIASES[upper]
            continue
        if section is None:
            continue
        out[section].setdefault(frc_key(section, line), []).append(line)
    return out


def frc_merge(target, source, keep_types=None):
    """Fold `source` sections into `target`; later definitions win.

    keep_types, when given, drops any entry naming a type the molecule does not
    actually contain -- that is what stops a published frcmod from dragging in
    parameters for the parts of the cofactor you did not model."""
    for section, entries in source.items():
        for key, lines in entries.items():
            if keep_types is not None:
                types = frc_types(section, lines[0])
                if not all(t in keep_types or t == "X" or t == "" for t in types):
                    continue
            target[section][key] = lines
    return target


def frc_write(path, title, sections):
    out = [title]
    for section in FRC_SECTIONS:
        out.append(section)
        for lines in sections[section].values():
            out.extend(lines)
        out.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return sum(len(v) for s in FRC_SECTIONS for v in sections[s].values())


# ---------------------------------------------------------------------------
# external programs
# ---------------------------------------------------------------------------
def amber_bin(name):
    home = os.environ.get("AMBERHOME", "")
    cand = os.path.join(home, "bin", name)
    if os.path.isfile(cand):
        return cand
    found = shutil.which(name) if hasattr(shutil, "which") else None
    if found:
        return found
    sys.exit("ERROR: cannot find '{}'. Set AMBERHOME or put it on PATH."
             .format(name))


def default_gaff():
    home = os.environ.get("AMBERHOME", "")
    cand = os.path.join(home, "dat", "leap", "parm", "gaff2.dat")
    return cand if os.path.isfile(cand) else None


def run_parmchk2(mol2_path, out_path, workdir):
    cmd = [amber_bin("parmchk2"), "-i", mol2_path, "-f", "mol2",
           "-o", out_path, "-s", "gaff2"]
    p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0 or not os.path.isfile(out_path):
        sys.exit("ERROR: parmchk2 failed:\n" + out)


def run_tleap(frcmod, mol2, leaprcs, workdir):
    """Returns (errors, warnings, log text)."""
    leap_in = os.path.join(workdir, "leap.in")
    with open(leap_in, "w") as fh:
        fh.write("source leaprc.gaff2\n")
        for extra in leaprcs:
            fh.write("source {}\n".format(extra))
        fh.write("loadamberparams {}\n".format(os.path.abspath(frcmod)))
        fh.write("mol = loadmol2 {}\n".format(os.path.abspath(mol2)))
        fh.write("saveamberparm mol {0}/prmtop {0}/inpcrd\nquit\n"
                 .format(os.path.abspath(workdir)))
    p = subprocess.Popen([amber_bin("tleap"), "-f", leap_in], cwd=workdir,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = p.communicate()[0].decode("utf-8", "replace")
    errors = warnings = None
    for line in log.splitlines():
        if "Exiting LEaP" in line:
            m = re.search(r"Errors\s*=\s*(\d+).*Warnings\s*=\s*(\d+)", line)
            if m:
                errors, warnings = int(m.group(1)), int(m.group(2))
    return errors, warnings, log


# ---------------------------------------------------------------------------
# selection file
# ---------------------------------------------------------------------------
def read_custom_selection(path):
    """{index: (new_type, parent_or_None)} -- parent may be filled in later."""
    out = OrderedDict()
    for line in open(path):
        body = line.split("#")[0].strip()
        if not body:
            continue
        cols = body.split()
        if len(cols) < 2:
            sys.exit("ERROR: bad line in {}: {!r}".format(path, line.rstrip()))
        out[int(cols[0])] = (cols[1], cols[2] if len(cols) > 2 else None)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a TSFF: re-typed mol2 + one self-contained frcmod.",
        epilog="example:\n"
               "  python build_tsff.py THI_resp.mol2 --sel sel.txt \\\n"
               "      --published NADH.lib frcmod.NADH -o THI\n")
    ap.add_argument("mol2", help="input mol2: GAFF/GAFF2 types, RESP charges, "
                                 "TS geometry")
    ap.add_argument("-o", "--out", required=True,
                    help="output prefix -> <out>.mol2 and <out>.frcmod")
    ap.add_argument("--sel", help="custom/reactive atom types (see --help)")
    ap.add_argument("--published", nargs=2, action="append", default=[],
                    metavar=("TEMPLATE", "FRCMOD"),
                    help="published set: structure carrying the types "
                         "(.lib/.off/.mol2) and its frcmod. Repeatable.")
    ap.add_argument("--gaff", default=None,
                    help="gaff2.dat (default: $AMBERHOME/dat/leap/parm/gaff2.dat)")
    ap.add_argument("--leaprc", action="append", default=[],
                    help="extra leaprc to source when verifying, e.g. "
                         "leaprc.protein.ff19SB. Repeatable.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the tleap check")
    opts = ap.parse_args(argv)

    gaff_path = opts.gaff or default_gaff()
    if not gaff_path or not os.path.isfile(gaff_path):
        sys.exit("ERROR: gaff2.dat not found; set AMBERHOME or pass --gaff.")

    out_mol2 = opts.out + ".mol2"
    out_frcmod = opts.out + ".frcmod"

    lines, atom_tokens, coords, types, bonds, atom_line_idx = read_mol2(opts.mol2)

    # ---- 1. published types, by structure matching -------------------------
    published, pub_frcmods = OrderedDict(), []
    if opts.published:
        G, Gadj = read_mol2_graph(opts.mol2)
        for template, frcmod_path in opts.published:
            for p in (template, frcmod_path):
                if not os.path.isfile(p):
                    sys.exit("ERROR: no such file: {}".format(p))
            T, Tadj = read_structure(template)
            avail = set(i for i in G if i not in published)
            heavy = match_heavy(T, Tadj, G, Gadj, avail)
            if not heavy:
                sys.stderr.write("WARNING: {} did not match the structure\n"
                                 .format(os.path.basename(template)))
                continue
            full = add_hydrogens(heavy, T, Tadj, G, Gadj)
            bad = check_bonds(full, Tadj, Gadj)
            if bad:
                sys.stderr.write("WARNING: {} bond(s) of {} not reproduced by "
                                 "the match\n".format(len(bad),
                                                      os.path.basename(template)))
            allowed = frcmod_defined_types([frcmod_path])
            kept = 0
            for t, g in full.items():
                ttype = T[t]["type"]
                if ttype in allowed and g not in published:
                    published[g] = ttype
                    kept += 1
            pub_frcmods.append(frcmod_path)
            sys.stderr.write("{:<20} matched {:>3} atoms, adopted {:>3} "
                             "published types\n"
                             .format(os.path.basename(template), len(full), kept))

    # ---- 2. custom selection on top (always wins) --------------------------
    custom = read_custom_selection(opts.sel) if opts.sel else OrderedDict()
    for idx in list(custom) + list(published):
        if idx not in atom_tokens:
            sys.exit("ERROR: atom index {} is not in {}".format(idx, opts.mol2))

    sel = OrderedDict()
    for idx in sorted(set(published) | set(custom)):
        if idx in custom:
            new_type, parent = custom[idx]
            if parent is None:
                # inside a published fragment -> clone from the published type
                parent = published.get(idx, types[idx])
        else:
            new_type, parent = published[idx], types[idx]
        if parent.upper() in ("DU", "DUMMY", ""):
            sys.exit("ERROR: atom {} ({}) has parent '{}', which is not a real "
                     "type.\n       Give an explicit parent in {} -- e.g. "
                     "'{} {} ho' for a transferring hydroxyl proton."
                     .format(idx, atom_tokens[idx][1], parent,
                             opts.sel or "the selection file", idx, new_type))
        sel[idx] = (new_type, parent)
    if not sel:
        sys.exit("ERROR: nothing to re-type; pass --sel and/or --published.")

    # A DU atom carries no real type, so parmchk2 cannot read it and nothing can
    # be cloned for it. Catch that here rather than let parmchk2 fail obscurely.
    orphan_du = [i for i in sorted(atom_tokens)
                 if types[i].upper() in ("DU", "DUMMY") and i not in sel]
    if orphan_du:
        sys.exit("ERROR: atom(s) {} are typed DU but are not selected.\n"
                 "       A DU atom has no parameters to inherit, so it must be "
                 "listed in --sel\n       with a real parent type -- e.g. "
                 "'{} XX ho' for a transferring hydroxyl proton."
                 .format(", ".join("{} ({})".format(i, atom_tokens[i][1])
                                   for i in orphan_du), orphan_du[0]))

    workdir = tempfile.mkdtemp(prefix="build_tsff_")
    try:
        # ---- 3. parmchk2 gap-fills, on an all-GAFF copy --------------------
        # parmchk2 only understands GAFF types, so it cannot read the re-typed
        # structure; and a DU atom has no type at all. Give it a copy where the
        # DU atoms carry their declared parent instead.
        gaff_lines = list(lines)
        for idx, (_nt, parent) in sel.items():
            if types[idx].upper() in ("DU", "DUMMY"):
                li = atom_line_idx[idx]
                gaff_lines[li] = set_mol2_type_inplace(gaff_lines[li], parent)
        gaff_mol2 = os.path.join(workdir, "allgaff.mol2")
        with open(gaff_mol2, "w") as fh:
            fh.write("\n".join(gaff_lines) + "\n")
        base_frcmod = os.path.join(workdir, "base.frcmod")
        run_parmchk2(gaff_mol2, base_frcmod, workdir)
        base_text = open(base_frcmod).read()

        # ---- 4. parameter sources ------------------------------------------
        masses, gbonds, gangles, gdihe, gvdw = read_gaff(gaff_path)
        ex_mass, ex_bond, ex_angle, ex_vdw, ex_dihe = {}, {}, {}, {}, {}
        for pf in list(pub_frcmods) + [base_frcmod]:
            m, b, a, d, v = read_frcmod(pf)
            ex_mass.update(m); ex_bond.update(b); ex_angle.update(a)
            ex_vdw.update(v); ex_dihe.update(d)

        # ---- 5. generate the terms the new types touch ---------------------
        # effective output type / parent type for every atom
        otype, ptype = {}, {}
        for idx in atom_tokens:
            if idx in sel:
                otype[idx], ptype[idx] = sel[idx][0], sel[idx][1]
            else:
                otype[idx] = ptype[idx] = types[idx]

        def touched(atoms):
            return any(a in sel for a in atoms)

        def eff(idx):
            """Effective type for parameter lookup: the published type when this
            atom has one (published types are those --params defines a MASS for),
            otherwise the GAFF parent type."""
            return otype[idx] if otype[idx] in ex_mass else ptype[idx]

        # adjacency
        adj = {idx: [] for idx in atom_tokens}
        for a, b in bonds:
            adj[a].append(b)
            adj[b].append(a)

        reviews = []   # human-readable notes about seeded / reactive terms

        # ---- BONDS -------------------------------------------------------------
        bgroups = OrderedDict()
        for a, b in bonds:
            if not touched((a, b)):
                continue
            okey = (otype[a], otype[b])
            ck = tuple(sorted(okey))
            rec = bgroups.setdefault(ck, {"okey": okey,
                                          "keys": analog_keys([(eff(a), ptype[a]),
                                                               (eff(b), ptype[b])]),
                                          "geoms": []})
            rec["geoms"].append(distance(coords[a], coords[b]))

        bond_lines = []
        for rec in bgroups.values():
            t1, t2_ = rec["okey"]
            geom = mean(rec["geoms"])
            pub = ex_bond.get((t1, t2_))
            if pub:                       # published (e.g. frcmod.NADH) wins, as-is
                bond_lines.append(fmt_bond(t1, t2_, pub[0], pub[1]))
                continue
            analog = find_analog(rec["keys"], gbonds, ex_bond)
            if analog:
                k, ceq = analog
                if abs(geom - ceq) > BOND_EQ_THRESH:
                    eq = geom
                    reviews.append("BOND {}-{}: reactive, req={:.4f} from geometry "
                                   "(GAFF {:.4f}); k={:.2f} is a seed"
                                   .format(t1, t2_, geom, ceq, k))
                else:
                    eq = ceq
            else:
                k, eq = DEFAULT_BOND_K, geom
                reviews.append("BOND {}-{}: NO GAFF analog; req={:.4f} from geometry, "
                               "k={:.1f} is a seed".format(t1, t2_, geom, k))
            bond_lines.append(fmt_bond(t1, t2_, k, eq))

        # ---- ANGLES ------------------------------------------------------------
        agroups = OrderedDict()
        for j in atom_tokens:
            nb = adj[j]
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    i, k_ = nb[x], nb[y]
                    if not touched((i, j, k_)):
                        continue
                    okey = (otype[i], otype[j], otype[k_])
                    ends = tuple(sorted((okey[0], okey[2])))
                    ck = (okey[1], ends)
                    rec = agroups.setdefault(ck, {"okey": okey,
                                                  "keys": analog_keys(
                                                      [(eff(i), ptype[i]),
                                                       (eff(j), ptype[j]),
                                                       (eff(k_), ptype[k_])]),
                                                  "geoms": []})
                    rec["geoms"].append(angle_deg(coords[i], coords[j], coords[k_]))

        angle_lines = []
        for rec in agroups.values():
            t1, t2_, t3 = rec["okey"]
            geom = mean(rec["geoms"])
            pub = ex_angle.get((t1, t2_, t3))
            if pub:
                angle_lines.append(fmt_angle(t1, t2_, t3, pub[0], pub[1]))
                continue
            analog = find_analog(rec["keys"], gangles, ex_angle)
            if analog:
                k, ceq = analog
                if abs(geom - ceq) > ANGLE_EQ_THRESH:
                    eq = geom
                    reviews.append("ANGLE {}-{}-{}: reactive, theta={:.2f} from "
                                   "geometry (GAFF {:.2f}); k={:.2f} is a seed"
                                   .format(t1, t2_, t3, geom, ceq, k))
                else:
                    eq = ceq
            else:
                k, eq = DEFAULT_ANGLE_K, geom
                reviews.append("ANGLE {}-{}-{}: NO GAFF analog; theta={:.2f} from "
                               "geometry, k={:.1f} is a seed"
                               .format(t1, t2_, t3, geom, k))
            angle_lines.append(fmt_angle(t1, t2_, t3, k, eq))

        # ---- DIHEDRALS ---------------------------------------------------------
        seen_d = set()
        dihe_lines = []
        for a, b in bonds:
            for i in adj[a]:
                if i == b:
                    continue
                for l in adj[b]:
                    if l == a or l == i:
                        continue
                    quad = (i, a, b, l)
                    if not touched(quad):
                        continue
                    okey = (otype[i], otype[a], otype[b], otype[l])
                    ck = min(okey, okey[::-1])
                    if ck in seen_d:
                        continue
                    seen_d.add(ck)
                    pub = lookup_dihe(ex_dihe, okey)
                    if pub is not None:   # published torsion (e.g. frcmod.NADH)
                        for (idivf, pk, phase, per) in pub:
                            dihe_lines.append(fmt_dihe(*okey, idivf=idivf, pk=pk,
                                                       phase=phase, per=per))
                        continue
                    terms = None
                    for key in analog_keys([(eff(i), ptype[i]), (eff(a), ptype[a]),
                                            (eff(b), ptype[b]), (eff(l), ptype[l])]):
                        terms = lookup_dihe(gdihe, key) or lookup_dihe(ex_dihe, key)
                        if terms is not None:
                            break
                    if terms is None:
                        idivf, pk, phase, per = DEFAULT_DIHE
                        dihe_lines.append(fmt_dihe(*okey, idivf=idivf, pk=pk,
                                                   phase=phase, per=per))
                        reviews.append("DIHE {}-{}-{}-{}: NO GAFF analog; zero-barrier "
                                       "placeholder".format(*okey))
                    else:
                        for (idivf, pk, phase, per) in terms:
                            dihe_lines.append(fmt_dihe(*okey, idivf=idivf, pk=pk,
                                                       phase=phase, per=per))

        # ---- MASS / NONBON for the new types -----------------------------------
        newtypes = OrderedDict()
        for idx, (nt, parent) in sel.items():
            newtypes.setdefault(nt, parent)
        mass_lines, nonbon_lines = [], []
        for nt, parent in newtypes.items():
            # published type -> its own mass; else parent's, from GAFF or --params
            m = ex_mass.get(nt)
            if m is None:
                m = masses.get(parent, ex_mass.get(parent))
            if m is None:
                sys.exit("ERROR: type '{}' has no mass (parent '{}' not in GAFF "
                         "or --params).".format(nt, parent))
            mass_lines.append("{} {}".format(nt, m))
            rstar_eps = ex_vdw.get(nt)
            if rstar_eps is None:
                rstar_eps = gvdw.get(parent, ex_vdw.get(parent))
            if rstar_eps is None:
                sys.exit("ERROR: type '{}' has no vdW (parent '{}' not in GAFF "
                         "MOD4 or --params).".format(nt, parent))
            nonbon_lines.append(fmt_nonbon(nt, rstar_eps[0], rstar_eps[1]))

        # ---- 6. one frcmod: gap-fills + published + generated --------------
        present = set(otype.values())
        sections = frc_sections(base_text)
        for pf in pub_frcmods:
            frc_merge(sections, frc_sections(open(pf).read()), keep_types=present)
        generated = OrderedDict((s, OrderedDict()) for s in FRC_SECTIONS)
        for section, produced in (("MASS", mass_lines), ("BOND", bond_lines),
                                  ("ANGLE", angle_lines), ("DIHE", dihe_lines),
                                  ("NONBON", nonbon_lines)):
            for line in produced:
                generated[section].setdefault(frc_key(section, line),
                                              []).append(line)
        frc_merge(sections, generated)

        title = "TSFF (build_tsff.py) | new types: " + ", ".join(
            "{}<-{}".format(nt, p) for nt, p in newtypes.items())
        total = frc_write(out_frcmod, title, sections)

        # ---- 7. the re-typed structure -------------------------------------
        write_mol2(out_mol2, lines, atom_tokens, atom_line_idx, sel)

        if reviews:
            with open(out_frcmod + ".review", "w") as fh:
                fh.write("Seeded / reactive terms in {}\n".format(out_frcmod))
                fh.write("(equilibrium values come from the TS geometry; the "
                         "force constants are seeds for the optimiser)\n\n")
                for r in reviews:
                    fh.write("  " + r + "\n")

        # ---- 8. verify ------------------------------------------------------
        sys.stderr.write("\nwrote {} ({} atoms re-typed)\n".format(
            out_mol2, len(sel)))
        sys.stderr.write("wrote {} ({} parameters, single self-contained file)\n"
                         .format(out_frcmod, total))
        if reviews:
            sys.stderr.write("wrote {}.review ({} seeded term(s) for the "
                             "optimiser to fit)\n".format(out_frcmod, len(reviews)))

        if not opts.no_verify:
            errors, warnings, log = run_tleap(out_frcmod, out_mol2,
                                              opts.leaprc, workdir)
            if errors is None:
                sys.stderr.write("\ntleap did not report a summary; see below\n")
                sys.stderr.write(log[-2000:])
                return 1
            sys.stderr.write("\ntleap: Errors = {}; Warnings = {}\n"
                             .format(errors, warnings))
            if errors:
                for line in log.splitlines():
                    if "No " in line or "Could not" in line:
                        sys.stderr.write("  " + line.strip() + "\n")
                return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
