#!/usr/bin/env python
"""
clone_atom_types.py
-------------------
Re-type a handful of atoms in a GAFF/GAFF2 mol2 into unique *custom* atom
types and emit an AMBER frcmod that supplies every force-field term those
new types touch -- so tleap can build the system even though GAFF knows
nothing about the new types.

This is the "change atom types" step of the Q2MM-AMBER TSFF workflow. It
does NOT try to be the final force field: the reactive (forming/breaking)
force constants it writes are only *seeds*; the swarm optimizer refits
them. Its job is to produce a tleap-clean frcmod + retyped mol2.

How parameters are chosen
-------------------------
For every bond / angle / dihedral that involves at least one re-typed atom
the tool maps each atom back to its *parent* GAFF type, looks the term up
in the GAFF2 parameter file, and re-emits it under the new type names:

  * term HAS a GAFF analog   -> clone the force constant; use the GAFF
                                equilibrium value, UNLESS the geometry in
                                the mol2 deviates from it beyond a
                                threshold (a stretched/reactive coordinate),
                                in which case the equilibrium value is taken
                                from the mol2 geometry and the line is flagged.
  * term has NO GAFF analog  -> equilibrium value from the mol2 geometry,
                                force constant is a flagged seed.

MASS and vdW (NONBON) for each new type are cloned from its parent type.
Impropers are NOT generated (GAFF impropers are usually wild-carded on the
peripheral atoms, so they resolve without custom types; run tleap and add
the rare missing one by hand -- see step 3 of the workflow).

Selection file
--------------
One atom per line (mol2 atom number, 1-based):

    # atom_index   new_type   parent_type
    43   OA   oh      # donor O
    44   HT   ho      # transferring H  (was the DU dummy)
    66   OB   oh      # acceptor O

`parent_type` may be omitted if the atom already carries a real GAFF type
(it is then read from the mol2); it is REQUIRED for atoms typed `DU`.

Usage
-----
    python clone_atom_types.py \\
        --mol2 THI_resp.mol2 --sel sel.txt \\
        --gaff /path/to/gaff2.dat \\
        --out-mol2 thio-ts.mol2 --out-frcmod thio-ts.frcmod

Then verify with tleap (source leaprc.gaff2 / loadamberparams / loadmol2 /
saveamberparm) and add any residual terms tleap reports.
"""
from __future__ import print_function

import argparse
import itertools
import math
import re
import sys
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-type selected atoms and clone their FF terms into a frcmod.")
    ap.add_argument("--mol2", required=True, help="input GAFF/GAFF2 mol2")
    ap.add_argument("--sel", required=True, help="selection file")
    ap.add_argument("--gaff", required=True, help="gaff2.dat parameter file")
    ap.add_argument("--params", action="append", default=[],
                    help="extra frcmod parameter source(s), checked before gaff2 "
                         "by OUTPUT type name (e.g. --params frcmod.NADH). "
                         "Published values are used as-is (no geometry override).")
    ap.add_argument("--out-mol2", required=True, help="output re-typed mol2")
    ap.add_argument("--out-frcmod", required=True, help="output frcmod")
    ap.add_argument("--title", default="TSFF custom atom types (clone_atom_types.py)")
    opts = ap.parse_args(argv)

    lines, atom_tokens, coords, types, bonds, atom_line_idx = read_mol2(opts.mol2)
    sel = read_selection(opts.sel, types)
    masses, gbonds, gangles, gdihe, gvdw = read_gaff(opts.gaff)

    # extra published parameter sources (e.g. frcmod.NADH), keyed by output type
    ex_mass, ex_bond, ex_angle, ex_vdw, ex_dihe = {}, {}, {}, {}, {}
    for pf in opts.params:
        m, b, a, d, v = read_frcmod(pf)
        ex_mass.update(m); ex_bond.update(b); ex_angle.update(a)
        ex_vdw.update(v); ex_dihe.update(d)

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

    # ---- write frcmod ------------------------------------------------------
    # A frcmod's line 1 is a free-text TITLE; any other non-section line makes
    # tleap emit "Unknown keyword" warnings, so keep the header to one line and
    # put the review notes in a sidecar .review file instead.
    out = []
    out.append(opts.title + " | new types: " + ", ".join(
        "{}<-{}".format(nt, p) for nt, p in newtypes.items()))
    out.append("MASS")
    out.extend(mass_lines)
    out.append("")
    out.append("BOND")
    out.extend(bond_lines)
    out.append("")
    out.append("ANGLE")
    out.extend(angle_lines)
    out.append("")
    out.append("DIHE")
    out.extend(dihe_lines)
    out.append("")
    out.append("IMPRO")
    out.append("")
    out.append("NONBON")
    out.extend(nonbon_lines)
    with open(opts.out_frcmod, "w") as fh:
        fh.write("\n".join(out) + "\n")

    if reviews:
        with open(opts.out_frcmod + ".review", "w") as fh:
            fh.write("Seeded / reactive / no-analog terms in {}\n".format(opts.out_frcmod))
            fh.write("(equilibrium values from geometry; force constants are seeds "
                     "the swarm refits)\n\n")
            for r in reviews:
                fh.write("  " + r + "\n")

    # ---- write mol2 --------------------------------------------------------
    write_mol2(opts.out_mol2, lines, atom_tokens, atom_line_idx, sel)

    # ---- summary -----------------------------------------------------------
    sys.stderr.write(
        "wrote {} ({} MASS, {} BOND, {} ANGLE, {} DIHE, {} NONBON)\n".format(
            opts.out_frcmod, len(mass_lines), len(bond_lines),
            len(angle_lines), len(dihe_lines), len(nonbon_lines)))
    sys.stderr.write("wrote {} ({} atoms re-typed)\n".format(
        opts.out_mol2, len(sel)))
    if reviews:
        sys.stderr.write("\nReview these seeded / reactive terms:\n")
        for r in reviews:
            sys.stderr.write("  " + r + "\n")
    sys.stderr.write(
        "\nNext: verify with tleap (source leaprc.gaff2 / loadamberparams {} / "
        "loadmol2 {} / saveamberparm). Add any term tleap still reports "
        "(rare impropers).\n".format(opts.out_frcmod, opts.out_mol2))


if __name__ == "__main__":
    main()
