#!/usr/bin/env python
"""Build a source-resolved hybrid transition-state force field.

Each atom has a final type plus explicit aliases into a published parameter set,
ff19SB, and GAFF2 namespaces. A term is accepted only when one source can
describe every atom in that interaction. Missing standard terms are errors.
For an initial Q2MM force field, an explicit --allow-ts-seeds option permits a
narrow, reported fallback for unsupported custom TS angles and torsions, and
--ts-equilibria takes reaction-center bond and angle equilibrium values from
the TS geometry while keeping their source force constants.
"""
from __future__ import print_function


import argparse
import csv
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict

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


def write_mol2_without_bonds(source, destination, omitted_bonds):
    """Write a LEaP-loading copy with selected bonds omitted.

    LEaP rejects a second carbon bond while reading a two-coordinate hydride.
    The omitted TS bond is restored after loading while the H atom's temporary
    perturbation flag relaxes that coordination check.
    """
    omitted = set(frozenset(pair) for pair in omitted_bonds)
    lines = open(source).read().splitlines()
    output, section, removed, kept_bonds = [], None, 0, 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@<TRIPOS>"):
            section = stripped[9:].split()[0]
            output.append(line)
            continue
        if section == "BOND" and stripped:
            tokens = line.split()
            pair = frozenset((int(tokens[1]), int(tokens[2])))
            if pair in omitted:
                removed += 1
                continue
            kept_bonds += 1
            tokens[0] = str(kept_bonds)
            output.append("{:>6} {:>5} {:>5} {}".format(
                tokens[0], tokens[1], tokens[2], " ".join(tokens[3:])))
            continue
        output.append(line)
    if removed != len(omitted):
        raise ValueError("could not omit all LEaP-restored TS bonds from {}"
                         .format(source))
    molecule = output.index("@<TRIPOS>MOLECULE")
    count_line = next(i for i in range(molecule + 2, len(output))
                      if output[i].strip())
    counts = output[count_line].split()
    counts[1] = str(int(counts[1]) - removed)
    output[count_line] = "{:>5} {:>5} {}".format(
        counts[0], counts[1], " ".join(counts[2:]))
    with open(destination, "w") as fh:
        fh.write("\n".join(output) + "\n")


def angle_degrees(a, b, c):
    """Return the a-b-c angle in degrees for three Cartesian coordinates."""
    ba = tuple(x - y for x, y in zip(a, b))
    bc = tuple(x - y for x, y in zip(c, b))
    nba = math.sqrt(sum(x * x for x in ba))
    nbc = math.sqrt(sum(x * x for x in bc))
    if not nba or not nbc:
        raise ValueError("cannot seed an angle containing coincident atoms")
    cosine = sum(x * y for x, y in zip(ba, bc)) / (nba * nbc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def bond_length(a, b):
    """Return the a-b distance for two Cartesian coordinates."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def measured_equilibrium(section, instances, coords):
    """Return (mean, min, max) of a BOND length or ANGLE over its instances.

    Every instance of one final-type key shares a single equilibrium value,
    so a key that occurs more than once takes the mean, as Q2MM does for
    reference values.
    """
    measure = bond_length if section == "BOND" else angle_degrees
    values = [measure(*[coords[i] for i in ids]) for ids in instances]
    return sum(values) / len(values), min(values), max(values)


# ---------------------------------------------------------------------------
# selection file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GAFF2 parameter file
# ---------------------------------------------------------------------------










def read_amber_params(path):
    """Read a parm.dat or frcmod as text-preserving frc_sections records.

    In parm.dat, atom types listed after the improper/H-bond blocks share the
    first type's Lennard-Jones parameters. Expand those equivalences so custom
    parents such as NA, CC, CR and CW have explicit NONBON lookup records.
    """
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines()
    first = next((line.strip() for line in lines[1:]
                  if line.strip() and not line.lstrip().startswith("#")), "")
    if first.upper() in FRC_ALIASES:
        return frc_sections(text)

    sections = OrderedDict((s, OrderedDict()) for s in FRC_SECTIONS)

    def add(section, line):
        key, numbers, comment, raw = frc_parse(section, line)
        # parm*.dat and gaff*.dat append numeric provenance fields after the
        # physical values (for example a year and reference IDs). frc_parse
        # cannot recognize that boundary from token syntax alone. Move those
        # fields into the comment so a record cloned under a new final type is
        # rendered with the requested key instead of falling back to `raw`.
        physical_columns = {"MASS": 2, "BOND": 2, "ANGLE": 2,
                            "DIHE": 4, "IMPROPER": 3, "NONBON": 2}
        count = physical_columns[section]
        if len(numbers) > count:
            suffix = " ".join(numbers[count:])
            comment = suffix + ((" " + comment) if comment else "")
            numbers = numbers[:count]
        sections[section].setdefault(key, []).append((numbers, comment, raw))

    mod4 = next((i for i, line in enumerate(lines)
                 if line.strip().startswith("MOD4")), None)
    if mod4 is None or lines[mod4].split()[1:] != ["RE"]:
        raise ValueError("{}: expected MOD4 RE Lennard-Jones parameters".format(path))
    blocks, block = [], []
    for line in lines[1:mod4]:
        if line.strip():
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    if len(blocks) < 4:
        raise ValueError("{}: incomplete Amber parameter blocks".format(path))
    for line in blocks.pop(0):
        add("MASS", line)
    # A hydrophilic atom list precedes the bond block; some files put a blank
    # line after it, others place it directly before the first bond record.
    if len(blocks[0][0]) < 3 or blocks[0][0][2] != "-":
        blocks[0].pop(0)
        if not blocks[0]:
            blocks.pop(0)
    for section in ("BOND", "ANGLE", "DIHE"):
        for line in blocks.pop(0):
            add(section, line)
    if blocks and len(blocks[0][0]) > 8 and blocks[0][0][8] == "-":
        for line in blocks.pop(0):
            add("IMPROPER", line)

    equivalent = {}
    for block in blocks:
        for line in block:
            words = line.split()
            if all(t in sections["MASS"] for t in words):
                equivalent.update((t, words[0]) for t in words[1:])
            else:
                # Legacy 10-12 H-bond records occur before LJ equivalences.
                # Zero rows have no effect; nonzero rows cannot be represented
                # by this tool's NONBON section and must not disappear silently.
                try:
                    zero_hbond = len(words) >= 4 and all(
                        float(n) == 0.0 for n in words[2:4])
                except ValueError:
                    zero_hbond = False
                if not zero_hbond:
                    raise ValueError("{}: unsupported parameter row: {}".format(path, line))

    i = mod4 + 1
    while i < len(lines) and lines[i].strip() and lines[i].strip() != "END":
        add("NONBON", lines[i])
        i += 1
    for atom_type, parent in equivalent.items():
        if atom_type not in sections["NONBON"]:
            records = sections["NONBON"][parent]
            sections["NONBON"][atom_type] = [
                (list(numbers), comment,
                 frc_render("NONBON", atom_type, numbers, comment, raw))
                for numbers, comment, raw in records]
    # Preserve any extra parameter sections before END (e.g. LJEDIT).
    tail = []
    for line in lines[i:]:
        if line.strip() == "END":
            break
        tail.append(line)
    _, extras = frc_sections("Amber extras\n" + "\n".join(tail))
    return sections, extras


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# frcmod formatting (mirrors the run_10 template your tool already parses)
# ---------------------------------------------------------------------------
def t2(t):
    return "{:<2}".format(t)









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


def off_units(lines):
    """Unit names from the !!index array that opens an OFF library."""
    units = []
    if lines and lines[0].startswith("!!index"):
        for line in lines[1:]:
            if line.startswith("!"):
                break
            units.append(line.strip().strip('"'))
    return units


def read_off(path, unit=None):
    """AMBER OFF / .lib library (the format tleap's saveOff writes).

    A library may hold many units -- amino12.lib holds every residue -- so
    `unit` picks one, and is required when there is a choice."""
    lines = open(path).read().splitlines()
    units = off_units(lines)
    if unit is None:
        if len(units) > 1:
            sys.exit("ERROR: {0} holds {1} units; choose one as {0}:UNIT "
                     "(one of: {2})".format(path, len(units), " ".join(units)))
        unit = units[0] if units else None
    elif units and unit not in units:
        sys.exit("ERROR: no unit '{}' in {}; it holds: {}"
                 .format(unit, path, " ".join(units)))
    atoms_hdr = "!entry.{}.unit.atoms table".format(unit) if unit else None
    conn_hdr = "!entry.{}.unit.connectivity table".format(unit) if unit else None
    atoms, bonds = OrderedDict(), []
    i, n = 0, len(lines)
    while i < n:
        head = lines[i]
        if (head.startswith(atoms_hdr) if atoms_hdr else
                head.startswith("!entry.") and ".unit.atoms table" in head):
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
        if (head.startswith(conn_hdr) if conn_hdr else
                head.startswith("!entry.") and ".unit.connectivity table" in head):
            i += 1
            while i < n and not lines[i].startswith("!"):
                t = lines[i].split()
                if len(t) >= 2:
                    bonds.append((int(t[0]), int(t[1])))
                i += 1
            continue
        i += 1
    if not atoms:
        sys.exit("ERROR: no unit.atoms table found in {}{}"
                 .format(path, " for unit " + unit if unit else ""))
    return atoms, adjacency(atoms, bonds)




def read_structure(path, unit=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mol2":
        if unit:
            sys.exit("ERROR: a mol2 template has no units ({}:{})".format(path, unit))
        return read_mol2_graph(path)
    if ext in (".lib", ".off"):
        return read_off(path, unit)
    sys.exit("ERROR: unsupported template '{}' (use .mol2, .lib or .off)"
             .format(path))


def adjacency(atoms, bonds):
    adj = dict((i, []) for i in atoms)
    for a, b in bonds:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    return adj


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


def prune_open(mapping, G, Gadj, unmatched_ok):
    """Drop matched model atoms that have an unmatched heavy neighbour.

    A truncated residue only ever LOSES atoms relative to its template, so a
    correctly matched model atom cannot carry a heavy neighbour the template
    atom lacks -- unless that neighbour was claimed by an earlier template,
    which lets two templates cover adjacent fragments. A match that violates
    this has slid onto the wrong site (a LYS backbone onto a carboxylate, a
    LYS chain onto a ring); dropping the offending atoms, and whatever they
    orphan, leaves only the genuine fragment. A fragment that really is bonded
    to an atom outside every template (a covalent intermediate) cannot be
    matched this way; type it with --types."""
    mt = dict(mapping)
    changed = True
    while changed:
        changed = False
        matched = set(mt.values())
        for t, g in list(mt.items()):
            for nb in Gadj[g]:
                if (G[nb]["elem"] != "H" and nb not in matched
                        and nb not in unmatched_ok):
                    del mt[t]
                    changed = True
                    break
    return mt


def match_heavy(T, Tadj, G, Gadj, avail, radius=3, max_seeds=400,
                anchor=None, unmatched_ok=()):
    """Largest connected common substructure over heavy atoms.

    Candidates are pruned with prune_open before they are compared, and are
    ranked by heavy atoms matched, then by hydrogens matched: a lysine NZ-CE
    and its backbone N-CA both cover two heavy atoms of a truncated
    CH3-NH3+, but only NZ-CE also accounts for five of its six hydrogens.
    With `anchor`, only correspondences that include that model atom count."""
    tsig = dict((i, signature(i, T, Tadj, radius)) for i in T)
    gsig = dict((i, signature(i, G, Gadj, radius)) for i in G)

    def scored(t, g):
        m = prune_open(grow(t, g, T, Tadj, G, Gadj, tsig, gsig, avail),
                       G, Gadj, unmatched_ok)
        if not m or (anchor is not None and anchor not in m.values()):
            return (0, 0), {}
        return (len(m), len(add_hydrogens(m, T, Tadj, G, Gadj)) - len(m)), m

    def head(sig, r):
        return "|".join(sig.split("|")[:r + 1])

    best, best_score = {}, (0, 0)
    if anchor is not None:
        # Pair the anchor with the template atoms whose neighbourhood looks
        # most like it, deepest matching signature first and bare element
        # last (a capped CH3 never matches its template CH2 exactly).
        for r in range(radius, -1, -1):
            for t in T:
                if T[t]["elem"] == "H" or T[t]["elem"] != G[anchor]["elem"]:
                    continue
                if head(tsig[t], r) != head(gsig[anchor], r):
                    continue
                score, m = scored(t, anchor)
                if score > best_score:
                    best, best_score = m, score
            if best:
                break
        return best
    for r in range(radius, 0, -1):
        seeds = []
        for t in T:
            if T[t]["elem"] == "H":
                continue
            for g in G:
                if g not in avail or G[g]["elem"] == "H":
                    continue
                if T[t]["elem"] != G[g]["elem"]:
                    continue
                if head(gsig[g], r) == head(tsig[t], r):
                    seeds.append((t, g))
        for t, g in seeds[:max_seeds]:
            score, m = scored(t, g)
            if score > best_score:
                best, best_score = m, score
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


def cap_hydrogens(mapping, T, Tadj, G, Gadj):
    """Types for the model hydrogens left over on matched heavy atoms.

    A truncated residue is capped with hydrogens the template does not have
    (a CH2 that became CH3). Each takes the type of the template hydrogens on
    that atom, when those agree; an atom whose template carries no hydrogen
    gets nothing and stays on its input type."""
    used = set(mapping.values())
    caps = {}
    for t, g in mapping.items():
        if T[t]["elem"] == "H":
            continue
        htypes = set(T[x]["type"] for x in Tadj[t] if T[x]["elem"] == "H")
        if len(htypes) != 1:
            continue
        htype = htypes.pop()
        for y in Gadj[g]:
            if G[y]["elem"] == "H" and y not in used:
                caps[y] = htype
    return caps




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
# How many numbers a well-formed record carries. Real frcmods in the wild also
# contain lines that do not fit (odd IMPROPER variants, LJEDIT-style rows, stray
# text); those are passed through byte-for-byte instead of being reformatted, so
# nothing is ever lost or silently altered.
EXPECTED_NUMBERS = {"MASS": (1, 2), "BOND": (2,), "ANGLE": (2,),
                    "DIHE": (4,), "IMPROPER": (3,), "NONBON": (2, 3)}


def frc_parse(section, line):
    """Split a frcmod line into (canonical key, number TEXTS, comment, raw line).

    The numbers are kept as their original strings and never re-formatted:
    re-printing them at a fixed number of decimals silently truncates published
    parameters (a vdW epsilon of 5e-06 becomes 0.0), so columns are aligned by
    padding the original text instead.

    For BOND/ANGLE/DIHE/IMPROPER the type field is fixed-width in AMBER (two
    characters per type joined by '-') and is re-padded to canonical form. For
    MASS/NONBON the key is a single type and is left alone -- ionic types such
    as `F-`, `Cl-` and `Na+` contain characters that must not be treated as
    separators."""
    width = KEY_WIDTH[section]
    if width is None:
        tokens = line.split()
        key, rest = tokens[0], tokens[1:]
    else:
        raw_key, rest = line[:width], line[width:].split()
        key = "-".join("{:<2}".format(p.strip()) for p in raw_key.split("-"))
    numbers, comment = [], []
    for token in rest:
        if not comment:
            try:
                float(token)
                numbers.append(token)          # keep the exact original text
                continue
            except ValueError:
                pass
        comment.append(token)
    return key, numbers, " ".join(comment), line


def frc_types(section, key):
    """The atom types a key refers to."""
    if KEY_WIDTH[section] is None:
        return [key]                           # a single type, hyphens and all
    return [t.strip() for t in key.split("-")]


def frc_render(section, key, numbers, comment, raw):
    """One column-aligned frcmod line, with every value bit-preserved.

    A record whose number count is not what the section expects is returned
    exactly as it came in: better an unaligned line than a mangled parameter."""
    if len(numbers) not in EXPECTED_NUMBERS[section]:
        return raw
    # Every column is preceded by its own space, so a value that fills the
    # field width cannot run into its neighbour (parameters like -0.68404904
    # are exactly as wide as the column).
    if section == "NONBON":
        # AMBER reads type, R*, epsilon and ignores the rest of the line, so a
        # third column is dropped rather than carried as a phantom parameter.
        used = numbers[:2]
        text = "  {:<4}{}".format(key, "".join(" {:>11}".format(n)
                                               for n in used))
    elif section == "MASS":
        text = "{:<4}{}".format(key, "".join(" {:>10}".format(n)
                                             for n in numbers))
    else:
        # LEaP reads IMPROPER values from column 15 onward (it skips the
        # optional IDIVF field with sscanf(line+15, ...)), so a value that
        # begins earlier is silently truncated: "10.5" at column 14 reads as
        # 0.5. Pad that key to 15 so every value starts at or after it.
        width = 15 if section == "IMPROPER" else KEY_WIDTH[section]
        text = "{:<{w}}{}".format(key, "".join(" {:>10}".format(n)
                                               for n in numbers),
                                  w=width)
    return text + ("   " + comment if comment else "")


def looks_like_header(stripped):
    """A bare all-caps word on its own line starts a section."""
    return (stripped.isupper() and stripped.replace("_", "").isalpha()
            and len(stripped.split()) == 1)


def frc_sections(text):
    """(sections, extras) for one frcmod.

    sections: {section: OrderedDict(key -> [(numbers, comment, raw), ...])}
    extras:   [(header, [raw lines])] for sections this tool does not model
              (LJEDIT, CMAP, ...), carried through verbatim so a published
              parameter file is never silently truncated.
    """
    sections = OrderedDict((s, OrderedDict()) for s in FRC_SECTIONS)
    extras, current, extra_block = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # parmchk2 writes zero-valued placeholders with this marker when it
        # cannot identify a parameter. They are diagnostics, not parameters.
        if "ATTN" in stripped and "need revision" in stripped:
            continue
        upper = stripped.upper()
        if upper in FRC_ALIASES:
            current, extra_block = FRC_ALIASES[upper], None
            continue
        if looks_like_header(stripped):
            current, extra_block = None, (stripped, [])
            extras.append(extra_block)
            continue
        if extra_block is not None:
            extra_block[1].append(line)
            continue
        if current is None:
            continue                       # the title line
        key, numbers, comment, raw_line = frc_parse(current, line)
        sections[current].setdefault(key, []).append((numbers, comment, raw_line))
    return sections, extras




def frc_write(path, title, sections, extras=()):
    out = [title]
    for section in FRC_SECTIONS:
        out.append(section)
        for key, records in sections[section].items():
            for numbers, comment, raw in records:
                out.append(frc_render(section, key, numbers, comment, raw))
        out.append("")
    for header, body in extras:
        out.append(header)
        out.extend(body)
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




def run_parmchk2(mol2_path, out_path, workdir, gaff_path):
    # -p makes parmchk2 check against the GAFF file given on the command line
    # (the one tleap loads) and ignore its built-in tables, which predate
    # c5/c6/ns/nt/nz and would emit "same as c3-n" approximations for terms
    # the file defines. parmchk2 ignores -s once -p is given.
    cmd = [amber_bin("parmchk2"), "-i", mol2_path, "-f", "mol2",
           "-o", out_path, "-p", gaff_path]
    p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    if p.returncode != 0 or not os.path.isfile(out_path):
        sys.exit("ERROR: parmchk2 failed:\n" + out)




# ---------------------------------------------------------------------------
# selection file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# source-aware resolver configuration
# ---------------------------------------------------------------------------
SOURCE_NAMES = ("custom", "published", "ff19sb", "gaff2")


def empty_sections():
    return OrderedDict((s, OrderedDict()) for s in FRC_SECTIONS)


def resolve_config_path(value, config_dir):
    value = os.path.expandvars(os.path.expanduser(value))
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(config_dir, value))


def read_mapping_config(path):
    """Read the deliberately small, dependency-free mapping format.

    [FILES] rows are KEY VALUE, [PRIORITY] has one source per row,
    [FF19SB_TEMPLATES] rows are LIB UNIT TARGET_ANCHOR, [TS_BONDS] rows are
    ATOM1 ATOM2 breaking|forming, [CUSTOM_PARAMETERS] rows are SECTION
    TYPE-KEY VALUES..., and [ATOMS] rows are atom_name category final_type
    source:type [...].
    """
    files, priority, atoms = OrderedDict(), [], OrderedDict()
    templates, ts_bonds = [], []
    inline_custom = empty_sections()
    section = None
    with open(path) as fh:
        mapping_lines = fh.readlines()
    for lineno, raw in enumerate(mapping_lines, 1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().upper()
            if section not in ("FILES", "PRIORITY", "FF19SB_TEMPLATES",
                               "TS_BONDS", "CUSTOM_PARAMETERS", "ATOMS"):
                sys.exit("ERROR: {}:{}: unknown section [{}]"
                         .format(path, lineno, section))
            continue
        cols = body.split()
        if section == "FILES":
            if len(cols) != 2:
                sys.exit("ERROR: {}:{}: [FILES] needs KEY VALUE"
                         .format(path, lineno))
            files[cols[0].upper()] = cols[1]
        elif section == "PRIORITY":
            if len(cols) != 1 or cols[0].lower() not in SOURCE_NAMES:
                sys.exit("ERROR: {}:{}: priority must be one of {}"
                         .format(path, lineno, ", ".join(SOURCE_NAMES)))
            priority.append(cols[0].lower())
        elif section == "FF19SB_TEMPLATES":
            if len(cols) != 3:
                sys.exit("ERROR: {}:{}: [FF19SB_TEMPLATES] needs "
                         "LIBRARY UNIT TARGET_ATOM_NAME".format(path, lineno))
            templates.append(tuple(cols))
        elif section == "TS_BONDS":
            if len(cols) != 3 or cols[2].lower() not in ("breaking", "forming"):
                sys.exit("ERROR: {}:{}: [TS_BONDS] needs "
                         "ATOM1 ATOM2 breaking|forming".format(path, lineno))
            ts_bonds.append((cols[0], cols[1], cols[2].lower()))
        elif section == "CUSTOM_PARAMETERS":
            if len(cols) < 3:
                sys.exit("ERROR: {}:{}: [CUSTOM_PARAMETERS] needs "
                         "SECTION TYPE-KEY VALUES...".format(path, lineno))
            parameter_section = FRC_ALIASES.get(cols[0].upper())
            if parameter_section is None:
                sys.exit("ERROR: {}:{}: unsupported custom parameter section "
                         "'{}'".format(path, lineno, cols[0]))
            type_parts = cols[1].split("-")
            expected_types = {"MASS": 1, "NONBON": 1, "BOND": 2,
                              "ANGLE": 3, "DIHE": 4, "IMPROPER": 4}
            if len(type_parts) != expected_types[parameter_section]:
                sys.exit("ERROR: {}:{}: {} key '{}' needs {} atom type(s)"
                         .format(path, lineno, parameter_section, cols[1],
                                 expected_types[parameter_section]))
            try:
                for value in cols[2:]:
                    float(value)
            except ValueError:
                sys.exit("ERROR: {}:{}: custom parameter values must be numeric"
                         .format(path, lineno))
            if len(cols[2:]) not in EXPECTED_NUMBERS[parameter_section]:
                sys.exit("ERROR: {}:{}: {} needs {} numerical value(s)"
                         .format(path, lineno, parameter_section,
                                 " or ".join(str(x) for x in
                                             EXPECTED_NUMBERS[parameter_section])))
            key = (type_parts[0] if len(type_parts) == 1 else
                   "-".join(t2(atom_type) for atom_type in type_parts))
            record = (cols[2:], "inline custom parameter", body)
            inline_custom[parameter_section].setdefault(key, []).append(record)
        elif section == "ATOMS":
            if len(cols) < 4:
                sys.exit("ERROR: {}:{}: atom row needs NAME CATEGORY FINAL "
                         "SOURCE:TYPE [...]".format(path, lineno))
            name, category, final_type = cols[:3]
            category = category.lower()
            if category not in SOURCE_NAMES:
                sys.exit("ERROR: {}:{}: bad atom category '{}'"
                         .format(path, lineno, category))
            if name in atoms:
                sys.exit("ERROR: {}:{}: duplicate atom name '{}'"
                         .format(path, lineno, name))
            aliases = OrderedDict()
            for token in cols[3:]:
                if ":" not in token:
                    sys.exit("ERROR: {}:{}: alias '{}' must be SOURCE:TYPE"
                             .format(path, lineno, token))
                source, atom_type = token.split(":", 1)
                source = source.lower()
                if source not in SOURCE_NAMES or source == "custom":
                    sys.exit("ERROR: {}:{}: alias source '{}' must be published, "
                             "ff19sb, or gaff2".format(path, lineno, source))
                if source in aliases:
                    sys.exit("ERROR: {}:{}: duplicate {} alias for {}"
                             .format(path, lineno, source, name))
                aliases[source] = atom_type
            atoms[name] = {"category": category, "final": final_type,
                           "aliases": aliases, "line": lineno}
        else:
            sys.exit("ERROR: {}:{}: put this row inside [FILES], [PRIORITY], "
                     "[FF19SB_TEMPLATES], [TS_BONDS], "
                     "[CUSTOM_PARAMETERS], or [ATOMS]"
                     .format(path, lineno))

    # Compatibility for mapping files created before the generic published
    # source replaced the NADH-specific label. New files should use the generic
    # keys so reports remain meaningful for any cofactor or ligand.
    if "PUBLISHED_LIB" not in files and "NADH_LIB" in files:
        files["PUBLISHED_LIB"] = files["NADH_LIB"]
    if "PUBLISHED_FRCMOD" not in files and "NADH_FRCMOD" in files:
        files["PUBLISHED_FRCMOD"] = files["NADH_FRCMOD"]
    required = ("MOL2", "PUBLISHED_LIB", "PUBLISHED_FRCMOD", "GAFF2",
                "FF19SB")
    missing = [key for key in required if key not in files]
    if missing:
        sys.exit("ERROR: {}: missing [FILES] entries: {}"
                 .format(path, ", ".join(missing)))
    if not priority:
        priority = list(SOURCE_NAMES)
    if len(set(priority)) != len(priority):
        sys.exit("ERROR: {}: duplicate source in [PRIORITY]".format(path))
    for source in SOURCE_NAMES:
        if source not in priority:
            priority.append(source)
    if priority[0] != "custom":
        priority.remove("custom")
        priority.insert(0, "custom")
    return files, priority, atoms, templates, ts_bonds, inline_custom


def locate_leaprc(name):
    candidate = os.path.expandvars(os.path.expanduser(name))
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    home = os.environ.get("AMBERHOME", "")
    candidate = os.path.join(home, "dat", "leap", "cmd", name)
    if os.path.isfile(candidate):
        return candidate
    sys.exit("ERROR: cannot find leaprc '{}'; set AMBERHOME or give its path"
             .format(name))


def locate_amber_file(name, current_dir):
    name = os.path.expandvars(os.path.expanduser(name.strip('"\'')))
    candidates = [name, os.path.join(current_dir, name)]
    home = os.environ.get("AMBERHOME", "")
    candidates += [os.path.join(home, "dat", "leap", part, name)
                   for part in ("parm", "cmd", "lib")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    sys.exit("ERROR: cannot resolve Amber file '{}' referenced from {}"
             .format(name, current_dir))


def read_leaprc_stack(name, seen=None):
    """Return (parameter files, atom-type definitions) loaded by a leaprc."""
    path = locate_leaprc(name)
    seen = set() if seen is None else seen
    if path in seen:
        return [], {}
    seen.add(path)
    text = open(path).read()
    params, type_defs = [], {}
    for atom_type, elem, hybrid in re.findall(
            r'\{\s*"([^"]+)"\s+"([^"]*)"\s+"([^"]*)"\s*\}', text):
        type_defs[atom_type] = (elem, hybrid)
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.search(r'\bloadamberparams\s+(\S+)', line,
                          flags=re.IGNORECASE)
        if match:
            params.append(locate_amber_file(match.group(1), os.path.dirname(path)))
        match = re.match(r'^source\s+(\S+)', line, flags=re.IGNORECASE)
        if match:
            child_params, child_types = read_leaprc_stack(match.group(1), seen)
            params.extend(child_params)
            type_defs.update(child_types)
    unique = []
    for param in params:
        if param not in unique:
            unique.append(param)
    return unique, type_defs


class ParameterDB(object):
    """One force-field source, kept separate from every other source."""

    def __init__(self, name):
        self.name = name
        self.sections = empty_sections()
        self.origins = dict((s, {}) for s in FRC_SECTIONS)

    def merge_file(self, path, origin=None):
        parsed, extras = read_amber_params(path)
        if extras and any(header.upper() == "LJEDIT" for header, _ in extras):
            sys.exit("ERROR: {} contains LJEDIT; source-aware LJEDIT resolution "
                     "is required before this file can be used".format(path))
        self.merge_sections(parsed, origin or path)

    def merge_sections(self, parsed, origin):
        for section, entries in parsed.items():
            for key, records in entries.items():
                self.sections[section][key] = records
                self.origins[section][key] = origin

    def lookup(self, section, query):
        """Return (matched types, records, origin), honoring Amber wildcards."""
        query = tuple(query)
        entries = self.sections[section]
        if section in ("MASS", "NONBON"):
            key = query[0]
            if key in entries:
                return (key,), entries[key], self.origins[section][key]
            return None

        candidates = []
        for order, (key, records) in enumerate(entries.items()):
            pattern = tuple(frc_types(section, key))
            orientations = (query, query[::-1])
            for orient in orientations:
                if len(pattern) != len(orient):
                    continue
                if all(p == "X" or p == q for p, q in zip(pattern, orient)):
                    specificity = sum(p != "X" for p in pattern)
                    candidates.append((specificity, order, pattern, records,
                                       self.origins[section][key]))
                    break
        if not candidates:
            return None
        _specificity, _order, pattern, records, origin = max(candidates,
                                                              key=lambda x: (x[0], x[1]))
        return pattern, records, origin

    def lookup_improper(self, center, neighbors, aliases):
        """Match an improper with the AMBER central atom in position three."""
        entries = self.sections["IMPROPER"]
        candidates = []
        for permutation in itertools.permutations(neighbors):
            atoms = (permutation[0], permutation[1], center, permutation[2])
            query = tuple(aliases[i] for i in atoms)
            for order, (key, records) in enumerate(entries.items()):
                pattern = tuple(frc_types("IMPROPER", key))
                if len(pattern) != 4:
                    continue
                if all(p == "X" or p == q for p, q in zip(pattern, query)):
                    specificity = sum(p != "X" for p in pattern)
                    candidates.append((specificity, order, atoms, pattern,
                                       records, self.origins["IMPROPER"][key]))
        if not candidates:
            return None
        _specificity, _order, atoms, pattern, records, origin = max(
            candidates, key=lambda x: (x[0], x[1]))
        return atoms, pattern, records, origin


def amber_auto_type(idx, atom_tokens, types, adj):
    """Limited Amber typing for saturated ribose-like C/O/H environments."""
    name = atom_tokens[idx][1]
    elem = element_of(name, types[idx])
    neighbors = adj[idx]
    if elem == "C" and types[idx].lower() in ("c3", "c5", "c6"):
        return "CT", "environment", "HIGH"
    if elem == "O":
        elems = [element_of(atom_tokens[j][1], types[j]) for j in neighbors]
        if "H" in elems and "C" in elems:
            return "OH", "environment", "HIGH"
        if sum(e != "H" for e in elems) == 2:
            return "OS", "environment", "HIGH"
    if elem == "H" and len(neighbors) == 1:
        parent = neighbors[0]
        pelem = element_of(atom_tokens[parent][1], types[parent])
        if pelem == "O":
            return "HO", "environment", "HIGH"
        if pelem == "N":
            return "H", "environment", "HIGH"
        if pelem == "C":
            ewg = sum(element_of(atom_tokens[j][1], types[j]) in ("N", "O", "S")
                      for j in adj[parent])
            return (("HC" if ewg == 0 else "H{}".format(min(ewg, 3))),
                    "environment", "MEDIUM")
    sys.exit("ERROR: atom {} ({}) cannot be typed safely by ff19sb:auto; "
             "give an explicit final type and ff19sb alias"
             .format(idx, name))


def infer_hybridization(atom_type):
    t = atom_type.strip()
    low = t.lower()
    if low in ("c1", "n1"):
        return "sp"
    if low in ("c", "c2", "ca", "cc", "cd", "ce", "cf", "cg", "ch",
               "cp", "cq", "cu", "cv", "cz", "n", "n2", "na", "nb",
               "nc", "nd", "ne", "nf", "nh", "ni", "nj", "nm", "nn",
               "no", "ns", "nt", "nu", "nv", "o", "o2", "s", "s2"):
        return "sp2"
    return "sp3"


def canonical_term(section, types_):
    types_ = tuple(types_)
    if section in ("BOND", "ANGLE", "DIHE"):
        return min(types_, types_[::-1])
    if section == "IMPROPER":
        return (types_[2],) + tuple(sorted((types_[0], types_[1], types_[3])))
    return types_


def key_for_types(section, types_):
    if section in ("MASS", "NONBON"):
        return types_[0]
    return "-".join(t2(t) for t in types_)


def region_name(atom_ids, atom_specs):
    names = []
    for idx in atom_ids:
        category = atom_specs[idx]["category"]
        label = "amber" if category == "ff19sb" else category
        if label not in names:
            names.append(label)
    if len(names) == 1:
        return "PURE_" + names[0].upper()
    return "_".join(name.upper() for name in names)


def write_atomtypes(path, atom_tokens, types, otype, atom_specs, known_types):
    """Write definitions for every final type used by the output molecule."""
    definitions = OrderedDict()
    for idx in sorted(atom_tokens):
        final_type = otype[idx]
        # A custom label can collide with an unrelated host-force-field type
        # (for example, the figure's sp2 CI and ff19SB's sp3 CI). Its chemical
        # identity comes from the mapped atom, not from that accidental name
        # collision.
        if atom_specs[idx]["category"] == "custom":
            elem = element_of(atom_tokens[idx][1], types[idx])
            hybrid = infer_hybridization(types[idx])
        elif final_type in known_types:
            known_elem, known_hybrid = known_types[final_type]
            elem = known_elem or element_of(atom_tokens[idx][1], types[idx])
            hybrid = known_hybrid or infer_hybridization(types[idx])
        else:
            elem = element_of(atom_tokens[idx][1], types[idx])
            hybrid = infer_hybridization(types[idx])
        value = (elem, hybrid)
        if final_type in definitions and definitions[final_type] != value:
            sys.exit("ERROR: final type '{}' is used for incompatible elements/"
                     "hybridizations".format(final_type))
        definitions[final_type] = value
    with open(path, "w") as fh:
        fh.write("addAtomTypes {\n")
        for atom_type, (elem, hybrid) in definitions.items():
            fh.write('    {{ "{}" "{}" "{}" }}\n'
                     .format(atom_type, elem, hybrid))
        fh.write("}\n")
    return definitions


def run_generated_tleap(leap_input, workdir):
    p = subprocess.Popen([amber_bin("tleap"), "-f", os.path.abspath(leap_input)],
                         cwd=workdir, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    log = p.communicate()[0].decode("utf-8", "replace")
    errors = warnings = None
    for line in log.splitlines():
        if "Exiting LEaP" in line:
            match = re.search(r"Errors\s*=\s*(\d+).*Warnings\s*=\s*(\d+)", line)
            if match:
                errors, warnings = int(match.group(1)), int(match.group(2))
    return p.returncode, errors, warnings, log


def validate_prmtop_bonds(path, expected_bonds):
    """Require the saved topology to retain every bond from the output mol2."""
    sections, current = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("%FLAG "):
                current = line.split(None, 1)[1].strip()
                sections[current] = []
            elif line.startswith("%FORMAT"):
                continue
            elif current is not None:
                sections[current].append(line)

    actual = set()
    for section in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = [int(value) for line in sections.get(section, [])
                  for value in line.split()]
        if len(values) % 3:
            raise ValueError("{} has a malformed {} section"
                             .format(path, section))
        for offset in range(0, len(values), 3):
            actual.add(frozenset((values[offset] // 3 + 1,
                                  values[offset + 1] // 3 + 1)))
    expected = set(frozenset(pair) for pair in expected_bonds)
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        describe = lambda pairs: ", ".join(
            "{}-{}".format(*sorted(pair)) for pair in sorted(
                pairs, key=lambda pair: sorted(pair))) or "none"
        raise ValueError("{} bond graph differs from the output mol2; missing: "
                         "{}; extra: {}".format(path, describe(missing),
                                                 describe(extra)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a source-resolved hybrid TS force field.",
        epilog="example:\n  python build_tsff.py atom_mapping.txt -o hybrid")
    ap.add_argument("mapping", help="sectioned atom/source mapping text file")
    ap.add_argument("-o", "--out", required=True,
                    help="output prefix (hybrid -> hybrid.mol2, .frcmod, ...)")
    ap.add_argument("--no-verify", action="store_true",
                    help="write outputs without running tleap")
    ap.add_argument("--allow-ts-seeds", action="store_true",
                    help="generate flagged initial angle/torsion seeds when a "
                         "custom TS interaction has no source parameter")
    ap.add_argument("--ts-equilibria", action="store_true",
                    help="take the equilibrium length/angle of every "
                         "reaction-center bond and angle (every instance "
                         "involves a custom atom) from the input TS "
                         "geometry; force constants keep their source values")
    opts = ap.parse_args(argv)

    config_path = os.path.abspath(opts.mapping)
    config_dir = os.path.dirname(config_path)
    (files, priority, configured_atoms, ff19_templates, configured_ts_bonds,
     inline_custom) = read_mapping_config(config_path)
    mol2_path = resolve_config_path(files["MOL2"], config_dir)
    published_lib = resolve_config_path(files["PUBLISHED_LIB"], config_dir)
    published_frcmod = resolve_config_path(files["PUBLISHED_FRCMOD"], config_dir)
    custom_value = files.get("CUSTOM_FRCMOD", "-")
    custom_frcmod = (None if custom_value == "-" else
                     resolve_config_path(custom_value, config_dir))
    for path in (mol2_path, published_lib, published_frcmod):
        if not os.path.isfile(path):
            sys.exit("ERROR: no such input file: {}".format(path))
    if custom_frcmod and not os.path.isfile(custom_frcmod):
        sys.exit("ERROR: no such custom frcmod: {}".format(custom_frcmod))

    gaff_param_paths, gaff_type_defs = read_leaprc_stack(files["GAFF2"])
    amber_param_paths, amber_type_defs = read_leaprc_stack(files["FF19SB"])
    if not gaff_param_paths:
        sys.exit("ERROR: {} does not load a GAFF parameter file"
                 .format(files["GAFF2"]))
    gaff_path = next((path for path in gaff_param_paths
                      if "gaff" in os.path.basename(path).lower()),
                     gaff_param_paths[-1])

    lines, atom_tokens, coords, types, bonds, atom_line_idx = read_mol2(mol2_path)
    names = {}
    for idx, tokens in atom_tokens.items():
        name = tokens[1]
        if name in names:
            sys.exit("ERROR: atom name '{}' is not unique (indices {} and {}); "
                     "the mapping uses stable atom names"
                     .format(name, names[name], idx))
        names[name] = idx
    unknown = sorted(set(configured_atoms) - set(names))
    missing = sorted(set(names) - set(configured_atoms))
    if unknown:
        sys.exit("ERROR: mapping names not present in mol2: {}"
                 .format(", ".join(unknown)))
    if missing:
        sys.exit("ERROR: every mol2 atom must be mapped; missing: {}"
                 .format(", ".join(missing)))

    adj = adjacency(atom_tokens, bonds)
    input_bonds = set(frozenset((a, b)) for a, b in bonds)
    ts_bond_roles = {}
    for first_name, second_name, role in configured_ts_bonds:
        if first_name not in names or second_name not in names:
            sys.exit("ERROR: TS bond {}-{} names an atom absent from the mol2"
                     .format(first_name, second_name))
        pair = frozenset((names[first_name], names[second_name]))
        if pair not in input_bonds:
            sys.exit("ERROR: TS bond {}-{} is not covalent in the input mol2"
                     .format(first_name, second_name))
        if pair in ts_bond_roles:
            sys.exit("ERROR: duplicate TS bond {}-{}"
                     .format(first_name, second_name))
        ts_bond_roles[pair] = role

    # LEaP enforces ordinary valence while reading mol2 and can silently drop
    # the forming bond of a hydride that is covalently attached to two atoms.
    # Load such forming bonds through a temporary perturbation override, then
    # immediately return the atom to its ordinary (unperturbed) state.
    ts_counts = {}
    for pair in ts_bond_roles:
        for idx in pair:
            ts_counts[idx] = ts_counts.get(idx, 0) + 1
    leap_restored_bonds = []
    for first_name, second_name, role in configured_ts_bonds:
        if role != "forming":
            continue
        first, second = names[first_name], names[second_name]
        hydrogen = next((idx for idx in (first, second)
                         if element_of(atom_tokens[idx][1], types[idx]) == "H"
                         and ts_counts.get(idx, 0) > 1), None)
        if hydrogen is not None:
            other = second if hydrogen == first else first
            leap_restored_bonds.append((other, hydrogen))

    # Match the published library once. It validates every declared published
    # alias and supplies published:auto without copying library charges.
    G, Gadj = read_mol2_graph(mol2_path)
    T, Tadj = read_structure(published_lib)
    heavy = match_heavy(T, Tadj, G, Gadj, set(G))
    if not heavy:
        sys.exit("ERROR: published library does not match any part of {}"
                 .format(mol2_path))
    full = add_hydrogens(heavy, T, Tadj, G, Gadj)
    published_types = dict((g, T[t]["type"]) for t, g in full.items())
    published_types.update(cap_hydrogens(full, T, Tadj, G, Gadj))

    # Standard protein fragments are typed by matching their local graph to
    # the named ff19SB residue unit. The target anchor selects the intended
    # fragment; atom names and numbering inside the Amber library need not
    # match the input mol2.
    ff19_template_types = {}
    ff19_template_origin = {}
    template_match_counts = []
    for library_name, unit, anchor_name in ff19_templates:
        if anchor_name not in names:
            sys.exit("ERROR: ff19SB template anchor '{}' is not an atom in {}"
                     .format(anchor_name, mol2_path))
        candidate = resolve_config_path(library_name, config_dir)
        library_path = (candidate if os.path.isfile(candidate) else
                        locate_amber_file(library_name, config_dir))
        RT, RTadj = read_structure(library_path, unit)
        residue_heavy = match_heavy(RT, RTadj, G, Gadj, set(G),
                                    anchor=names[anchor_name])
        if not residue_heavy:
            sys.exit("ERROR: ff19SB template {}:{} did not match the fragment "
                     "anchored at {}".format(library_path, unit, anchor_name))
        residue_full = add_hydrogens(residue_heavy, RT, RTadj, G, Gadj)
        residue_types = dict((g, RT[t]["type"])
                             for t, g in residue_full.items())
        residue_types.update(cap_hydrogens(residue_full, RT, RTadj, G, Gadj))
        origin = "{}:{}@{}".format(library_path, unit, anchor_name)
        for idx, atom_type in residue_types.items():
            if (idx in ff19_template_types and
                    ff19_template_types[idx] != atom_type):
                sys.exit("ERROR: ff19SB templates assign atom {} both {} and {}"
                         .format(atom_tokens[idx][1],
                                 ff19_template_types[idx], atom_type))
            ff19_template_types[idx] = atom_type
            ff19_template_origin[idx] = origin
        template_match_counts.append((unit, anchor_name, len(residue_types),
                                      len(residue_heavy)))

    def resolve_ff19_auto(idx):
        if idx in ff19_template_types:
            return (ff19_template_types[idx], ff19_template_origin[idx],
                    "EXACT")
        return amber_auto_type(idx, atom_tokens, types, adj)

    atom_specs, otype = {}, {}
    auto_notes = {}
    for name, configured in configured_atoms.items():
        idx = names[name]
        spec = {"category": configured["category"],
                "aliases": OrderedDict(configured["aliases"]),
                "confidence": "EXACT", "assignment": "explicit"}
        final_type = configured["final"]
        if final_type == "original":
            final_type = types[idx]
            spec["assignment"] = "original mol2"
        elif final_type == "auto":
            if spec["category"] == "published":
                if idx not in published_types:
                    sys.exit("ERROR: {} is published:auto but was not matched "
                             "in {}".format(name, published_lib))
                final_type = published_types[idx]
                spec["assignment"] = "published library"
            elif spec["category"] == "ff19sb":
                final_type, assignment, confidence = resolve_ff19_auto(idx)
                spec["assignment"], spec["confidence"] = assignment, confidence
                auto_notes[idx] = confidence
            elif spec["category"] == "gaff2":
                final_type = types[idx]
                spec["assignment"] = "original mol2"
            else:
                sys.exit("ERROR: custom atom {} needs an explicit final type"
                         .format(name))
        spec["final"] = final_type
        for source, alias in list(spec["aliases"].items()):
            if alias == "original":
                spec["aliases"][source] = types[idx]
            elif alias == "auto":
                if source == "published" and idx in published_types:
                    spec["aliases"][source] = published_types[idx]
                elif source == "ff19sb":
                    spec["aliases"][source] = resolve_ff19_auto(idx)[0]
                else:
                    sys.exit("ERROR: cannot resolve {}:auto for {}"
                             .format(source, name))
        if not spec["aliases"]:
            sys.exit("ERROR: atom {} has no declared parameter-source alias"
                     .format(name))
        if "published" in spec["aliases"]:
            if idx not in published_types:
                sys.exit("ERROR: atom {} declares a published alias but {} did "
                         "not match it".format(name, published_lib))
            expected = published_types[idx]
            if spec["aliases"]["published"] != expected:
                sys.exit("ERROR: atom {} declares published:{} but {} maps it "
                         "to {}".format(name,
                                        spec["aliases"]["published"],
                                        published_lib, expected))
        atom_specs[idx] = spec
        otype[idx] = final_type

    # A final type represents one element and one parameter identity. Different
    # atoms may expose additional boundary aliases, but an alias shared by two
    # instances of the same final type cannot disagree.
    final_aliases = {}
    final_elements = {}
    for idx, spec in atom_specs.items():
        ft = spec["final"]
        elem = element_of(atom_tokens[idx][1], types[idx])
        if ft in final_elements and final_elements[ft] != elem:
            sys.exit("ERROR: final type {} is assigned to both {} and {}"
                     .format(ft, final_elements[ft], elem))
        final_elements[ft] = elem
        known = final_aliases.setdefault(ft, {})
        for source, alias in spec["aliases"].items():
            if source in known and known[source] != alias:
                sys.exit("ERROR: final type {} has conflicting {} aliases: {} "
                         "and {}".format(ft, source, known[source], alias))
            known[source] = alias

    workdir = tempfile.mkdtemp(prefix="build_tsff_resolver_")
    try:
        # parmchk2 is part of the GAFF2 source. DU atoms are replaced only in
        # this temporary copy, using their explicitly declared GAFF2 alias.
        gaff_lines = list(lines)
        for idx in sorted(atom_tokens):
            if types[idx].upper() not in ("DU", "DUMMY"):
                continue
            alias = atom_specs[idx]["aliases"].get("gaff2")
            if not alias:
                sys.exit("ERROR: DU atom {} needs a declared gaff2 alias for "
                         "parmchk2".format(atom_tokens[idx][1]))
            li = atom_line_idx[idx]
            gaff_lines[li] = set_mol2_type_inplace(gaff_lines[li], alias)
        gaff_mol2 = os.path.join(workdir, "all_gaff.mol2")
        with open(gaff_mol2, "w") as fh:
            fh.write("\n".join(gaff_lines) + "\n")
        gap_frcmod = os.path.join(workdir, "parmchk2.frcmod")
        run_parmchk2(gaff_mol2, gap_frcmod, workdir, gaff_path)

        dbs = OrderedDict((name, ParameterDB(name)) for name in SOURCE_NAMES)
        for path in gaff_param_paths:
            dbs["gaff2"].merge_file(path)
        dbs["gaff2"].merge_file(gap_frcmod, "parmchk2 gap fill")
        for path in amber_param_paths:
            dbs["ff19sb"].merge_file(path)
            # A published frcmod is commonly an overlay on Amber parameters,
            # rather than a complete force field. Keep that dependency in the
            # published namespace so terms spanning a cofactor boundary
            # can be resolved without inventing an ff19SB identity for NF.
            dbs["published"].merge_file(path)
        dbs["published"].merge_file(published_frcmod)
        if custom_frcmod:
            dbs["custom"].merge_file(custom_frcmod)
        inline_custom_origin = config_path + " [CUSTOM_PARAMETERS]"
        dbs["custom"].merge_sections(inline_custom, inline_custom_origin)

        amber_origins = set(os.path.abspath(path) for path in amber_param_paths)

        def numerical_source(namespace, hit):
            """Report the file that supplied the numbers, including overlays."""
            origin = os.path.abspath(hit[2]) if os.path.isabs(hit[2]) else hit[2]
            if namespace == "published" and origin in amber_origins:
                return "ff19sb"
            return namespace

        def missing_parameter(section, atom_ids, attempts):
            final = tuple(otype[i] for i in atom_ids)
            lines_ = ["UNRESOLVED {}: {}".format(section, "-".join(final))]
            for idx in atom_ids:
                aliases = ", ".join("{}:{}".format(k, v)
                                    for k, v in atom_specs[idx]["aliases"].items())
                lines_.append("  {} ({}): {} [{}]".format(
                    atom_tokens[idx][1], otype[idx], aliases or "no aliases",
                    atom_specs[idx]["category"]))
            lines_.append("  attempted: " + ("; ".join(attempts) or "none"))
            lines_.append("No declared common source supplied this term. Add an "
                          "alias only if chemically justified, or define the "
                          "exact final-type term in [CUSTOM_PARAMETERS]. For an "
                          "initial TSFF, --allow-ts-seeds permits flagged "
                          "angle/torsion seeds involving custom atoms.")
            raise ValueError("\n".join(lines_))

        def generated_ts_seed(section, atom_ids):
            """A deliberately narrow fallback for an initial Q2MM force field."""
            if (not opts.allow_ts_seeds or
                    not any(atom_specs[i]["category"] == "custom"
                            for i in atom_ids)):
                return None
            final = tuple(otype[i] for i in atom_ids)
            comment = "GENERATED_TS_SEED; optimize with Q2MM"
            if section == "ANGLE":
                theta = angle_degrees(coords[atom_ids[0]], coords[atom_ids[1]],
                                      coords[atom_ids[2]])
                numbers = ["50.0000", "{:.4f}".format(theta)]
            elif section == "DIHE":
                # A zero barrier is neutral for the unsupported initial term;
                # Q2MM must determine the barrier from the QM training data.
                numbers = ["1", "0.0000", "0.0000", "2.0000"]
            else:
                return None
            record = (numbers, comment, "")
            return "ts_seed", (final, [record], "generated by build_tsff.py")

        def resolve_term(section, atom_ids):
            final = tuple(otype[i] for i in atom_ids)
            attempts = []
            custom_hit = dbs["custom"].lookup(section, final)
            if custom_hit is not None:
                return "custom", custom_hit, attempts
            attempts.append("custom:{} absent".format("-".join(final)))
            for source in priority:
                if source == "custom":
                    continue
                aliases = []
                for idx in atom_ids:
                    alias = atom_specs[idx]["aliases"].get(source)
                    if alias is None:
                        break
                    aliases.append(alias)
                if len(aliases) != len(atom_ids):
                    attempts.append("{}:no common alias".format(source))
                    continue
                hit = dbs[source].lookup(section, tuple(aliases))
                if hit is not None:
                    return numerical_source(source, hit), hit, attempts
                attempts.append("{}:{} absent".format(source,
                                                       "-".join(aliases)))
            seed = generated_ts_seed(section, atom_ids)
            if seed is not None:
                source, hit = seed
                return source, hit, attempts
            missing_parameter(section, atom_ids, attempts)

        def resolve_atom_parameter(section, final_type, atom_ids):
            attempts = []
            custom_hit = dbs["custom"].lookup(section, (final_type,))
            if custom_hit is not None:
                return "custom", custom_hit, attempts
            attempts.append("custom:{} absent".format(final_type))
            for source in priority:
                if source == "custom":
                    continue
                aliases = [atom_specs[i]["aliases"].get(source) for i in atom_ids]
                if any(alias is None for alias in aliases) or len(set(aliases)) != 1:
                    attempts.append("{}:no common alias".format(source))
                    continue
                hit = dbs[source].lookup(section, (aliases[0],))
                if hit is not None:
                    return numerical_source(source, hit), hit, attempts
                attempts.append("{}:{} absent".format(source, aliases[0]))
            missing_parameter(section, [atom_ids[0]], attempts)

        output_sections = empty_sections()
        resolved = dict((s, OrderedDict()) for s in FRC_SECTIONS)
        report_rows = []
        counters = dict((s, 0) for s in FRC_SECTIONS)
        prefixes = {"MASS": "M", "BOND": "B", "ANGLE": "A", "DIHE": "D",
                    "IMPROPER": "I", "NONBON": "LJ"}

        def save_resolution(section, atom_ids, output_types, source, hit, attempts):
            matched, records, origin = hit
            identity = canonical_term(section, output_types)
            existing = resolved[section].get(identity)
            signature = (source, tuple(matched), tuple(tuple(r[0]) for r in records))
            instance = ",".join(str(i) for i in atom_ids)
            if existing is not None:
                if existing["signature"] != signature:
                    raise ValueError("final parameter {} {} resolves differently "
                                     "in two regions; give the atoms distinct final "
                                     "types".format(section, "-".join(identity)))
                if instance not in existing["row"]["atom_indices"].split(";"):
                    existing["row"]["atom_indices"] += ";" + instance
                return
            key_types = tuple(output_types)
            if section in ("BOND", "ANGLE", "DIHE"):
                key_types = identity
            key = key_for_types(section, key_types)
            output_sections[section][key] = records
            counters[section] += 1
            values = " | ".join(" ".join(record[0]) for record in records)
            ts_role = ""
            if section == "BOND":
                ts_role = ts_bond_roles.get(frozenset(atom_ids), "")
            elif source == "ts_seed":
                ts_role = "coupled_{}_seed".format(section.lower())
            elif source == "custom" and origin == inline_custom_origin:
                atom_set = set(atom_ids)
                if any(set(ts_pair).issubset(atom_set)
                       for ts_pair in ts_bond_roles):
                    ts_role = "coupled_{}_custom".format(section.lower())
            status = ("generated_ts_seed" if source == "ts_seed" else
                      "custom_override" if source == "custom" else
                      "ts_initial_{}".format(ts_role) if ts_role else
                      "published" if source == "published" else "inherited")
            row = OrderedDict([
                ("parameter_id", "{}{:03d}".format(prefixes[section],
                                                    counters[section])),
                ("section", section),
                ("final_types", "-".join(key_types)),
                ("lookup_types", "-".join(matched)),
                ("source", source),
                ("source_file", origin),
                ("status", status),
                ("ts_role", ts_role),
                ("region", region_name(atom_ids, atom_specs)),
                ("atom_indices", instance),
                ("values", values),
                ("earlier_attempts", "; ".join(attempts)),
                ("equilibrium", ""),
            ])
            resolved[section][identity] = {"signature": signature, "row": row}
            report_rows.append(row)

        by_final = OrderedDict()
        for idx in sorted(atom_tokens):
            by_final.setdefault(otype[idx], []).append(idx)
        for final_type, atom_ids in by_final.items():
            for section in ("MASS", "NONBON"):
                source, hit, attempts = resolve_atom_parameter(section, final_type,
                                                               atom_ids)
                save_resolution(section, atom_ids, (final_type,), source, hit,
                                attempts)

        for a, b in bonds:
            atom_ids = (a, b)
            source, hit, attempts = resolve_term("BOND", atom_ids)
            save_resolution("BOND", atom_ids, tuple(otype[i] for i in atom_ids),
                            source, hit, attempts)

        for center in sorted(atom_tokens):
            neighbors = adj[center]
            for i, k in itertools.combinations(neighbors, 2):
                atom_ids = (i, center, k)
                source, hit, attempts = resolve_term("ANGLE", atom_ids)
                save_resolution("ANGLE", atom_ids,
                                tuple(otype[x] for x in atom_ids), source, hit,
                                attempts)

        for a, b in bonds:
            for i in adj[a]:
                if i == b:
                    continue
                for l in adj[b]:
                    if l == a or l == i:
                        continue
                    atom_ids = (i, a, b, l)
                    source, hit, attempts = resolve_term("DIHE", atom_ids)
                    save_resolution("DIHE", atom_ids,
                                    tuple(otype[x] for x in atom_ids), source,
                                    hit, attempts)

        # An improper is emitted only when a source explicitly has a matching
        # improper for a three-neighbour center. No planarity term is invented.
        for center in sorted(atom_tokens):
            neighbors = tuple(adj[center])
            if len(neighbors) != 3:
                continue
            candidate = None
            attempts = []
            final_alias = dict((idx, otype[idx]) for idx in (center,) + neighbors)
            custom_hit = dbs["custom"].lookup_improper(center, neighbors,
                                                       final_alias)
            if custom_hit is not None:
                atoms_order, matched, records, origin = custom_hit
                candidate = ("custom", atoms_order,
                             (matched, records, origin))
            else:
                attempts.append("custom:absent")
                for source in priority:
                    if source == "custom":
                        continue
                    aliases = {}
                    for idx in (center,) + neighbors:
                        alias = atom_specs[idx]["aliases"].get(source)
                        if alias is None:
                            break
                        aliases[idx] = alias
                    if len(aliases) != 4:
                        attempts.append("{}:no common alias".format(source))
                        continue
                    source_hit = dbs[source].lookup_improper(center, neighbors,
                                                              aliases)
                    if source_hit is not None:
                        atoms_order, matched, records, origin = source_hit
                        effective = numerical_source(
                            source, (matched, records, origin))
                        candidate = (effective, atoms_order,
                                     (matched, records, origin))
                        break
                    attempts.append("{}:absent".format(source))
            if candidate is not None:
                source, atoms_order, hit = candidate
                save_resolution("IMPROPER", atoms_order,
                                tuple(otype[x] for x in atoms_order), source, hit,
                                attempts)

        # A reaction-center bond or angle keeps its source force constant but
        # takes the equilibrium value measured in the input TS geometry, so a
        # force-constant fit to the TS Hessian is not distorted by the tension
        # of ground-state reference values. A key changes only when every
        # instance involves a custom atom; a key shared with the host region
        # keeps its source value. [CUSTOM_PARAMETERS] values and generated
        # seeds (already measured from the TS) are left as they are.
        ts_equilibria = []
        if opts.ts_equilibria:
            for section in ("BOND", "ANGLE"):
                for identity, entry in resolved[section].items():
                    row = entry["row"]
                    if row["source"] in ("custom", "ts_seed"):
                        continue
                    instances = [tuple(int(i) for i in inst.split(","))
                                 for inst in row["atom_indices"].split(";")]
                    if not all(any(atom_specs[i]["category"] == "custom"
                                   for i in ids) for ids in instances):
                        continue
                    mean, low, high = measured_equilibrium(section, instances,
                                                           coords)
                    key = key_for_types(section, identity)
                    records = output_sections[section][key]
                    # No hyphen in this note: Q2MM tells bond, angle and
                    # torsion rows apart by counting hyphens in the line.
                    note = "TS_GEOMETRY_EQ was {} n={}".format(
                        records[0][0][1], len(instances))
                    if len(instances) > 1:
                        note += " min {:.4f} max {:.4f}".format(low, high)
                    # Build new records: source records are shared with every
                    # other key that resolved to the same parameter.
                    output_sections[section][key] = [
                        ([numbers[0], "{:.4f}".format(mean)],
                         "; ".join(part for part in
                                   (note, " ".join(numbers[2:]), comment)
                                   if part),
                         raw)
                        for numbers, comment, raw in records]
                    row["values"] = " | ".join(
                        " ".join(numbers) for numbers, _, _ in
                        output_sections[section][key])
                    row["equilibrium"] = note
                    ts_equilibria.append(row)

        out_mol2 = os.path.abspath(opts.out + ".mol2")
        out_leap_mol2 = os.path.abspath(opts.out + ".leap.mol2")
        out_frcmod = os.path.abspath(opts.out + ".frcmod")
        out_atomtypes = os.path.abspath(opts.out + "_atomtypes.leap")
        out_report = os.path.abspath(opts.out + "_parameter_report.csv")
        out_seed_report = os.path.abspath(opts.out + "_ts_seed_report.csv")
        out_ts_fit_report = os.path.abspath(opts.out + "_ts_fit_report.csv")
        out_types = os.path.abspath(opts.out + "_atom_report.csv")
        out_leap = os.path.abspath(opts.out + ".leap.in")
        out_standalone_leap = os.path.abspath(opts.out + ".standalone.leap.in")
        out_prmtop = os.path.abspath(opts.out + ".prmtop")
        out_inpcrd = os.path.abspath(opts.out + ".inpcrd")
        out_standalone_prmtop = os.path.abspath(opts.out + ".standalone.prmtop")
        out_standalone_inpcrd = os.path.abspath(opts.out + ".standalone.inpcrd")

        selection = OrderedDict((idx, (otype[idx], types[idx]))
                                for idx in sorted(atom_tokens)
                                if otype[idx] != types[idx])
        write_mol2(out_mol2, lines, atom_tokens, atom_line_idx, selection)
        leap_mol2 = out_mol2
        if leap_restored_bonds:
            write_mol2_without_bonds(out_mol2, out_leap_mol2,
                                     leap_restored_bonds)
            leap_mol2 = out_leap_mol2
        title = "Source-resolved TSFF; priority: " + " > ".join(priority)
        if opts.allow_ts_seeds:
            title += "; flagged TS seeds enabled"
        if opts.ts_equilibria:
            title += "; reaction-center equilibria from TS geometry"
        total = frc_write(out_frcmod, title, output_sections)
        known_types = dict(gaff_type_defs)
        known_types.update(amber_type_defs)
        definitions = write_atomtypes(out_atomtypes, atom_tokens, types, otype,
                                      atom_specs, known_types)

        fieldnames = ["parameter_id", "section", "final_types", "lookup_types",
                      "source", "source_file", "status", "ts_role", "region",
                      "atom_indices", "values", "earlier_attempts",
                      "equilibrium"]
        with open(out_report, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
        seed_rows = [row for row in report_rows if row["source"] == "ts_seed"]
        with open(out_seed_report, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seed_rows)
        ts_fit_rows = [row for row in report_rows if row["ts_role"]]
        with open(out_ts_fit_report, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ts_fit_rows)
        with open(out_types, "w", newline="") as fh:
            fieldnames_atoms = ["index", "name", "input_type", "final_type",
                                "category", "aliases", "assignment", "confidence",
                                "charge"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames_atoms)
            writer.writeheader()
            for idx in sorted(atom_tokens):
                spec = atom_specs[idx]
                writer.writerow({
                    "index": idx, "name": atom_tokens[idx][1],
                    "input_type": types[idx], "final_type": otype[idx],
                    "category": spec["category"],
                    "aliases": ";".join("{}:{}".format(k, v)
                                        for k, v in spec["aliases"].items()),
                    "assignment": spec["assignment"],
                    "confidence": spec["confidence"],
                    "charge": atom_tokens[idx][8],
                })

        with open(out_leap, "w") as fh:
            fh.write("source {}\n".format(files["GAFF2"]))
            fh.write("source {}\n".format(files["FF19SB"]))
            fh.write("loadamberparams {}\n".format(out_frcmod))
            fh.write("mol = loadmol2 {}\n".format(leap_mol2))
            for other, hydrogen in leap_restored_bonds:
                hname = atom_tokens[hydrogen][1]
                oname = atom_tokens[other][1]
                fh.write("set mol.1.{} pert true\n".format(hname))
                fh.write("bond mol.1.{} mol.1.{}\n".format(oname, hname))
                fh.write("set mol.1.{} pert false\n".format(hname))
            fh.write("check mol\n")
            fh.write("saveamberparm mol {} {}\nquit\n"
                     .format(out_prmtop, out_inpcrd))
        with open(out_standalone_leap, "w") as fh:
            fh.write("loadamberparams {}\n".format(out_frcmod))
            fh.write("mol = loadmol2 {}\n".format(leap_mol2))
            for other, hydrogen in leap_restored_bonds:
                hname = atom_tokens[hydrogen][1]
                oname = atom_tokens[other][1]
                fh.write("set mol.1.{} pert true\n".format(hname))
                fh.write("bond mol.1.{} mol.1.{}\n".format(oname, hname))
                fh.write("set mol.1.{} pert false\n".format(hname))
            fh.write("check mol\n")
            fh.write("saveamberparm mol {} {}\nquit\n"
                     .format(out_standalone_prmtop,
                             out_standalone_inpcrd))

        sys.stderr.write("published library matched {} atoms ({} heavy)\n"
                         .format(len(published_types), len(heavy)))
        for unit, anchor_name, atom_count, heavy_count in template_match_counts:
            sys.stderr.write("ff19SB {}@{} matched {} atoms ({} heavy)\n"
                             .format(unit, anchor_name, atom_count, heavy_count))
        sys.stderr.write("wrote {} ({} atom types changed)\n"
                         .format(out_mol2, len(selection)))
        if leap_restored_bonds:
            sys.stderr.write("wrote {} (LEaP loading copy; {} TS bond(s) "
                             "restored after load)\n"
                             .format(out_leap_mol2,
                                     len(leap_restored_bonds)))
        sys.stderr.write("wrote {} ({} topology-specific parameters)\n"
                         .format(out_frcmod, total))
        if opts.ts_equilibria:
            sys.stderr.write("  {} bond and {} angle equilibrium values taken "
                             "from the TS geometry\n".format(
                                 sum(r["section"] == "BOND" for r in ts_equilibria),
                                 sum(r["section"] == "ANGLE" for r in ts_equilibria)))
        sys.stderr.write("wrote {} ({} atom types used by this topology)\n"
                         .format(out_atomtypes, len(definitions)))
        sys.stderr.write("wrote {} (parameter provenance)\n".format(out_report))
        sys.stderr.write("wrote {} ({} generated TS seeds to fit)\n"
                         .format(out_seed_report, len(seed_rows)))
        sys.stderr.write("wrote {} ({} TS terms to fit with Q2MM)\n"
                         .format(out_ts_fit_report, len(ts_fit_rows)))
        sys.stderr.write("wrote {} (atom typing and aliases)\n".format(out_types))
        sys.stderr.write("wrote {} (reproducible LEaP input)\n".format(out_leap))
        sys.stderr.write("wrote {} (standalone completeness check)\n"
                         .format(out_standalone_leap))

        if not opts.no_verify:
            returncode, errors, warnings, log = run_generated_tleap(out_leap,
                                                                     workdir)
            if returncode or errors is None or errors:
                sys.stderr.write("\ntleap validation failed:\n")
                sys.stderr.write(log[-4000:])
                return 1
            sys.stderr.write("\ntleap: Errors = {}; Warnings = {}\n"
                             .format(errors, warnings))
            validate_prmtop_bonds(out_prmtop, bonds)
            returncode, errors, warnings, log = run_generated_tleap(
                out_standalone_leap, workdir)
            if returncode or errors is None or errors:
                sys.stderr.write("\nstandalone tleap validation failed:\n")
                sys.stderr.write(log[-4000:])
                return 1
            validate_prmtop_bonds(out_standalone_prmtop, bonds)
            sys.stderr.write("standalone tleap: Errors = {}; Warnings = {}\n"
                             .format(errors, warnings))
    except ValueError as exc:
        sys.stderr.write("ERROR: {}\n".format(exc))
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
