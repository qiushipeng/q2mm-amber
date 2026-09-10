#!/usr/bin/env python3
"""
Recover swarm data from a q2mm-amber SWARM run's log and particle dirs.

Two uses:

1. Runs that predate the history dump. `loop.py._dump_swarm_history` now
   writes `hybrid_opt_history.bin` at the end of every SWARM run, but runs
   made before that change have no pickle at all. This is the only way to
   get anything out of them.

2. The gbest trajectory, which the pickle does NOT contain. `recorder()`
   appends X and Y but *overwrites* best_x/best_y, so the history keeps only
   the final best position. The verbose log records the best position at
   every iteration, so the two sources are complementary.

What it reads:

  * gbest trajectory  -- the "Iter: N, Best fit: Y at [X]" lines root.log
                         gets when PSO_DE runs with verbose=True
  * taper schedule    -- the "cp: ... w: ... cg ..." lines
  * last particle X   -- swarm_particles/p_NNN/<ff>.frcmod, each holding
                         that particle's LAST EVALUATED position

Note this parses the frcmod q2mm-amber *wrote*, whose columns are padded
wider than a hand-made frcmod's; the fixed offset in parse_frcmod is safe
for that layout but not for arbitrary input.

Never recoverable from disk: per-iteration X and Y for the whole swarm (use
the pickle) and pbest, which is never recorded anywhere.

Usage:
    python3 extract_swarm.py <run_dir> [-o out.npz]
"""

import argparse
import os
import re
import sys

import numpy as np

ITER_RE = re.compile(r"^Iter:\s*(\d+),\s*Best fit:\s*(\S+)\s*at\s*\[", re.M)
TAPER_RE = re.compile(r"^cp:\s*(\S+)\s+w:\s*(\S+)\s+cg\s+(\S+)\s*$", re.M)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# frcmod sections whose numeric columns are optimizable parameters, in the
# order data_structs writes them. BOND: force const, equil. ANGLE: same.
FRCMOD_SECTIONS = ("BOND", "ANGLE", "DIHE", "IMPROPER")


def parse_gbest(log_path):
    """Return (iters, gbest_y, gbest_x) from the verbose Iter: lines."""
    with open(log_path, "r", errors="replace") as fh:
        text = fh.read()

    iters, ys, xs = [], [], []
    for m in ITER_RE.finditer(text):
        iters.append(int(m.group(1)))
        ys.append(float(m.group(2)))
        # the vector runs from the '[' to the matching ']'
        start = text.index("[", m.start())
        end = text.index("]", start)
        xs.append([float(v) for v in NUM_RE.findall(text[start + 1:end])])

    if not iters:
        return np.array([]), np.array([]), np.zeros((0, 0))

    width = len(xs[0])
    if any(len(x) != width for x in xs):
        bad = [i for i, x in enumerate(xs) if len(x) != width]
        raise ValueError(
            "inconsistent gbest_x width at iters {} (expected {})".format(
                [iters[i] for i in bad[:5]], width
            )
        )
    return np.array(iters), np.array(ys), np.array(xs)


def parse_taper(log_path):
    """Return (cp, w, cg) arrays from the hyperparameter log lines."""
    with open(log_path, "r", errors="replace") as fh:
        rows = [
            (float(a), float(b), float(c))
            for a, b, c in TAPER_RE.findall(fh.read())
        ]
    if not rows:
        return np.zeros((0, 3))
    return np.array(rows)


def parse_frcmod(path):
    """Pull the numeric parameter columns out of an Amber frcmod."""
    vals = []
    section = None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            head = line.strip().split()
            if head and head[0] in FRCMOD_SECTIONS:
                section = head[0]
                continue
            if not line.strip():
                section = None
                continue
            if section is None:
                continue
            # atom-type field is fixed width and may contain '-'; numbers
            # start after it.
            nums = NUM_RE.findall(line[11:] if len(line) > 11 else "")
            if section in ("BOND", "ANGLE"):
                vals.extend(float(v) for v in nums[:2])
            elif section in ("DIHE", "IMPROPER"):
                vals.extend(float(v) for v in nums[1:3])
    return vals


def parse_last_x(run_dir, n_params=None):
    """Return (particle_ids, X, cols) from each particle's final frcmod.

    A frcmod holds every parameter in the force field, but only the
    optimized subset differs between particles. `cols` indexes the columns
    that actually vary, which is what corresponds to the optimizer's X.
    """
    base = os.path.join(run_dir, "swarm_particles")
    if not os.path.isdir(base):
        return np.array([]), np.zeros((0, 0)), np.array([])

    ids, rows = [], []
    for name in sorted(os.listdir(base)):
        if not name.startswith("p_"):
            continue
        pdir = os.path.join(base, name)
        frcmods = [f for f in os.listdir(pdir) if f.endswith(".frcmod")]
        if not frcmods:
            continue
        vals = parse_frcmod(os.path.join(pdir, frcmods[0]))
        if vals:
            ids.append(int(name[2:]))
            rows.append(vals)

    if not rows:
        return np.array([]), np.zeros((0, 0)), np.array([])
    width = max(len(r) for r in rows)
    keep = [i for i, r in enumerate(rows) if len(r) == width]
    ids = [ids[i] for i in keep]
    full = np.array([rows[i] for i in keep])

    cols = np.flatnonzero(full.std(axis=0) > 1e-9)
    if n_params is not None and len(cols) != n_params:
        print(
            "warning: {} varying frcmod columns but optimizer used {} params; "
            "returning the full frcmod instead".format(len(cols), n_params),
            file=sys.stderr,
        )
        return np.array(ids), full, np.arange(full.shape[1])
    return np.array(ids), full[:, cols], cols


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", help="write results to this .npz")
    args = ap.parse_args()

    log = os.path.join(args.run_dir, "root.log")
    if not os.path.isfile(log):
        sys.exit("no root.log in {}".format(args.run_dir))

    iters, gbest_y, gbest_x = parse_gbest(log)
    taper = parse_taper(log)
    n_params = gbest_x.shape[1] if len(iters) else None
    pids, last_x, cols = parse_last_x(args.run_dir, n_params)

    print("run:            {}".format(args.run_dir))
    print("iterations:     {}".format(len(iters)))
    if len(iters):
        print("gbest_x width:  {} params".format(gbest_x.shape[1]))
        print("gbest_y:        {:.6f} -> {:.6f}".format(gbest_y[0], gbest_y[-1]))
        improved = int(np.sum(np.diff(gbest_y) < 0))
        print("gbest improved: {} times".format(improved))
    print("taper records:  {}".format(len(taper)))
    if len(last_x):
        print("last X:         {} particles x {} params".format(*last_x.shape))
        spread = last_x.std(axis=0)
        print("param spread:   min {:.4g}  median {:.4g}  max {:.4g}".format(
            spread.min(), float(np.median(spread)), spread.max()))
        collapsed = int(np.sum(spread < 1e-6))
        print("collapsed dims: {} / {}".format(collapsed, len(spread)))

    if args.out:
        np.savez(args.out, iters=iters, gbest_y=gbest_y, gbest_x=gbest_x,
                 taper=taper, particle_ids=pids, last_x=last_x,
                 last_x_cols=cols)
        print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
