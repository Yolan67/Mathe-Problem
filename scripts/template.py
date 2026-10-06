#!/usr/bin/env python3
"""Backbone templates for R^11 = R^S (+) R^T.

template.py --T 8,9,10 --groups "0,1;2,3;4,5;6,7" --out PREFIX [--noaxes ..] [--type0 any|none|unions]
Blocks: 4-subsets of {0..10}, pairwise intersections <= 2, |B∩T| <= 1, and blocks with a
T-point meet every group in <= 1 point.  Blocks inside S (type 0) are restricted by --type0:
  any    : any 4-subset of S
  unions : unions of two groups of size 2
  none   : no type-0 blocks
Maximum packing via misg (with weights equal).  Backbone = axes +-2e_s (s in S) +
sign vectors on blocks.  Writes PREFIX_blocks.txt and PREFIX_backbone.txt.
"""
import argparse, itertools, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--T', default='8,9,10')
ap.add_argument('--groups', default='0,1;2,3;4,5;6,7')
ap.add_argument('--type0', default='any')
ap.add_argument('--out', required=True)
ap.add_argument('--secs', type=float, default=20)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
T = set(int(x) for x in a.T.split(',') if x)
groups = [set(int(x) for x in g.split(',')) for g in a.groups.split(';') if g]
S = [i for i in range(11) if i not in T]
cands = []
for B in itertools.combinations(range(11), 4):
    sB = set(B)
    nt = len(sB & T)
    if nt > 1:
        continue
    if nt == 1:
        if any(len(sB & g) > 1 for g in groups):
            continue
    else:
        if a.type0 == 'none':
            continue
        if a.type0 == 'unions':
            gs = [g for g in groups if g <= sB]
            if not (len(gs) == 2 and all(len(g) == 2 for g in gs)):
                continue
    cands.append(B)
n = len(cands)
with open(a.out + '_pack.graph', 'wb') as f:
    np.array([n, 0], dtype=np.int32).tofile(f)
    for i, B in enumerate(cands):
        nb = [j for j, C in enumerate(cands) if j != i and len(set(B) & set(C)) > 2]
        np.array([len(nb)] + nb, dtype=np.int32).tofile(f)
subprocess.run(['./misg', a.out + '_pack.graph', '-o', a.out + '_pack.sol', '-T', str(a.secs), '-q', '1', '-s', str(a.seed)],
               stdout=subprocess.DEVNULL)
sol = [cands[int(l)] for l in open(a.out + '_pack.sol')]
pts = []
for s in S:
    for sg in (2, -2):
        v = [0] * 11; v[s] = sg; pts.append(v)
for B in sol:
    for sg in itertools.product((1, -1), repeat=4):
        v = [0] * 11
        for i, s in zip(B, sg): v[i] = s
        pts.append(v)
with open(a.out + '_blocks.txt', 'w') as f:
    for B in sol: f.write(' '.join(map(str, B)) + '\n')
with open(a.out + '_backbone.txt', 'w') as f:
    for v in pts: f.write(' '.join(map(str, v)) + '\n')
from collections import Counter
print(f"blocks {len(sol)} (type1 {sum(1 for B in sol if set(B)&T)}), backbone {len(pts)}")
