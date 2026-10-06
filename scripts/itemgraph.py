#!/usr/bin/env python3
"""Weighted item graph for S-sign-symmetric configurations in R^8 (+) R^3.

Item = (support sigma in {0..7}, w-set):
  B  : |sigma|=4, w=0                       weight 16
  L  : |sigma|=3, w in {+d,-d}, |d|=1        weight 16
  P  : |sigma|=2, w (single), |w|^2=2        weight 4
  S1 : |sigma|=1, w (single), |w|^2=3        weight 2
  Z  : sigma empty, w (single), |w|=2        weight 1
(the 16 axes +-2e_s are compatible with all items and always present)
Conflict between items iff |sigma∩sigma'| + max <w,w'> > 2 (+1e-9), or same support
with incompatible w's.  Directions: orbits of base vectors under the octahedral group.
Writes PREFIX.graph (for misw), PREFIX_items.json and init from the 604.
"""
import itertools, json, sys
import numpy as np
r2 = 2 ** .5
pref = sys.argv[1]
bases = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (r2, 1, 1), (r2, 1 + r2, 1 - r2), (2, 1, 0), (2, 1, 1),
         (1 + r2, 1, 0), (1 + r2, 1 + r2, 1), (1, r2, 0), (r2, r2, 1)]
if len(sys.argv) > 2:
    bases = bases[:int(sys.argv[2])]
dirs = []
def addd(v):
    v = np.array(v, float); v /= np.linalg.norm(v)
    for w in dirs:
        if np.abs(v - w).max() < 1e-9: return
    dirs.append(v)
for b in bases:
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            addd([s[i] * b[p[i]] for i in range(3)])
# also the actual frame of the 604 (rotated copy) and its cuboctahedron
f = [np.array([1 / r2, .5, .5]), np.array([1 / r2, -.5, -.5]), np.array([0, 1 / r2, -1 / r2])]
for v in f: addd(v); addd(-v)
for i, j in itertools.combinations(range(3), 2):
    for s1, s2 in itertools.product((1, -1), repeat=2): addd(s1 * f[i] + s2 * f[j])
D = np.array(dirs)
print('directions', len(D), file=sys.stderr)
lines = []
for v in D:
    if not any(abs(abs(v @ w) - 1) < 1e-9 for w in lines): lines.append(v)
lines = np.array(lines)
items = []  # (kind, sigma tuple, list of w vectors, weight)
for B in itertools.combinations(range(8), 4): items.append(('B', B, [np.zeros(3)], 16))
for t in itertools.combinations(range(8), 3):
    for d in lines: items.append(('L', t, [d, -d], 16))
for P in itertools.combinations(range(8), 2):
    for d in D: items.append(('P', P, [r2 * d], 4))
for s in range(8):
    for d in D: items.append(('S1', (s,), [3 ** .5 * d], 2))
for d in D: items.append(('Z', (), [2 * d], 1))
n = len(items)
print('items', n, file=sys.stderr)
sig = [set(it[1]) for it in items]
Wl = [np.array(it[2]) for it in items]
adj = [[] for _ in range(n)]
kinds = [it[0] for it in items]
# vectorised by kind groups
for i in range(n):
    for j in range(i + 1, n):
        c = len(sig[i] & sig[j])
        m = (Wl[i] @ Wl[j].T).max()
        if c + m > 2 + 1e-9:
            adj[i].append(j); adj[j].append(i)
with open(pref + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for i in range(n):
        np.array([items[i][3], len(adj[i])] + adj[i], dtype=np.int32).tofile(fo)
json.dump([[it[0], list(it[1]), [w.tolist() for w in it[2]], it[3]] for it in items], open(pref + '_items.json', 'w'))
# init from 604 structure
init = []
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
blocks = [tuple(map(int, l.split())) for l in open('configs/layer30_blocks.txt')]
def find(kind, s, w0):
    for i, it in enumerate(items):
        if it[0] == kind and tuple(it[1]) == tuple(s) and any(np.abs(w - w0).max() < 1e-9 for w in it[2]):
            return i
    return None
for B in blocks[:6]: init.append(find('B', B, np.zeros(3)))
for B in blocks[6:]:
    k = B[-1] - 8; e = np.zeros(3); e[k] = 1
    init.append(find('L', B[:3], e))
for P in pairs:
    for v in f:
        for s in (1, -1): init.append(find('P', P, r2 * s * v))
for i, j in itertools.combinations(range(3), 2):
    for s1, s2 in itertools.product((1, -1), repeat=2): init.append(find('Z', (), r2 * (s1 * f[i] + s2 * f[j])))
print('init items', len(init), 'missing', sum(x is None for x in init), 'weight', sum(items[x][3] for x in init if x is not None) + 16, file=sys.stderr)
open(pref + '_init.txt', 'w').write('\n'.join(str(x) for x in init if x is not None))
