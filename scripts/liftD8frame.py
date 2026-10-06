#!/usr/bin/env python3
"""E8 in the D8 frame (norm 4: (±√2,±√2,0^6), (±√2/2)^8 even) with lifts into T=R^3.
Candidates:
  E8 roots (240, w=0)
  lifted spinors: (±√2/2)^7 on S\{q}, w = ±(√2/2) e_k      (norm 7/2+1/2)
  lifted spinors type 2: (±√2/2)^6 on S\{q,q'}, w with |w|^2=1, w=±(√2/2)e_k±(√2/2)e_l
  cross roots: ±√2 e_a ± √2 e_k ; T-roots (±√2,±√2) on T ; T-axes ±2e_k
  pair-frame type: (±1,±1)/... skipped
Builds misg graph; prints counts."""
import itertools, sys, numpy as np
r2 = 2 ** .5; h = r2 / 2
V = []
for i, j in itertools.combinations(range(8), 2):
    for s, t in itertools.product((1, -1), repeat=2):
        v = np.zeros(11); v[i] = s * r2; v[j] = t * r2; V.append(v)
for sg in itertools.product((1, -1), repeat=8):
    if np.prod(sg) == 1:
        v = np.zeros(11); v[:8] = h * np.array(sg); V.append(v)
nE8 = len(V)
for q in range(8):
    others = [i for i in range(8) if i != q]
    for sg in itertools.product((1, -1), repeat=7):
        for k in range(3):
            for s in (1, -1):
                v = np.zeros(11); v[others] = h * np.array(sg); v[8 + k] = s * h; V.append(v)
for q, q2 in itertools.combinations(range(8), 2):
    others = [i for i in range(8) if i not in (q, q2)]
    for sg in itertools.product((1, -1), repeat=6):
        for k, l in itertools.combinations(range(3), 2):
            for s, t in itertools.product((1, -1), repeat=2):
                v = np.zeros(11); v[others] = h * np.array(sg); v[8 + k] = s * h; v[8 + l] = t * h; V.append(v)
for a in range(8):
    for k in range(3):
        for s, t in itertools.product((1, -1), repeat=2):
            v = np.zeros(11); v[a] = s * r2; v[8 + k] = t * r2; V.append(v)
for k, l in itertools.combinations(range(3), 2):
    for s, t in itertools.product((1, -1), repeat=2):
        v = np.zeros(11); v[8 + k] = s * r2; v[8 + l] = t * r2; V.append(v)
for k in range(3):
    for s in (2, -2):
        v = np.zeros(11); v[8 + k] = s; V.append(v)
V = np.array(V); n = len(V)
assert np.allclose((V ** 2).sum(1), 4)
print('candidates', n, file=sys.stderr)
out = sys.argv[1]
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 4096):
        G = V[s0:s0 + 4096] @ V.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
