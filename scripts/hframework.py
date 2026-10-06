#!/usr/bin/env python3
"""Candidate set for R^11 = H1 (+) H2 (+) T, H1 = coords 0..3, H2 = coords 4..7, T = coords 8..10.

Pure parts: 24-cells 2*2T in H1, H2 ; core = cuboctahedron in the 604 frame f (12).
Mixed vectors (p1,p2,p3) with squared factor norms
  (2,2,0): p1 in sqrt2*D, p2 in sqrt2*D         (D = dual 24-cell unit directions (±1,±1,0,0)/sqrt2)
  (2,0,2),(0,2,2): sqrt2*D x sqrt2*{±f_i}
  (2,1,1),(1,2,1): sqrt2*D x U x T1     (U = 2T unit directions; T1 = unit T-directions catalogue)
  (1,1,2): U x U x sqrt2*{±f_i}
All vectors have squared norm 4; conflict iff ip > 2 (+1e-9).  Writes graph + init(604).
"""
import itertools, sys
import numpy as np
r2 = 2 ** .5
out = sys.argv[1]
tvar = sys.argv[2] if len(sys.argv) > 2 else 'full'
def uniq(vs):
    res = []
    for v in vs:
        v = np.array(v, float)
        if not any(np.abs(v - w).max() < 1e-9 for w in res): res.append(v)
    return np.array(res)
# 2T unit quaternions
U = [v for v in itertools.product((1, -1, 0), repeat=4) if sum(abs(x) for x in v) == 1]
U += [tuple(s / 2 for s in sg) for sg in itertools.product((1, -1), repeat=4)]
U = uniq(U)                                   # 24
D = uniq([np.array(p) / r2 for p in set(itertools.permutations((1, 1, 0, 0))) for p in [p]] +
         [np.array(p) * np.array(sg) / r2 for p in set(itertools.permutations((1, 1, 0, 0))) for sg in itertools.product((1, -1), repeat=4)])
D = uniq([d for d in D if abs(np.linalg.norm(d) - 1) < 1e-9])   # 24
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
F6 = np.vstack([f, -f])
core = uniq([r2 * (s1 * f[i] + s2 * f[j]) for i, j in itertools.combinations(range(3), 2) for s1 in (1, -1) for s2 in (1, -1)])
E6 = np.vstack([np.eye(3), -np.eye(3)])
cube_std = np.array([np.array(s) / 3 ** .5 for s in itertools.product((1, -1), repeat=3)])
cube_f = np.array([np.array(s) @ f / 3 ** .5 for s in itertools.product((1, -1), repeat=3)])
cubo_std = uniq([np.array(v) / r2 for v in itertools.product((1, -1, 0), repeat=3) if sum(abs(x) for x in v) == 2])
T1 = uniq(list(E6) + list(F6) + list(cube_std) + list(cube_f) + ([] if tvar == 'small' else list(cubo_std) + list(core / 2)))
print('U', len(U), 'D', len(D), 'T1', len(T1), 'core', len(core), file=sys.stderr)
V = []
def add(p1, p2, p3): V.append(np.concatenate([p1, p2, p3]))
z4 = np.zeros(4); z3 = np.zeros(3)
for u in U: add(2 * u, z4, z3); add(z4, 2 * u, z3)
for c in core: add(z4, z4, c)
for d1 in D:
    for d2 in D: add(r2 * d1, r2 * d2, z3)
for d in D:
    for g in F6: add(r2 * d, z4, r2 * g); add(z4, r2 * d, r2 * g)
for d in D:
    for u in U:
        for t in T1:
            add(r2 * d, u, t); add(u, r2 * d, t)
for u1 in U:
    for u2 in U:
        for g in F6: add(u1, u2, r2 * g)
V = np.array(V)
assert np.allclose((V ** 2).sum(1), 4)
key = np.round(V * 1e6).astype(np.int64)
_, idx = np.unique(key, axis=0, return_index=True)
V = V[np.sort(idx)]
n = len(V)
print('candidates', n, file=sys.stderr)
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = V[s0:s0 + 2048] @ V.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
X = np.loadtxt('records/N604/config_float.txt')
kv = {tuple(k): i for i, k in enumerate(np.round(V * 1e6).astype(np.int64))}
init = [kv.get(tuple(k)) for k in np.round(X * 1e6).astype(np.int64)]
print('604 found', sum(i is not None for i in init), file=sys.stderr)
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
