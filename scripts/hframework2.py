#!/usr/bin/env python3
"""Complete binary-octahedral (2O) framework: R^11 = H1 (+) H2 (+) T, components in 2O directions
(U = 2T units, D = dual 24-cell units; O = U u D, 48 directions) in every quaternionic slot.
Squared factor-norm patterns (norm 4 total):
  (4,0,0),(0,4,0): 2*O ; (0,0,4): 2*T2
  (2,2,0): sqrt2*O x sqrt2*O
  (2,0,2),(0,2,2): sqrt2*O x sqrt2*T2
  (2,1,1),(1,2,1): sqrt2*O x O x T1
  (1,1,2): O x O x sqrt2*T2
  (3,1,0)?? not included (norm-3 single quaternion component is not a 2O multiple)
T1 (unit T-directions for norm-1 T parts) and T2 (unit directions for norm-2/4 T parts) are catalogues.
usage: hframework2.py OUT [t1set] [t2set]  sets from letters O (std axes) F (604 frame) C (cube) K (core dirs)"""
import itertools, sys
import numpy as np
r2 = 2 ** .5
out = sys.argv[1]; t1s = sys.argv[2] if len(sys.argv) > 2 else 'OF'; t2s = sys.argv[3] if len(sys.argv) > 3 else 'FK'
unitslot = sys.argv[4] if len(sys.argv) > 4 else 'O'
def uniq(vs):
    res = []
    for v in vs:
        v = np.array(v, float)
        if not any(np.abs(v - w).max() < 1e-9 for w in res): res.append(v)
    return np.array(res)
U = [v for v in itertools.product((1, -1, 0), repeat=4) if sum(abs(x) for x in v) == 1]
U += [tuple(s / 2 for s in sg) for sg in itertools.product((1, -1), repeat=4)]
U = uniq(U)
D = uniq([np.array(p, float) * np.array(sg) / r2 for p in set(itertools.permutations((1, 1, 0, 0))) for sg in itertools.product((1, -1), repeat=4)])
O = np.vstack([U, D]); assert len(O) == 48
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
def cat(s):
    L = []
    if 'O' in s: L += list(np.vstack([np.eye(3), -np.eye(3)]))
    if 'F' in s: L += list(np.vstack([f, -f]))
    if 'C' in s: L += [np.array(v) / 3 ** .5 for v in itertools.product((1, -1), repeat=3)]
    if 'K' in s: L += [(s1 * f[a] + s2 * f[b]) / r2 for a, b in itertools.combinations(range(3), 2) for s1 in (1, -1) for s2 in (1, -1)]
    if 'Q' in s: L += [np.array(v) / r2 for v in itertools.product((1, -1, 0), repeat=3) if sum(map(abs, v)) == 2]
    return uniq(L)
T1 = cat(t1s); T2 = cat(t2s)
print('O', len(O), 'T1', len(T1), 'T2', len(T2), file=sys.stderr)
V = []
z4 = np.zeros(4); z3 = np.zeros(3)
def add(a, b, c): V.append(np.concatenate([a, b, c]))
for u in O: add(2 * u, z4, z3); add(z4, 2 * u, z3)
for t in T2: add(z4, z4, 2 * t)
for a in O:
    for b in O: add(r2 * a, r2 * b, z3)
for a in O:
    for t in T2: add(r2 * a, z4, r2 * t); add(z4, r2 * a, r2 * t)
W = O if unitslot == 'O' else U
for a in O:
    for b in W:
        for t in T1: add(r2 * a, b, t); add(b, r2 * a, t)
for a in W:
    for b in W:
        for t in T2: add(a, b, r2 * t)
V = np.array(V)
assert np.allclose((V ** 2).sum(1), 4)
key = np.round(V * 1e6).astype(np.int64)
_, idx = np.unique(key, axis=0, return_index=True)
V = V[np.sort(idx)]; n = len(V)
print('candidates', n, file=sys.stderr)
np.save(out + '.npy', V)
Vf = V.astype(np.float64); m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = Vf[s0:s0 + 2048] @ Vf.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
X = np.loadtxt(sys.argv[5]) * 2 if len(sys.argv) > 5 else np.loadtxt('records/N604/config_float.txt')
kv = {tuple(k): i for i, k in enumerate(np.round(V * 1e6).astype(np.int64))}
init = [kv.get(tuple(k)) for k in np.round(X * 1e6).astype(np.int64)]
print('604 found', sum(i is not None for i in init), file=sys.stderr)
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
