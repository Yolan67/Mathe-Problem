#!/usr/bin/env python3
"""R^8 (+) T candidate framework with a mixed T-direction catalogue (octahedral, cube, cuboctahedral,
604-frame, icosahedral, dodecahedral, icosidodecahedral).  Norm-4 vectors:
  D8 part   : Z^8 norm-4 shell (±2e_s, (±1)^4 on any 4-subset), T = 0
  layers    : (±1)^3 on any triple of R^8  x  unit T-direction
  slices    : (±1,±1) on any pair, or ±sqrt2 e_s  x  sqrt2 * T-direction
  core      : 2 * T-direction
usage: icofw.py OUT [catalogue letters, default all: OCQFKIDJ]"""
import sys, itertools
import numpy as np
out = sys.argv[1]; cat = sys.argv[2] if len(sys.argv) > 2 else 'OCQFKIDJ'
r2 = 2 ** .5; phi = (1 + 5 ** .5) / 2
def uniq(vs):
    res = []
    for v in vs:
        v = np.array(v, float); v = v / np.linalg.norm(v)
        if not any(np.abs(v - w).max() < 1e-9 for w in res): res.append(v)
    return res
def cyc(v):
    return [v, (v[1], v[2], v[0]), (v[2], v[0], v[1])]
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
dirs = []
if 'O' in cat: dirs += [s * e for e in np.eye(3) for s in (1, -1)]
if 'C' in cat: dirs += [np.array(s) for s in itertools.product((1, -1), repeat=3)]
if 'Q' in cat: dirs += [np.array(v) for v in itertools.product((1, -1, 0), repeat=3) if sum(map(abs, v)) == 2]
if 'F' in cat: dirs += [s * x for x in f for s in (1, -1)]
if 'K' in cat: dirs += [s1 * f[a] + s2 * f[b] for a, b in itertools.combinations(range(3), 2) for s1 in (1, -1) for s2 in (1, -1)]
if 'I' in cat: dirs += [np.array(w) for s1 in (1, -1) for s2 in (1, -1) for w in cyc((0, s1, s2 * phi))]
if 'D' in cat: dirs += [np.array(w) for s1 in (1, -1) for s2 in (1, -1) for w in cyc((0, s1 / phi, s2 * phi))]
if 'J' in cat: dirs += [np.array(w) for s in itertools.product((1, -1), repeat=3) for w in cyc((s[0] * .5, s[1] * phi / 2, s[2] / (2 * phi)))]
D = np.array(uniq(dirs))
print('T directions', len(D), file=sys.stderr)
V = []
def add(u, w): V.append(np.concatenate([u, w]))
z3 = np.zeros(3)
for s in range(8):
    for g in (2, -2):
        u = np.zeros(8); u[s] = g; add(u, z3)
for Bk in itertools.combinations(range(8), 4):
    for sg in itertools.product((1, -1), repeat=4):
        u = np.zeros(8); u[list(Bk)] = sg; add(u, z3)
U3 = []
for t in itertools.combinations(range(8), 3):
    for sg in itertools.product((1, -1), repeat=3):
        u = np.zeros(8); u[list(t)] = sg; U3.append(u)
U2 = []
for p in itertools.combinations(range(8), 2):
    for sg in itertools.product((1, -1), repeat=2):
        u = np.zeros(8); u[list(p)] = sg; U2.append(u)
for s in range(8):
    for g in (r2, -r2):
        u = np.zeros(8); u[s] = g; U2.append(u)
for w in D:
    for u in U3: add(u, w)
    for u in U2: add(u, r2 * w)
    add(np.zeros(8), 2 * w)
V = np.array(V)
assert np.allclose((V ** 2).sum(1), 4)
n = len(V); print('candidates', n, file=sys.stderr)
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 4096):
        Gm = V[s0:s0 + 4096] @ V.T
        for i in range(Gm.shape[0]):
            row = np.nonzero(Gm[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
X = np.loadtxt('records/N604/config_float.txt')
kv = {tuple(k): i for i, k in enumerate(np.round(V * 1e6).astype(np.int64))}
init = [kv.get(tuple(k)) for k in np.round(X * 1e6).astype(np.int64)]
print('604 found', sum(i is not None for i in init), file=sys.stderr)
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
