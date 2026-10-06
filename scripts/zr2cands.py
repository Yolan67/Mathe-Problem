#!/usr/bin/env python3
"""Complete candidate set C = { a + b/sqrt2 : a,b in Z^11, |a|^2 + |b|^2/2 = 4, a.b = 0 } (all norm-4 vectors of
(1/sqrt2) Z[sqrt2]^11 in the standard frame); contains the exact 604.  Counts conflicts (ip > 2) with the 604,
keeps candidates with <= KMAX conflicts, writes MIS graph (604 + kept) with init = 604.
usage: zr2cands.py KMAX OUT"""
import sys, itertools
import numpy as np
KMAX = int(sys.argv[1]); out = sys.argv[2]
r2 = 2 ** .5
X = np.loadtxt('records/N604/config_float.txt')
def shell(n, dim=11):
    """all integer vectors of squared norm n in Z^dim"""
    res = []
    def rec(pos, rem, cur):
        if pos == dim:
            if rem == 0: res.append(cur.copy())
            return
        m = int(rem ** .5)
        for v in range(-m, m + 1):
            if (dim - pos - 1) == 0 and v * v != rem: continue
            cur[pos] = v; rec(pos + 1, rem - v * v, cur)
        cur[pos] = 0
    rec(0, n, np.zeros(dim, np.int64))
    return np.array(res) if res else np.zeros((0, dim), np.int64)
cands = []
for na in range(0, 5):
    nb = 8 - 2 * na
    A = shell(na); B = shell(nb)
    print('|a|^2', na, len(A), '|b|^2', nb, len(B), flush=True)
    for a in A:
        Bs = B[(B @ a) == 0]
        if len(Bs): cands.append(a[None, :] + Bs / r2)
C = np.vstack(cands)
assert np.allclose((C ** 2).sum(1), 4)
print('candidates', len(C), flush=True)
conf = np.zeros(len(C), np.int32)
for s0 in range(0, len(C), 100000):
    conf[s0:s0 + 100000] = (C[s0:s0 + 100000] @ X.T > 2 + 1e-9).sum(1)
key604 = {tuple(np.round(x * 1e6).astype(np.int64)) for x in X}
is604 = np.array([tuple(np.round(c * 1e6).astype(np.int64)) in key604 for c in C])
print('604 points found in C:', is604.sum())
h = np.bincount(conf[~is604])
print('conflict histogram (non-604):', h[:20].tolist())
keep = C[(~is604) & (conf <= KMAX)]
V = np.vstack([X, keep]); n = len(V)
print('graph n', n, flush=True)
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = V[s0:s0 + 2048] @ V.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2)
open(out + '_init.txt', 'w').write('\n'.join(map(str, range(604))))
