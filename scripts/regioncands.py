#!/usr/bin/env python3
"""Region-code candidates: in sqrt8-scaled F2 coordinates (D11 roots = (±2,±2,0^9), norm 8), all vectors
with entries in {0,±1,±1/sqrt2}, norm 8 and at least five ±1 entries (types 8x1, 7x1+2xh, 6x1+4xh, 5x1+6xh).
All are compatible with the 220 D11 roots.  Count conflicts (ip>4) with the 384 layer points; keep those with
<= KMAX conflicts; write MIS graph of (384 layers + kept) with init = layers.
usage: regioncands.py KMAX OUT"""
import sys, itertools
import numpy as np
KMAX = int(sys.argv[1]); out = sys.argv[2]
r2 = 2 ** .5; h = 1 / r2
X = np.loadtxt('records/N604/config_float.txt')
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
def toF2(X):
    Y = np.zeros_like(X)
    for a in range(4):
        Y[:, 2 * a] = (X[:, 2 * a] + X[:, 2 * a + 1]) / r2; Y[:, 2 * a + 1] = (X[:, 2 * a] - X[:, 2 * a + 1]) / r2
    Y[:, 8:] = X[:, 8:] @ f.T
    return Y * r2          # norm-4 -> norm-8 scale
Y = toF2(X)
assert np.allclose((Y ** 2).sum(1), 8)
L = Y[112:496]; Dp = np.vstack([Y[:112], Y[496:]])
assert np.abs(np.sort(np.abs(Dp), 1)[:, -2:] - 2).max() < 1e-9
types = [(8, 0), (7, 2), (6, 4), (5, 6)]
kept = []; tot = 0
signs = {k: np.array(list(itertools.product((1., -1.), repeat=k))) for k in range(1, 12)}
for n1, nh in types:
    for S1 in itertools.combinations(range(11), n1):
        rest = [i for i in range(11) if i not in S1]
        for Sh in itertools.combinations(rest, nh):
            k = n1 + nh
            sg = signs[k]
            V = np.zeros((len(sg), 11))
            V[:, list(S1)] = sg[:, :n1]
            if nh: V[:, list(Sh)] = sg[:, n1:] * h
            C = (V @ L.T > 4 + 1e-9).sum(1)
            tot += len(V)
            sel = C <= KMAX
            if sel.any(): kept.append(V[sel])
kept = np.vstack(kept) if kept else np.zeros((0, 11))
if len(kept) == 0: kept = np.zeros((0, 11))
# remove the layer points themselves
key = {tuple(np.round(y * 1e6).astype(np.int64)) for y in L}
kept = np.array([v for v in kept if tuple(np.round(v * 1e6).astype(np.int64)) not in key])
print('total candidates', tot, 'kept (<=%d conflicts, not in 604)' % KMAX, len(kept), flush=True)
C = (kept @ L.T > 4 + 1e-9).sum(1) if len(kept) else np.array([])
print('conflict histogram', np.bincount(C.astype(int)) if len(C) else [])
V = np.vstack([L, kept]); n = len(V)
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = V[s0:s0 + 2048] @ V.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 4 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2)
open(out + '_init.txt', 'w').write('\n'.join(map(str, range(384))))
print('graph n', n)
