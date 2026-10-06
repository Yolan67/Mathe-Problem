#!/usr/bin/env python3
"""Norm-8 shell of Z^11 (D11-root frame of the 604): types (±2,±2,0^9) [220], (±1)^8 0^3 [42240],
(±2,(±1)^4,0^6) [73920].  Conflict iff ip > 4.  usage: shell8.py OUT types   (types subset of 'abc')"""
import sys, itertools
import numpy as np
out, types = sys.argv[1], sys.argv[2]
V = []
if 'a' in types:
    for i, j in itertools.combinations(range(11), 2):
        for s, t in itertools.product((2, -2), repeat=2):
            v = np.zeros(11, np.int8); v[i] = s; v[j] = t; V.append(v)
if 'b' in types:
    for S in itertools.combinations(range(11), 8):
        for sg in itertools.product((1, -1), repeat=8):
            v = np.zeros(11, np.int8); v[list(S)] = sg; V.append(v)
if 'c' in types:
    for i in range(11):
        rest = [k for k in range(11) if k != i]
        for S in itertools.combinations(rest, 4):
            for s0 in (2, -2):
                for sg in itertools.product((1, -1), repeat=4):
                    v = np.zeros(11, np.int8); v[i] = s0; v[list(S)] = sg; V.append(v)
V = np.array(V, np.int8); n = len(V)
print('vertices', n, file=sys.stderr)
np.save(out + '.npy', V)
Vf = V.astype(np.float32); m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 4096):
        G = Vf[s0:s0 + 4096] @ Vf.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 4.5)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
