#!/usr/bin/env python3
"""Half-space surgery on the complete Z[sqrt2] candidate set C.  For axis c and config X: keep F = {x : x_c >= 0}
(or the reverse side), candidates = C-points with x_c < 0 compatible with all of F; MIS over candidates.
usage: halfsurg.py CONFIG c OUT"""
import sys, subprocess
import numpy as np
X = np.loadtxt(sys.argv[1]); X *= 2 / np.linalg.norm(X, axis=1)[:, None]   # norm-4 scale
c = int(sys.argv[2]); out = sys.argv[3]
r2 = 2 ** .5
AB = np.fromfile('runs/zr2/C_ab.bin', dtype=np.int8).reshape(-1, 22).astype(float)
C = AB[:, :11] + AB[:, 11:] / r2
F = X[X[:, c] >= -1e-9]; L = X[X[:, c] < -1e-9]
cand = C[C[:, c] < -1e-9]
ok = np.ones(len(cand), bool)
for s0 in range(0, len(cand), 50000):
    ok[s0:s0 + 50000] = ((cand[s0:s0 + 50000] @ F.T) <= 2 + 1e-9).all(1)
cand = cand[ok]
print('fixed', len(F), 'removed', len(L), 'candidates', len(cand), flush=True)
n = len(cand)
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = cand[s0:s0 + 2048] @ cand.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo)
key = {tuple(np.round(v * 1e6).astype(np.int64)): i for i, v in enumerate(cand)}
init = [key.get(tuple(np.round(v * 1e6).astype(np.int64))) for v in L]
print('removed points present among candidates:', sum(i is not None for i in init))
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
np.save(out + '.npy', cand)
r = subprocess.run(['./misg', out + '.graph', '-i', out + '_init.txt', '-o', out + '.sol', '-T', '120', '-q', '1'], capture_output=True, text=True)
print('MIS (seeded):', r.stdout.strip(), ' need >', len(L))
r = subprocess.run(['./misg', out + '.graph', '-o', out + '_b.sol', '-T', '120', '-q', '1', '-s', '2'], capture_output=True, text=True)
print('MIS (unseeded):', r.stdout.strip())
