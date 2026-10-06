#!/usr/bin/env python3
"""Cap surgery on the complete Z[sqrt2] candidate set: remove all 604 points with <x,v> < thr, keep the rest,
re-pack the removed region with C-candidates compatible with all kept points (MIS).
usage: capsurg.py seed thr"""
import sys, subprocess
import numpy as np
seed = int(sys.argv[1]); thr = float(sys.argv[2])
rng = np.random.default_rng(seed)
X = np.loadtxt('records/N604/config_float.txt')                 # norm-4 scale
r2 = 2 ** .5
AB = np.fromfile('runs/zr2/C_ab.bin', dtype=np.int8).reshape(-1, 22).astype(float)
C = AB[:, :11] + AB[:, 11:] / r2
v = rng.normal(size=11); v /= np.linalg.norm(v)
h = X @ v / 2
K = X[h >= thr]; Rm = X[h < thr]
ch = C @ v / 2
cand = C[ch < thr + 0.05]
ok = np.ones(len(cand), bool)
for s0 in range(0, len(cand), 50000):
    ok[s0:s0 + 50000] = ((cand[s0:s0 + 50000] @ K.T) <= 2 + 1e-9).all(1)
cand = cand[ok]; n = len(cand)
out = f'/tmp/claude-0/cap_{seed}'
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = cand[s0:s0 + 2048] @ cand.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo)
key = {tuple(np.round(x * 1e6).astype(np.int64)): i for i, x in enumerate(cand)}
init = [key.get(tuple(np.round(x * 1e6).astype(np.int64))) for x in Rm]
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
r = subprocess.run(['./misg', out + '.graph', '-i', out + '_init.txt', '-o', out + '.sol', '-T', '90', '-q', '1'], capture_output=True, text=True)
best = int(r.stdout.split()[-1])
print(f'seed {seed} removed {len(Rm)} kept {len(K)} candidates {n} MIS {best} -> total {len(K) + best}', flush=True)
if len(K) + best > 604:
    sol = [int(l) for l in open(out + '.sol')]
    np.savetxt(f'runs/half/cap_{seed}_N{len(K) + best}.txt', np.vstack([K, cand[sol]]), fmt='%.17g')
