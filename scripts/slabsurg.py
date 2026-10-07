#!/usr/bin/env python3
"""Slab surgery on the complete Z[sqrt2] candidate set C: keep only the equator E = {x : x_c = 0} of a config,
re-pack both caps (x_c != 0) with C-candidates compatible with E (MIS, seeded with the removed points).
usage: slabsurg.py CONFIG c OUT [T]"""
import sys, subprocess
import numpy as np
X = np.loadtxt(sys.argv[1]); X *= 2 / np.linalg.norm(X, axis=1)[:, None]
c = int(sys.argv[2]); out = sys.argv[3]; T = sys.argv[4] if len(sys.argv) > 4 else '300'
r2 = 2 ** .5
AB = np.fromfile('runs/zr2/C_ab.bin', dtype=np.int8).reshape(-1, 22).astype(float)
C = AB[:, :11] + AB[:, 11:] / r2
E = X[np.abs(X[:, c]) < 1e-9]; R = X[np.abs(X[:, c]) >= 1e-9]
cand = C[np.abs(C[:, c]) > 1e-9]
ok = np.ones(len(cand), bool)
for s0 in range(0, len(cand), 50000):
    ok[s0:s0 + 50000] = ((cand[s0:s0 + 50000] @ E.T) <= 2 + 1e-9).all(1)
cand = cand[ok]; n = len(cand)
print('equator', len(E), 'removed', len(R), 'candidates', n, flush=True)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = cand[s0:s0 + 2048] @ cand.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, flush=True)
key = {tuple(np.round(v * 1e6).astype(np.int64)): i for i, v in enumerate(cand)}
init = [key.get(tuple(np.round(v * 1e6).astype(np.int64))) for v in R]
print('removed points among candidates:', sum(i is not None for i in init), flush=True)
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
np.save(out + '.npy', cand)
for seed, use_init in [(1, True), (2, False)]:
    cmd = ['./misg', out + '.graph', '-o', out + f'_s{seed}.sol', '-T', T, '-q', '1', '-s', str(seed)]
    if use_init: cmd += ['-i', out + '_init.txt']
    r = subprocess.run(cmd, capture_output=True, text=True)
    best = int(r.stdout.split()[-1])
    print(f'MIS seed {seed} {"seeded" if use_init else "unseeded"}: {best}  (need > {len(R)})  total {len(E) + best}', flush=True)
    if len(E) + best > 604:
        sol = [int(l) for l in open(out + f'_s{seed}.sol')]
        np.savetxt(out + f'_N{len(E) + best}.txt', np.vstack([E, cand[sol]]), fmt='%.17g')
