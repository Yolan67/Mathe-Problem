#!/usr/bin/env python3
"""Reverse slab surgery on the complete Z[sqrt2] candidate set C: keep both caps (x_c != 0) of a config fixed and
re-pack the whole equator (x_c = 0) by MIS over the C-candidates with x_c = 0 that are compatible with the caps.
usage: eqsurg.py CONFIG c OUT [T]"""
import sys, subprocess
import numpy as np
X = np.loadtxt(sys.argv[1]); X *= 2 / np.linalg.norm(X, axis=1)[:, None]
c = int(sys.argv[2]); out = sys.argv[3]; T = sys.argv[4] if len(sys.argv) > 4 else '120'
r2 = 2 ** .5
AB = np.fromfile('runs/zr2/C_ab.bin', dtype=np.int8).reshape(-1, 22).astype(float)
C = AB[:, :11] + AB[:, 11:] / r2
E = X[np.abs(X[:, c]) < 1e-9]; R = X[np.abs(X[:, c]) >= 1e-9]
cand = C[np.abs(C[:, c]) < 1e-9]
ok = np.ones(len(cand), bool)
for s0 in range(0, len(cand), 50000):
    ok[s0:s0 + 50000] = ((cand[s0:s0 + 50000] @ R.T) <= 2 + 1e-9).all(1)
cand = cand[ok]; n = len(cand)
key = {tuple(k): i for i, k in enumerate(np.round(cand * 1e6).astype(np.int64))}
init = [key.get(tuple(k)) for k in np.round(E * 1e6).astype(np.int64)]
print('equator', len(E), 'caps', len(R), 'equator candidates', n, 'equator points among them', sum(i is not None for i in init), flush=True)
np.save(out + '.npy', cand)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = cand[s0:s0 + 2048] @ cand.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, flush=True)
open(out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))
for seed, use_init in [(1, True), (2, False)]:
    cmd = ['./misg', out + '.graph', '-o', out + f'_s{seed}.sol', '-T', T, '-q', '1', '-s', str(seed)]
    if use_init: cmd += ['-i', out + '_init.txt']
    subprocess.run(cmd, stdout=subprocess.DEVNULL)
    sol = [int(t) for t in open(out + f'_s{seed}.sol').read().split()]
    tot = len(R) + len(sol)
    print(f'MIS seed {seed} {"seeded" if use_init else "unseeded"}: {len(sol)} (need > {len(E)})  total {tot}', flush=True)
    if tot > 604:
        np.savetxt(out + f'_N{tot}.txt', np.vstack([R, cand[sol]]), fmt='%.17g'); print('SAVED', out + f'_N{tot}.txt')
