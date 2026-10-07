#!/usr/bin/env python3
"""Lift test: embed an R^11 config (norm 4) into R^12 (x_11 = 0) and find the largest set of compatible
Z[sqrt2]-candidates c = a + b/sqrt2 (a,b in Z^12, a.b = 0, |c|^2 = 4) with c_11 != 0 (MIS).
Tells whether the config is the equator of a larger R^12 kissing configuration.
usage: lift12.py CONFIG OUT [T]"""
import sys, subprocess
import numpy as np
X = np.loadtxt(sys.argv[1]); X *= 2 / np.linalg.norm(X, axis=1)[:, None]
out = sys.argv[2]; T = sys.argv[3] if len(sys.argv) > 3 else '300'
X12 = np.hstack([X, np.zeros((len(X), 1))])
r2 = 2 ** .5
def shell(n, dim):
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
keep = []; tot = 0
for na in range(0, 5):
    nb = 8 - 2 * na
    A = shell(na, 12); B = shell(nb, 12)
    for a in A:
        Bs = B[(B @ a) == 0]
        if not len(Bs): continue
        Cc = a[None, :] + Bs / r2
        Cc = Cc[np.abs(Cc[:, 11]) > 1e-9]
        tot += len(Cc)
        if len(Cc):
            ok = ((Cc @ X12.T) <= 2 + 1e-9).all(1)
            if ok.any(): keep.append(Cc[ok])
    print('|a|^2', na, 'done, candidates so far', tot, flush=True)
K = np.vstack(keep); n = len(K)
print('lift candidates (c_11 != 0, compatible):', n, flush=True)
np.save(out + '.npy', K)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for s0 in range(0, n, 2048):
        G = K[s0:s0 + 2048] @ K.T
        for i in range(G.shape[0]):
            row = np.nonzero(G[i] > 2 + 1e-9)[0]; row = row[row != s0 + i].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, flush=True)
for seed in (1, 2):
    subprocess.run(['./misg', out + '.graph', '-o', out + f'_s{seed}.sol', '-T', T, '-q', '1', '-s', str(seed)],
                   stdout=subprocess.DEVNULL)
    sol = [int(t) for t in open(out + f'_s{seed}.sol').read().split()]
    print('MIS seed', seed, ':', len(sol), ' -> R^12 config size', len(X) + len(sol), flush=True)
    S = np.vstack([X12, K[sol]])
    np.savetxt(out + f'_s{seed}_N{len(S)}.txt', S, fmt='%.15f')
