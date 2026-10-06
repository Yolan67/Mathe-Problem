#!/usr/bin/env python3
"""For a 12-dim configuration: find 24-cells supported on coordinate 4-sets B, report lightness of B for the other
points, and write 841-attempt inputs (other points + 25 random points near R^B) for the lightest B.
usage: light24.py CONFIG.txt OUTPREFIX [ntop]"""
import sys, itertools
import numpy as np
X = np.loadtxt(sys.argv[1]); X /= np.linalg.norm(X, axis=1)[:, None]
out = sys.argv[2]; ntop = int(sys.argv[3]) if len(sys.argv) > 3 else 2
res = []
for B in itertools.combinations(range(12), 4):
    B = list(B); rest = [i for i in range(12) if i not in B]
    inB = np.abs(X[:, rest]).max(1) < 1e-9
    if inB.sum() != 24: continue
    Y = X[~inB]
    m = (Y[:, B] ** 2).sum(1)
    res.append((m.sum() / len(Y), np.sqrt(m.max()), (np.sqrt(m) > 0.5 + 1e-9).sum(), B))
res.sort()
for r in res[:6]: print('mass %.4f  max|x_B| %.4f  #>1/2 %d  B=%s' % r)
print('24-cells found:', len(res))
rng = np.random.default_rng(1)
for k, (_, _, _, B) in enumerate(res[:ntop]):
    rest = [i for i in range(12) if i not in B]
    inB = np.abs(X[:, rest]).max(1) < 1e-9
    Y = X[~inB]
    Z = rng.normal(size=(25, 12)); Z[:, rest] *= 0.1
    Z /= np.linalg.norm(Z, axis=1)[:, None]
    np.savetxt(f'{out}_B{k}.txt', np.vstack([Y, Z]), fmt='%.17g')
