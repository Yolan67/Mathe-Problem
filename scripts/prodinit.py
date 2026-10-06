#!/usr/bin/env python3
"""Init for product-sphere tests: points (u,w), |u|^2=3/4, |w|^2=1/4 (unit vectors in R^11).
usage: prodinit.py OUT N seed [dirs: oct|cube|rand]"""
import sys, itertools
import numpy as np
out, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else 'rand'
rng = np.random.default_rng(seed)
if mode == 'oct': D = np.vstack([np.eye(3), -np.eye(3)])
elif mode == 'cube': D = np.array(list(itertools.product((1, -1), repeat=3))) / 3 ** .5
else: D = None
X = np.zeros((N, 11))
for i in range(N):
    u = rng.normal(size=8); u *= (3 / 4) ** .5 / np.linalg.norm(u)
    w = D[i % len(D)] if D is not None else rng.normal(size=3)
    w = w / np.linalg.norm(w) * .5 + (rng.normal(size=3) * 0.02 if D is not None else 0)
    w *= .5 / np.linalg.norm(w)
    X[i, :8] = u; X[i, 8:] = w
np.savetxt(out + '.txt', X, fmt='%.17g')
open(out + '_fib.txt', 'w').write('\n'.join(map(str, range(N))))
