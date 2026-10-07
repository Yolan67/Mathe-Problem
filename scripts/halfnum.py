#!/usr/bin/env python3
"""Numerical half-space surgery input: order = equator (x_c = 0) + lower half (x_c < 0) [both to be fixed]
+ upper half (x_c > 0) [free] + one new point at the deepest hole of the upper half-space.
usage: halfnum.py CONFIG c OUT   -> writes OUT (unit vectors), prints nfix"""
import sys
import numpy as np
X = np.loadtxt(sys.argv[1]); X /= np.linalg.norm(X, axis=1)[:, None]
c = int(sys.argv[2]); out = sys.argv[3]
E = X[np.abs(X[:, c]) < 1e-9]; L = X[X[:, c] < -1e-9]; U = X[X[:, c] > 1e-9]
rng = np.random.default_rng(1)
best = None
for rep in range(40):
    Y = rng.normal(size=(50000, X.shape[1])); Y[:, c] = np.abs(Y[:, c])
    Y /= np.linalg.norm(Y, axis=1)[:, None]
    m = (Y @ X.T).max(1)
    i = m.argmin()
    if best is None or m[i] < best[0]: best = (m[i], Y[i])
y = best[1].copy()
for it in range(3000):   # subgradient descent of max cos over the upper half-space
    G = X @ y; k = G.argmax()
    y -= 0.002 * (X[k] - G[k] * y); y[c] = max(y[c], 0); y /= np.linalg.norm(y)
print('equator', len(E), 'lower', len(L), 'upper', len(U), 'new point maxcos', (X @ y).max(), 'height', y[c])
Z = np.vstack([E, L, U, y[None, :]])
np.savetxt(out, Z, fmt='%.17g')
print('nfix', len(E) + len(L))
