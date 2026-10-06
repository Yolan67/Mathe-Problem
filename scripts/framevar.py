#!/usr/bin/env python3
"""Exact 604 frame variants: slices + core use frame g = R f (R in SO(3) random near identity, or random),
accepted iff all |g_ik| <= 1/sqrt2 (then the 604 is valid).  Writes runs/frames/fv_s{seed}.txt (unit vectors).
usage: framevar.py seed [maxangle_deg]"""
import sys
import numpy as np
seed = int(sys.argv[1]); amax = float(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(seed)
X = np.loadtxt('records/N604/config_float.txt')
r2 = 2 ** .5
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
def rot(axis, th):
    a = axis / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
for tries in range(100000):
    R = rot(rng.normal(size=3), np.radians(rng.uniform(0, amax)))
    g = f @ R.T
    if np.abs(g).max() <= 1 / r2 + 1e-12: break
else:
    sys.exit('no frame found')
# map: points whose T-part is in frame f (slices 496..591, core 592..603): w -> R w
Y = X.copy()
Y[496:604, 8:] = X[496:604, 8:] @ R.T
Y /= np.linalg.norm(Y, axis=1)[:, None]
G = Y @ Y.T; np.fill_diagonal(G, -9)
print('max cos', G.max(), 'max|g|', np.abs(g).max())
assert G.max() <= .5 + 1e-12
np.savetxt(f'runs/frames/fv_s{seed}.txt', Y, fmt='%.17g')
