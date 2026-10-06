#!/usr/bin/env python3
"""Build explicit R^11 points from piece parameters and check full Gram independently."""
import sys, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('gp', 'scripts/g33pieces.py'); gp = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ['x']; spec.loader.exec_module(gp); sys.argv = _a
nR, nP, poles = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
par = np.load(sys.argv[4])
kinds = ['R'] * nR + ['P0'] * nP
v = np.load('/tmp/claude-0/k12_v.npy'); vn = v / np.linalg.norm(v)
M = np.eye(6, dtype=complex) - np.outer(vn, vn.conj())
U, s, Wh = np.linalg.svd(M); Bc = U[:, :5]
pts = []
for k, (phi, t) in zip(kinds, par.reshape(-1, 2)):
    lam = np.cos(t) * (6 / gp.sq[k]) ** .5; hh = 6 ** .5 * np.sin(t)
    for x in gp.B[k]:
        z = (np.exp(1j * phi) * lam * x) @ Bc.conj()   # 5 complex coords
        pts.append(np.concatenate([z.real, z.imag, [hh]]))
if poles:
    for sgn in (1, -1): pts.append(np.array([0] * 10 + [sgn * 6 ** .5]))
X = np.array(pts)
print('N', len(X), 'norms', np.unique(np.round((X ** 2).sum(1), 6)))
G = X @ X.T; np.fill_diagonal(G, -99)
print('max ip %.9f (must be <= 3)' % G.max(), ' max cos %.9f' % (G.max() / 6))
np.save(sys.argv[4].replace('.npy', '_pts.npy'), X)
