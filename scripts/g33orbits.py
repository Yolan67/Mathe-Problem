#!/usr/bin/env python3
"""Find small G33-orbits in C^5 = R^10 via fixed spaces of random subgroups."""
import numpy as np, sys
G = np.load('/tmp/claude-0/g33_G10.npy').astype(np.float64)
n = len(G); rng = np.random.default_rng(1)
def fixspace(Ms):
    A = np.vstack([M - np.eye(10) for M in Ms])
    U, s, Vt = np.linalg.svd(A)
    return Vt[s.size - (s < 1e-6).sum():] if (s < 1e-6).any() else np.zeros((0, 10))
def orbit_size(x):
    Y = G @ x
    k = np.round(Y * 1e5).astype(np.int64)
    return len(np.unique(k, axis=0))
found = {}
for trial in range(3000):
    m = 1 + (trial % 3)
    Ms = [G[rng.integers(n)] for _ in range(m)]
    W = fixspace(Ms)
    if W.shape[0] == 0: continue
    x = rng.normal(size=W.shape[0]) @ W; x /= np.linalg.norm(x)
    sz = orbit_size(x)
    key = (sz, W.shape[0])
    if key not in found:
        found[key] = W
        print('orbit size', sz, 'fix dim (real)', W.shape[0], flush=True)
np.save('/tmp/claude-0/g33_orbittypes.npy', {k: v for k, v in found.items()}, allow_pickle=True)
