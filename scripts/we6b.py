#!/usr/bin/env python3
"""Decompose wedge^3 of the W(E6) reflection rep (vectorised); save irreducible pieces."""
import numpy as np, itertools, sys
G = np.load('/tmp/claude-0/we6_G6.npy')   # (51840,6,6)
trip = list(itertools.combinations(range(6), 3))
T = np.array(trip)
def wedge3_batch(Ms):
    # R[r,c] = det(M[rows_r][:, cols_c])
    A = Ms[:, T[:, None, :, None], T[None, :, None, :]]   # (b,20,20,3,3)
    return np.linalg.det(A)
W3 = np.concatenate([wedge3_batch(G[i:i + 2000]) for i in range(0, len(G), 2000)])
print('W3', W3.shape, file=sys.stderr)
rng = np.random.default_rng(1)
H = rng.normal(size=(20, 20)); H = H + H.T
P = np.einsum('gij,jk,glk->il', W3, H, W3) / len(G)
ev, U = np.linalg.eigh(P)
print(np.round(ev, 5))
np.save('/tmp/claude-0/we6_W3.npy', W3.astype(np.float32)); np.save('/tmp/claude-0/we6_U3.npy', U); np.save('/tmp/claude-0/we6_ev3.npy', ev)
