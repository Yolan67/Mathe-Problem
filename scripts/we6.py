#!/usr/bin/env python3
"""W(E6): generate group from simple reflections (6x6), build exterior powers, decompose."""
import numpy as np, itertools, sys
# E6 simple roots (standard 8-dim coords from E8), then project to 6-dim orthonormal basis
E8 = []
s = [
 [1,-1,0,0,0,0,0,0],[0,1,-1,0,0,0,0,0],[0,0,1,-1,0,0,0,0],[0,0,0,1,-1,0,0,0],[0,0,0,1,1,0,0,0],
 [-.5,-.5,-.5,-.5,-.5,-.5,-.5,-.5]]
# Use Bourbaki E6 simple roots inside E8: a1=1/2(1,-1,-1,-1,-1,-1,-1,1), a2=e1+e2, a3=e2-e1, a4=e3-e2, a5=e4-e3, a6=e5-e4
a = np.array([[.5,-.5,-.5,-.5,-.5,-.5,-.5,.5],[1,1,0,0,0,0,0,0],[-1,1,0,0,0,0,0,0],[0,-1,1,0,0,0,0,0],[0,0,-1,1,0,0,0,0],[0,0,0,-1,1,0,0,0]],float)
# orthonormal basis of span(a)
Q, _ = np.linalg.qr(a.T)
A6 = a @ Q   # simple roots in R^6
refl = [np.eye(6) - 2 * np.outer(r, r) / (r @ r) for r in A6]
# BFS generate group
def key(M): return tuple(np.round(M * 1e6).astype(np.int64).ravel())
G = [np.eye(6)]; seen = {key(G[0])}; frontier = [G[0]]
while frontier:
    new = []
    for M in frontier:
        for R in refl:
            P = R @ M; k = key(P)
            if k not in seen: seen.add(k); G.append(P); new.append(P)
    frontier = new
print('W(E6) order', len(G), file=sys.stderr)
G = np.array(G)
np.save('/tmp/claude-0/we6_G6.npy', G)
# exterior cube: basis e_i^e_j^e_k, i<j<k (20)
trip = list(itertools.combinations(range(6), 3))
def wedge3(M):
    R = np.zeros((20, 20))
    for c, (i, j, k) in enumerate(trip):
        sub = M[np.ix_([i, j, k], range(6))]
        for r, (p, q, s_) in enumerate(trip):
            R[r, c] = np.linalg.det(M[np.ix_([p, q, s_], [i, j, k])])
    return R
W3 = np.array([wedge3(M) for M in G])
rng = np.random.default_rng(1)
H = rng.normal(size=(20, 20)); H = H + H.T
P = np.einsum('gij,jk,glk->il', W3, H, W3) / len(G)
ev, U = np.linalg.eigh(P)
print('wedge3 invariant operator eigenvalues (grouped):', np.round(ev, 6), file=sys.stderr)
np.save('/tmp/claude-0/we6_W3.npy', W3); np.save('/tmp/claude-0/we6_U3.npy', U); np.save('/tmp/claude-0/we6_ev3.npy', ev)
