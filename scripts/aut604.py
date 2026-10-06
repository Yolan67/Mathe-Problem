#!/usr/bin/env python3
"""Automorphism group (orthogonal maps preserving the point set) of a configuration, by backtracking
over images of a greedily chosen basis, pruned by contact degree and inner products.
Writes runs/aut/<name>_perms.npy (permutations) and _mats.npy (11x11 matrices)."""
import sys, os
import numpy as np
fn = sys.argv[1]; name = sys.argv[2]
X = np.loadtxt(fn); X /= np.linalg.norm(X, axis=1)[:, None]
N, d = X.shape
G = X @ X.T
R = np.round(G * 1e6).astype(np.int64)
np.fill_diagonal(G, -9)
deg = (np.abs(G - .5) < 1e-7).sum(1)
# refine colour classes by multiset of inner products (rounded)
col = deg.copy()
for it in range(3):
    keys = [tuple(sorted(np.round(G[i], 5).tolist())) + (col[i],) for i in range(N)]
    u = {k: j for j, k in enumerate(sorted(set(keys)))}
    col = np.array([u[k] for k in keys])
print('colour classes:', len(set(col)), sorted(np.bincount(col).tolist())[:20], flush=True)
# basis: greedy from the smallest colour class
order = np.argsort(np.bincount(col)[col], kind='stable')
B = []
for i in order:
    M = X[B + [i]]
    if np.linalg.matrix_rank(M, tol=1e-8) == len(B) + 1: B.append(i)
    if len(B) == d: break
B = np.array(B)
XB = X[B]
key = {tuple(np.round(x * 1e6).astype(np.int64)): i for i, x in enumerate(X)}
sols = []
def rec(k, img):
    if k == d:
        g = np.linalg.solve(XB, X[img]).T     # g @ XB[j] = X[img[j]]  -> g = X[img]^T XB^{-T}
        Y = X @ g.T
        perm = []
        for y in Y:
            j = key.get(tuple(np.round(y * 1e6).astype(np.int64)))
            if j is None: return
            perm.append(j)
        sols.append((np.array(perm), g)); return
    b = B[k]
    for c in np.nonzero(col == col[b])[0]:
        if any(R[c, img[j]] != R[b, B[j]] for j in range(k)): continue
        if c in img: continue
        rec(k + 1, img + [c])
rec(0, [])
print('group order', len(sols), flush=True)
os.makedirs('runs/aut', exist_ok=True)
np.save(f'runs/aut/{name}_perms.npy', np.array([p for p, g in sols]))
np.save(f'runs/aut/{name}_mats.npy', np.array([g for p, g in sols]))
# orbits
P = np.array([p for p, g in sols])
seen = -np.ones(N, int); orbs = []
for i in range(N):
    if seen[i] >= 0: continue
    o = sorted(set(P[:, i].tolist())); orbs.append(o)
    for j in o: seen[j] = len(orbs) - 1
print('orbits:', sorted(len(o) for o in orbs))
