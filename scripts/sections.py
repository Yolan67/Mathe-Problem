#!/usr/bin/env python3
"""Largest hyperplane sections of a kissing configuration Y in R^d: every section Y ∩ u^perp is a kissing
configuration in R^(d-1).  Random (d-1)-subsets (biased toward the current best section) -> normal u -> count.
usage: sections.py CONFIG [samples] [tol] [OUT]"""
import sys
import numpy as np
Y = np.loadtxt(sys.argv[1]); Y /= np.linalg.norm(Y, axis=1)[:, None]
n, d = Y.shape
S = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-7
out = sys.argv[4] if len(sys.argv) > 4 else None
rng = np.random.default_rng(0)
best = {}  # frozenset(indices) -> normal
top = 0; topset = None
for s in range(S):
    if topset is not None and rng.random() < 0.5:
        pool = np.array(sorted(topset)); idx = rng.choice(pool, d - 1, replace=False)
    else:
        idx = rng.choice(n, d - 1, replace=False)
    A = Y[idx]
    _, sv, Vt = np.linalg.svd(A)
    if sv[-1] < 1e-6: continue
    u = Vt[-1]
    on = np.nonzero(np.abs(Y @ u) < tol)[0]
    key = frozenset(on.tolist())
    if key not in best:
        best[key] = u
        if len(on) > top:
            top = len(on); topset = key
            print(f'sample {s}: new best section {top}', flush=True)
sizes = sorted((len(k) for k in best), reverse=True)
print('distinct sections found', len(best), 'top sizes', sizes[:15])
if out and topset is not None:
    u = best[topset]
    Z = Y[sorted(topset)]
    # coordinates in an orthonormal basis of u^perp
    Q, _ = np.linalg.qr(np.hstack([u[:, None], rng.normal(size=(d, d - 1))]))
    B = Q[:, 1:]
    np.savetxt(out, Z @ B, fmt='%.17g')
    G = (Z @ B) @ (Z @ B).T; np.fill_diagonal(G, -2)
    print('section saved', out, 'size', len(Z), 'max cos', G.max())
