#!/usr/bin/env python3
"""Invariants of configurations (norm-4 scale or unit scale auto-detected): contacts, ip distribution.
usage: inv604.py file.txt|graph.npy:sol ..."""
import sys
import numpy as np
def load(a):
    if ':' in a:
        npy, sol = a.split(':')
        V = np.load(npy); idx = [int(l) for l in open(sol)]
        return V[idx]
    return np.loadtxt(a)
for a in sys.argv[1:]:
    P = load(a)
    P = P / np.linalg.norm(P, axis=1)[:, None]
    G = P @ P.T; np.fill_diagonal(G, -9)
    iu = np.triu_indices(len(P), 1); g = G[iu]
    c = (np.abs(g - .5) < 1e-7).sum()
    vals, cnt = np.unique(np.round(g, 5), return_counts=True)
    top = sorted(zip(vals.tolist(), cnt.tolist()))[-5:]
    anti = sum(1 for i in range(len(P)) if (np.abs(P + P[i]).max(1) < 1e-7).any())
    print(f'{a}: N={len(P)} maxcos={g.max():.12f} contacts={c} antipodal_pts={anti} distinct_ips={len(vals)} top={top}')
