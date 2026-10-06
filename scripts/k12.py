#!/usr/bin/env python3
"""Coxeter-Todd lattice K12 as Eisenstein lattice; minimal vectors; G33 roots (orthogonal to a min vector)."""
import itertools, numpy as np, sys
w = np.exp(2j * np.pi / 3)
units = [w ** k * s for k in range(3) for s in (1, -1)]
th = w - w ** 2   # sqrt(-3)
V = []
# type A: theta*(u e_i - u w^k e_j)
for i, j in itertools.combinations(range(6), 2):
    for u in units:
        for k in range(3):
            x = np.zeros(6, complex); x[i] = th * u; x[j] = -th * u * w ** k; V.append(x)
# type B: all coords units congruent to eps mod theta (eps=±1), sum ≡ 0 mod 3
for eps in (1, -1):
    cls = [eps * w ** k for k in range(3)]
    for c in itertools.product(range(3), repeat=6):
        n = np.bincount(np.array(c), minlength=3)
        if n[0] % 3 == n[1] % 3 == n[2] % 3:
            V.append(np.array([cls[k] for k in c]))
V = np.array(V)
print('K12 min vectors', len(V), 'norms', np.unique(np.round((np.abs(V) ** 2).sum(1), 6)), file=sys.stderr)
R = np.concatenate([V.real, V.imag], axis=1)   # real 12-dim
G = R @ R.T; np.fill_diagonal(G, -99)
print('max real ip', G.max().round(6), '(cos', (G.max() / 6).round(6), ')', file=sys.stderr)
np.save('/tmp/claude-0/k12_min.npy', V)
v = V[0]
h = V @ v.conj()   # Hermitian inner products with v
orth = V[np.abs(h) < 1e-9]
print('orthogonal to v:', len(orth), file=sys.stderr)
vals = np.round(h, 6)
from collections import Counter
print('Hermitian ip value classes:', sorted(Counter(np.round(np.abs(h) ** 2, 4)).items()), file=sys.stderr)
np.save('/tmp/claude-0/g33_roots_c.npy', orth)
np.save('/tmp/claude-0/k12_v.npy', v)
