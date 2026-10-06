#!/usr/bin/env python3
"""840-point kissing arrangements in R^12 = R^6 (+) R^6 (Takhanov-Assylbekov-Yun structure):
60 vectors on each R^6 (±2e_i + a '48-system') and 720 bridge vectors from the 1-factorization of K6.
Builds the classical member; with --deform seed, perturbs both 48-systems and writes an input with the
other 744 points first (fixed) for re-optimisation.  Output scale: unit vectors."""
import sys, itertools
import numpy as np
# 1-factorization of K6 on {0..5}: GK(6) standard
F = [[(0, 5), (1, 4), (2, 3)], [(1, 5), (2, 0), (3, 4)], [(2, 5), (3, 1), (4, 0)], [(3, 5), (4, 2), (0, 1)], [(4, 5), (0, 3), (1, 2)]]
assert len({frozenset(e) for f in F for e in f}) == 15
def block48(f, off):
    V = []
    for (a, b), (c, d) in itertools.combinations(f, 2):
        for sg in itertools.product((1, -1), repeat=4):
            v = np.zeros(12); v[[a + off, b + off, c + off, d + off]] = sg; V.append(v)
    return V
V = []
for i in range(12):
    for s in (2, -2):
        v = np.zeros(12); v[i] = s; V.append(v)
L48 = block48(F[0], 0); R48 = block48(F[0], 6)
br = []
for j in range(5):
    for (a, b) in F[j]:
        for (c, d) in F[j]:
            for sg in itertools.product((1, -1), repeat=4):
                v = np.zeros(12); v[[a, b, c + 6, d + 6]] = sg; br.append(v)
V = np.array(V + br + L48 + R48) / 2
G = V @ V.T; np.fill_diagonal(G, -9)
print('N', len(V), 'max ip', G.max())
np.savetxt('runs/c840/c840.txt', V, fmt='%.17g')
if len(sys.argv) > 2 and sys.argv[1] == '--deform':
    seed = int(sys.argv[2]); sig = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    rng = np.random.default_rng(seed)
    W = V.copy()
    W[744:] += rng.normal(size=(96, 12)) * sig
    W[744:744 + 48, 6:] = 0; W[744 + 48:, :6] = 0      # keep each 48-system in its own R^6
    W /= np.linalg.norm(W, axis=1)[:, None]
    np.savetxt(f'runs/c840/def_s{seed}.txt', W, fmt='%.17g')
