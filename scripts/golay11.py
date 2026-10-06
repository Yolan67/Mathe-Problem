#!/usr/bin/env python3
"""D11 roots (norm 8: (±2,±2,0^9)) + ternary Golay [11,6,5] codewords of weight w in W (as ±1 vectors,
scaled to norm 8), optionally in all coordinate permutations of a cyclic/QR form.  MIS candidate graph.
usage: golay11.py OUT weights(e.g. 8,9,11)"""
import sys, itertools
import numpy as np
out = sys.argv[1]; W = [int(x) for x in sys.argv[2].split(',')]
# ternary Golay code [11,6,5]: quadratic residue code mod 11 over GF(3); generator polynomial g(x)
# QR mod 11 = {1,3,4,5,9}; g(x) = x^5 + x^4 - x^3 + x^2 - 1 (one of the two factors of (x^11-1)/(x-1) over GF(3))
g = [-1, 0, 1, -1, 1, 1]   # coefficients of x^0..x^5 :  -1 + x^2 - x^3 + x^4 + x^5
G = np.zeros((6, 11), int)
for i in range(6):
    for j, c in enumerate(g): G[i, (i + j) % 11] = c % 3
words = set()
for coef in itertools.product(range(3), repeat=6):
    w = tuple((np.array(coef) @ G) % 3)
    words.add(w)
words = np.array(sorted(words))
wt = (words != 0).sum(1)
from collections import Counter
print('code size', len(words), 'weights', sorted(Counter(wt).items()), file=sys.stderr)
pm = np.where(words == 1, 1.0, np.where(words == 2, -1.0, 0.0))
V = []
for i, j in itertools.combinations(range(11), 2):
    for s, t in itertools.product((2, -2), repeat=2):
        v = np.zeros(11); v[i] = s; v[j] = t; V.append(v)
nroots = len(V)
for w in W:
    for c in pm[wt == w]:
        V.append(c * (8 / w) ** .5)
V = np.array(V); n = len(V)
top2 = np.sort(np.abs(V), 1)[:, -2:].sum(1)
print('candidates', n, 'region-violating (top2>2):', (top2[nroots:] > 2 + 1e-9).sum(), file=sys.stderr)
Gm = V @ V.T; np.fill_diagonal(Gm, -99)
print('max ip among code words:', Gm[nroots:, nroots:].max(), file=sys.stderr)
np.save(out + '.npy', V)
m = 0
with open(out + '.graph', 'wb') as fo:
    np.array([n, 0], dtype=np.int32).tofile(fo)
    for i in range(n):
        row = np.nonzero(Gm[i] > 4 + 1e-9)[0].astype(np.int32)
        np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
print('edges', m // 2, file=sys.stderr)
