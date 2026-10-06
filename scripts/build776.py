#!/usr/bin/env python3
"""12-dim analogue of the 604: R^8 (+) R^4.  D8 part (112) + 4 lifted spinor layers (4x128)
+ four D6 slices (pair (+) R^4: 16 pair vectors x 8 frame vectors) + D4 core (24 in frame coords).
Frame in R^4: rows of Hadamard/2 (coords ±1/2).  Writes /tmp/claude-0/c776.txt and checks."""
import itertools, json, numpy as np
r2 = 2 ** .5
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
trans = []
for miss in range(4):
    others = [p for i, p in enumerate(pairs) if i != miss]
    for bits in itertools.product((0, 1), repeat=3):
        trans.append(tuple(sorted(o[b] for o, b in zip(others, bits))))
d = json.load(open('configs/layer_systems.json'))
cl = [tuple(map(tuple, c)) for c in d['cliques']]
# find 4 pairwise disjoint cliques
sel = None
for comb in itertools.combinations(range(len(cl)), 4):
    S = [set(cl[i]) for i in comb]
    if all(not (S[i] & S[j]) for i in range(4) for j in range(i + 1, 4)):
        sel = comb; break
print('4 disjoint layers:', sel)
pts = []
for s in range(8):
    for g in (2, -2):
        v = np.zeros(12); v[s] = g; pts.append(v)
for (a, b), (c, e) in itertools.combinations(pairs, 2):
    for sg in itertools.product((1, -1), repeat=4):
        v = np.zeros(12); v[[a, b, c, e]] = sg; pts.append(v)
for k, ci in enumerate(sel):
    for t in cl[ci]:
        for sg in itertools.product((1, -1), repeat=4):
            v = np.zeros(12); v[list(t) + [8 + k]] = sg; pts.append(v)
Hd = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2.0
F = [r2 * s * h for h in Hd for s in (1, -1)]   # 8 frame vectors, norm sqrt2
for (a, b) in pairs:
    for sa, sb in itertools.product((1, -1), repeat=2):
        for w in F:
            v = np.zeros(12); v[a] = sa; v[b] = sb; v[8:] = w; pts.append(v)
for i, j in itertools.combinations(range(4), 2):
    for si, sj in itertools.product((1, -1), repeat=2):
        v = np.zeros(12); v[8:] = r2 * (si * Hd[i] + sj * Hd[j]); pts.append(v)
X = np.array(pts)
G = X @ X.T; np.fill_diagonal(G, -9)
print('N', len(X), 'norms', np.unique(np.round((X ** 2).sum(1), 9)), 'max ip', round(G.max(), 9))
np.savetxt('/tmp/claude-0/c776.txt', X, fmt='%.17g')
