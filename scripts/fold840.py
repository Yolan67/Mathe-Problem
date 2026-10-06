#!/usr/bin/env python3
"""Build the 840-point code configuration in R^12 (24 axes + 16*A(12,4,4)=51 blocks) and fold to R^11.
fold rules: h = a*x10 + b*x11 (then rows renormalised), other coords kept."""
import sys, itertools, numpy as np
sys.path.insert(0, 'scripts')
from build_code582 import find_packing
P = find_packing(n=12, k=4, target=51, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
assert P and len(P) == 51
pts = []
for i in range(12):
    for s in (2, -2):
        v = [0] * 12; v[i] = s; pts.append(v)
for B in P:
    for sg in itertools.product((1, -1), repeat=4):
        v = [0] * 12
        for a, s in zip(B, sg): v[a] = s
        pts.append(v)
X = np.array(pts, float)
np.savetxt('/tmp/claude-0/c840.txt', X, fmt='%d')
mode = sys.argv[1]
a, b = {'diag': (1, 1), 'axis': (1, 0), 'mix': (1, 0.5), 'mix2': (1, -0.3)}[mode]
h = (a * X[:, 10] + b * X[:, 11]) / np.hypot(a, b)
# keep the radial length: |(x10,x11)| with sign of projection (norm preserving fold)
r = np.hypot(X[:, 10], X[:, 11]); sgn = np.sign(h); sgn[sgn == 0] = 1
Y = np.column_stack([X[:, :10], sgn * r])
np.savetxt('/tmp/claude-0/fold_%s.txt' % mode, Y, fmt='%.17g')
G = Y @ Y.T; np.fill_diagonal(G, -9)
print(mode, 'points', len(Y), 'pairs violating', int((G > 2 + 1e-9).sum() // 2))
