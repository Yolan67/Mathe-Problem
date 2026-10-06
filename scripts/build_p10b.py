#!/usr/bin/env python3
"""10-dim code configuration P10b (500 = 20 + 16*30, blocks = SQS(10)) embedded
in R^11 (last coordinate 0).  Writes configs/p10b_eq.txt (11 coords) and
configs/p10b_10d.txt (10 coords)."""
import itertools, sys
sys.path.insert(0, 'scripts')
from build_code582 import find_packing
P = find_packing(n=10, k=4, target=30, seed=1)
assert P and len(P) == 30
pts = []
for i in range(10):
    for s in (2, -2):
        v = [0]*10; v[i] = s; pts.append(v)
for B in P:
    for sg in itertools.product((1, -1), repeat=4):
        v = [0]*10
        for a, s in zip(B, sg): v[a] = s
        pts.append(v)
with open('configs/p10b_10d.txt', 'w') as f:
    for v in pts: f.write(' '.join(map(str, v)) + '\n')
with open('configs/p10b_eq.txt', 'w') as f:
    for v in pts: f.write(' '.join(map(str, v + [0])) + '\n')
with open('configs/sqs10_blocks.txt', 'w') as f:
    for B in P: f.write(' '.join(map(str, B)) + '\n')
print(len(pts))
