#!/usr/bin/env python3
"""Exact 604-point kissing configuration in R^11 (coordinates in Q(sqrt2)).

R^11 = R^8 (coords 0..7, pairs {0,1},{2,3},{4,5},{6,7}) + R^3 (coords 8,9,10).
* backbone (496): +-2e_s (s<8) and (+-1)^4 on 30 blocks (configs/layer30_blocks.txt)
* 96 = 16 pair vectors u = (+-1,+-1) on a pair  x  6 frame vectors w = +-sqrt2 f_i,
       f = orthonormal frame with all |coords| <= 1/sqrt2
* 12 = 2 * cuboctahedron in frame coordinates: sqrt2 (+-f_a +-f_b), a<b
All vectors have squared norm exactly 4.  Coordinates written as 'a:b' = a + b sqrt2.
"""
import itertools, sys
from fractions import Fraction as Fr
# numbers in Q(sqrt2) as (a,b)
def add(x, y): return (x[0] + y[0], x[1] + y[1])
def neg(x): return (-x[0], -x[1])
def mul(x, y): return (x[0] * y[0] + 2 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])
R = (Fr(0), Fr(1))           # sqrt2
H = (Fr(1, 2), Fr(0))        # 1/2
Rh = (Fr(0), Fr(1, 2))       # sqrt2/2 = 1/sqrt2
Z0 = (Fr(0), Fr(0))
f = [[Rh, H, H], [Rh, neg(H), neg(H)], [Z0, Rh, neg(Rh)]]   # unit frame
def scal(c, v): return [mul(c, t) for t in v]
def vadd(v, w): return [add(a, b) for a, b in zip(v, w)]
pts = []
for l in open('configs/backbone_L30.txt'):
    pts.append([(Fr(int(t)), Fr(0)) for t in l.split()])
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
W = []
for fi in f:
    for s in (1, -1):
        W.append(scal((Fr(s), Fr(0)), scal(R, fi)))
for (a, b) in pairs:
    for sa, sb in itertools.product((1, -1), repeat=2):
        for w in W:
            v = [Z0] * 8
            v[a] = (Fr(sa), Fr(0)); v[b] = (Fr(sb), Fr(0))
            pts.append(v + w)
for i, j in itertools.combinations(range(3), 2):
    for si, sj in itertools.product((1, -1), repeat=2):
        z = scal(R, vadd(scal((Fr(si), Fr(0)), f[i]), scal((Fr(sj), Fr(0)), f[j])))
        pts.append([Z0] * 8 + z)
def fmt(x): return str(x[0]) if x[1] == 0 else f"{x[0]}:{x[1]}"
out = sys.argv[1] if len(sys.argv) > 1 else 'records/N604/config_exact.txt'
with open(out, 'w') as fo:
    for v in pts: fo.write(' '.join(fmt(t) for t in v) + '\n')
print(len(pts), 'points ->', out)
