#!/usr/bin/env python3
"""assemble.py BACKBONE.txt CANDS.json SOL.txt OUT.txt
Backbone: integer coordinates.  Candidates: json from freecands (pairs p,q over den).
Writes exact file (coordinates 'a:b' = a + b sqrt2, a,b rationals) and OUT.f64.txt floats."""
import sys, json
from fractions import Fraction as Fr
R2 = 2 ** 0.5
bb = [list(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()]
d = json.load(open(sys.argv[2])); den = d['den']; vecs = d['vecs']
sol = [int(l) for l in open(sys.argv[3]) if l.strip()]
out = sys.argv[4]
def fmt(p, q):
    a, b = Fr(p, den), Fr(q, den)
    return str(a) if b == 0 else f"{a}:{b}"
with open(out, 'w') as f, open(out.replace('.txt', '.f64.txt'), 'w') as g:
    for v in bb:
        f.write(' '.join(str(x) for x in v) + '\n')
        g.write(' '.join('%.17g' % x for x in v) + '\n')
    for i in sol:
        v = vecs[i]
        f.write(' '.join(fmt(p, q) for p, q in v) + '\n')
        g.write(' '.join('%.17g' % ((p + q * R2) / den) for p, q in v) + '\n')
print(len(bb) + len(sol), 'points ->', out)
