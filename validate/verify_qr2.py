#!/usr/bin/env python3
"""Independent exact validator for kissing configurations with coordinates in Q(sqrt2).

Input file: one point per line, whitespace separated coordinates.  Each
coordinate is either a rational "p" / "p/q" or a pair "a:b" meaning a + b*sqrt(2)
with rationals a, b.  Everything is parsed with fractions.Fraction; no floating
point is used in the decision.

Certified configuration: u_i = x_i/|x_i|.  For a pair the condition
<u_i,u_j> <= 1/2 is equivalent to
    <x_i,x_j> <= 0   or   4 <x_i,x_j>^2 <= |x_i|^2 |x_j|^2,
and all quantities lie in Q(sqrt2), where the sign of a + b sqrt2 is decided
exactly.  Reports max <u_i,u_j> and min |u_i-u_j| to 50 digits.

Usage: verify_qr2.py FILE [--dim 11]          exit code 0 iff valid
"""
import sys
from fractions import Fraction as Fr
from decimal import Decimal, getcontext
getcontext().prec = 70
SQ2 = Decimal(2).sqrt()


class Q2:
    __slots__ = ('a', 'b')

    def __init__(self, a, b=0):
        self.a = Fr(a); self.b = Fr(b)

    def __add__(s, o): return Q2(s.a + o.a, s.b + o.b)
    def __sub__(s, o): return Q2(s.a - o.a, s.b - o.b)
    def __mul__(s, o): return Q2(s.a * o.a + 2 * s.b * o.b, s.a * o.b + s.b * o.a)
    def sign(s):
        a, b = s.a, s.b
        if a >= 0 and b >= 0:
            return 0 if (a == 0 and b == 0) else 1
        if a <= 0 and b <= 0:
            return -1
        # opposite signs: compare a^2 with 2 b^2
        d = a * a - 2 * b * b
        if a > 0:   # a>0, b<0
            return 1 if d > 0 else -1   # d == 0 impossible (sqrt2 irrational) unless a=b=0
        return -1 if d > 0 else 1        # a<0, b>0
    def dec(s):
        return Decimal(s.a.numerator) / Decimal(s.a.denominator) + Decimal(s.b.numerator) / Decimal(s.b.denominator) * SQ2


def parse_coord(t):
    if ':' in t:
        a, b = t.split(':')
        return Q2(Fr(a), Fr(b))
    return Q2(Fr(t), 0)


def main():
    args = sys.argv[1:]
    dim = 11
    if '--dim' in args:
        i = args.index('--dim'); dim = int(args[i + 1]); del args[i:i + 2]
    X = []
    for line in open(args[0]):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        v = [parse_coord(t) for t in line.split()]
        if len(v) != dim:
            print(f"FAIL: point with {len(v)} coordinates"); print("RESULT: INVALID"); return 1
        X.append(v)
    N = len(X)
    def ip(u, v):
        s = Q2(0)
        for x, y in zip(u, v):
            if (x.a or x.b) and (y.a or y.b):
                s = s + x * y
        return s
    nrm = [ip(v, v) for v in X]
    for k, n in enumerate(nrm):
        if n.sign() <= 0:
            print(f"FAIL: point {k} is zero"); print("RESULT: INVALID"); return 1
    norms = set((n.a, n.b) for n in nrm)
    bad = 0
    maxc = Decimal(-2)
    for i in range(N):
        for j in range(i + 1, N):
            g = ip(X[i], X[j])
            if g.sign() <= 0:
                continue
            lhs = Q2(4) * g * g
            rhs = nrm[i] * nrm[j]
            if (lhs - rhs).sign() > 0:
                bad += 1
                if bad <= 10:
                    print(f"VIOLATION pair ({i},{j})")
            c = g.dec() / (nrm[i].dec() * nrm[j].dec()).sqrt()
            if c > maxc:
                maxc = c
    print(f"points: {N}  dim: {dim}")
    print(f"distinct squared norms: {len(norms)}  e.g. {[(str(a), str(b)) for a, b in list(norms)[:3]]}")
    print(f"max <u_i,u_j>   = {maxc:.50f}")
    print(f"min |u_i - u_j| = {(2 - 2 * maxc).sqrt():.50f}")
    print(f"pairs violating: {bad}")
    print("RESULT: VALID" if bad == 0 else "RESULT: INVALID")
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
