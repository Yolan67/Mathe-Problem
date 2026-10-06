#!/usr/bin/env python3
"""Independent exact validator for kissing configurations.

Input file: one point per line, whitespace separated coordinates.
Coordinates may be integers, fractions "p/q" or decimal strings; they are
parsed exactly (fractions.Fraction), never via floating point.

The configuration that is certified is {x_i / |x_i|}.  Normalisation does not
need to be carried out explicitly: for nonzero rational vectors x, y

    <x/|x|, y/|y|> <= 1/2   <=>   <x,y> <= 0   or   4 <x,y>^2 <= |x|^2 |y|^2

which is decided in exact integer arithmetic.  The unit vectors x_i/|x_i| then
satisfy |u_i| = 1 exactly and <u_i,u_j> <= 1/2, i.e. |u_i - u_j| >= 1.

Additionally max <u_i,u_j> and min |u_i-u_j| are reported with 60 significant
digits (decimal module) for information.

Usage: verify.py FILE [--dim 11]
Exit code 0 iff the configuration is valid.
"""
import sys
from fractions import Fraction
from decimal import Decimal, getcontext
from math import lcm

getcontext().prec = 60


def parse(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            pts.append([Fraction(t) for t in line.replace(',', ' ').split()])
    return pts


def to_int(v):
    d = 1
    for c in v:
        d = lcm(d, c.denominator)
    return [int(c * d) for c in v]


def main():
    args = sys.argv[1:]
    dim = 11
    if '--dim' in args:
        i = args.index('--dim')
        dim = int(args[i + 1])
        del args[i:i + 2]
    path = args[0]
    raw = parse(path)
    N = len(raw)
    ok = True
    for k, v in enumerate(raw):
        if len(v) != dim:
            print(f"FAIL: point {k} has {len(v)} coordinates, expected {dim}")
            ok = False
    X = [to_int(v) for v in raw]
    nrm = [sum(c * c for c in v) for v in X]
    for k, n in enumerate(nrm):
        if n == 0:
            print(f"FAIL: point {k} is the zero vector")
            ok = False
    if not ok:
        print("RESULT: INVALID")
        return 1

    bad = 0
    best_num = None  # track max cosine exactly as (ip, n_i*n_j) with ip>0
    best_pair = None
    for i in range(N):
        xi = X[i]
        ni = nrm[i]
        for j in range(i + 1, N):
            xj = X[j]
            ip = sum(a * b for a, b in zip(xi, xj))
            if ip > 0:
                lhs = 4 * ip * ip
                rhs = ni * nrm[j]
                if lhs > rhs:
                    bad += 1
                    if bad <= 10:
                        print(f"VIOLATION: pair ({i},{j}) cos^2 = {Fraction(ip*ip, rhs)} > 1/4")
                # compare ip^2/(ni nj) with current best
                if best_num is None or ip * ip * best_num[1] > best_num[0] * rhs:
                    best_num = (ip * ip, rhs)
                    best_pair = (i, j, ip)
    if best_num is None:
        maxcos = Decimal(0)
    else:
        i, j, ip = best_pair
        maxcos = Decimal(ip) / (Decimal(nrm[i]) * Decimal(nrm[j])).sqrt()
    mindist = (2 - 2 * maxcos).sqrt()
    print(f"points: {N}  dim: {dim}")
    print(f"max <u_i,u_j>   = {maxcos}")
    print(f"min |u_i - u_j| = {mindist}")
    print(f"max cos^2 exact = {Fraction(*best_num) if best_num else 0}  (must be <= 1/4)")
    print(f"pairs violating: {bad}")
    if bad == 0:
        print("RESULT: VALID")
        return 0
    print("RESULT: INVALID")
    return 1


if __name__ == '__main__':
    sys.exit(main())
