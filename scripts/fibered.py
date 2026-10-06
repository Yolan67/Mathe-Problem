#!/usr/bin/env python3
"""Fibered candidate sets for R^11 = R^8 (+) R^3 and their conflict graph.

Every candidate is (u, w) with |u|^2 + |w|^2 = 4.  Classes (|u|^2, w-set):
  A : |u|^2=4, w=0
  L : |u|^2=3, w in {+-e_k}
  P : |u|^2=2, w in +-sqrt2 * frame f (f1,f2,f3 of records/N604)
  Q : |u|^2=2, w in {(+-1,+-1,0) perms}
  C : |u|^2=1, w in {(+-1,+-1,+-1)}
  Z : u=0, w in 2*cuboctahedron(frame) (12)  [+ optionally 2*octahedron (+-2e_k)]
u ranges over vectors in (1/2)Z^8 of the given norm whose coordinates are
all integers or (option --half) with half-integers allowed.
Values are exact in Q(sqrt2); conflicts (ip > 2) decided in floating point with
margin 1e-9 (exact here since all inner products are of the form a + b sqrt2 with
small rationals and an irrational value is never within 1e-9 of 2).
Writes PREFIX.npy, PREFIX.json (exact), PREFIX.graph, and PREFIX_init.txt (indices
of the exact 604 configuration if all its vectors are candidates).
"""
import argparse, itertools, json, sys
import numpy as np
from fractions import Fraction as Fr
R2 = 2 ** 0.5

def vecs_norm(n2, dim=8, half=False, maxsupp=8, coset=False):
    """all u in Z^dim (or (1/2)Z^dim if half) with |u|^2 = n2 (n2 Fraction).
    coset: additionally all u in (Z+1/2)^dim (every coordinate half-odd)."""
    out = []
    if coset:
        t4 = int(n2 * 4)
        def rc(i, rem, cur):
            if i == dim:
                if rem == 0: out.append(tuple(Fr(c, 2) for c in cur))
                return
            for c in range(-7, 8, 2):
                if c * c <= rem: rc(i + 1, rem - c * c, cur + [c])
        rc(0, t4, [])
    scale = 2 if half else 1
    target = int(n2 * scale * scale)
    def rec(i, rem, cur):
        if i == dim:
            if rem == 0:
                out.append(tuple(Fr(c, scale) for c in cur))
            return
        m = int(rem ** 0.5)
        for c in range(-m, m + 1):
            if c * c <= rem:
                rec(i + 1, rem - c * c, cur + [c])
    rec(0, target, [])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--half', default='', help='classes using half-integers, e.g. "APL"')
    ap.add_argument('--classes', default='ALPQCZ')
    ap.add_argument('--octa', action='store_true')
    ap.add_argument('--coset', default='', help='classes that also get (Z+1/2)^8 u-vectors')
    ap.add_argument('--dirs', action='store_true', help='use direction catalogue for L,P,Z')
    a = ap.parse_args()
    # exact elements of Q(sqrt2) as (a,b)
    h = Fr(1, 2)
    f = [[(0, h), (h, 0), (h, 0)], [(0, h), (-h, 0), (-h, 0)], [(0, 0), (0, h), (0, -h)]]
    def sc(c, v): return [(c[0] * x[0] + 2 * c[1] * x[1], c[0] * x[1] + c[1] * x[0]) for x in v]
    def ad(v, w): return [(x[0] + y[0], x[1] + y[1]) for x, y in zip(v, w)]
    r2 = (Fr(0), Fr(1))
    W = {}
    W['A'] = [[(Fr(0), Fr(0))] * 3]
    W['L'] = []
    for k in range(3):
        for s in (1, -1):
            w = [(Fr(0), Fr(0))] * 3; w[k] = (Fr(s), Fr(0)); W['L'].append(w)
    W['P'] = [sc((Fr(s), Fr(0)), sc(r2, fi)) for fi in f for s in (1, -1)]
    W['Q'] = []
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((1, -1), repeat=2):
            w = [(Fr(0), Fr(0))] * 3; w[i] = (Fr(si), Fr(0)); w[j] = (Fr(sj), Fr(0)); W['Q'].append(w)
    W['C'] = [[(Fr(s0), Fr(0)), (Fr(s1), Fr(0)), (Fr(s2), Fr(0))] for s0, s1, s2 in itertools.product((1, -1), repeat=3)]
    Zs = []
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((1, -1), repeat=2):
            Zs.append(sc(r2, ad(sc((Fr(si), Fr(0)), f[i]), sc((Fr(sj), Fr(0)), f[j]))))
    if a.octa:
        for k in range(3):
            for s in (2, -2):
                w = [(Fr(0), Fr(0))] * 3; w[k] = (Fr(s), Fr(0)); Zs.append(w)
    if a.dirs:
        # unit directions in Q(sqrt2)^3: axes, face diagonals, frame, cuboctahedron-in-frame
        D = []
        for k in range(3):
            for s_ in (1, -1):
                w = [(Fr(0), Fr(0))] * 3; w[k] = (Fr(s_), Fr(0)); D.append(w)
        for i, j in itertools.combinations(range(3), 2):
            for si, sj in itertools.product((1, -1), repeat=2):
                w = [(Fr(0), Fr(0))] * 3; w[i] = (Fr(0), Fr(si, 2)); w[j] = (Fr(0), Fr(sj, 2)); D.append(w)
        for fi in f:
            for s_ in (1, -1): D.append(sc((Fr(s_), Fr(0)), fi))
        for z in Zs[:12]:
            D.append(sc((Fr(1, 2), Fr(0)), z))   # z has length 2
        # dedupe
        DD = []
        for w in D:
            if w not in DD: DD.append(w)
        D = DD
        print('directions', len(D), file=sys.stderr)
        W['L'] = D
        W['P'] = [sc(r2, w) for w in D]
        W['C'] = [sc((Fr(0), Fr(0)), w) for w in D]  # placeholder replaced below
        # |w|^2 = 3 needs sqrt3 -> skip C directions except cube vertices
        W['C'] = [[(Fr(s0), Fr(0)), (Fr(s1), Fr(0)), (Fr(s2), Fr(0))] for s0, s1, s2 in itertools.product((1, -1), repeat=3)]
        Zs = Zs + [sc((Fr(2), Fr(0)), w) for w in D]
    unorm = {'A': Fr(4), 'L': Fr(3), 'P': Fr(2), 'Q': Fr(2), 'C': Fr(1)}
    cands = []
    for cl in a.classes:
        if cl == 'Z':
            for z in Zs:
                cands.append(('Z', [(Fr(0), Fr(0))] * 8 + z))
            continue
        U = vecs_norm(unorm[cl], half=(cl in a.half), coset=(cl in a.coset))
        print(cl, len(U), 'u-vectors x', len(W[cl]), 'w', file=sys.stderr)
        for u in U:
            for w in W[cl]:
                cands.append((cl, [(x, Fr(0)) for x in u] + w))
    # dedupe
    seen = {}
    for cl, v in cands:
        key = tuple(v)
        if key not in seen:
            seen[key] = cl
    keys = list(seen.keys())
    n = len(keys)
    F = np.array([[float(x[0]) + float(x[1]) * R2 for x in v] for v in keys])
    nr = (F * F).sum(1)
    assert np.allclose(nr, 4), nr[np.abs(nr - 4) > 1e-9][:5]
    print('candidates', n, file=sys.stderr)
    json.dump({'vecs': [[[str(x[0]), str(x[1])] for x in v] for v in keys], 'cls': [seen[k] for k in keys]}, open(a.out + '.json', 'w'))
    np.save(a.out + '.npy', F)
    with open(a.out + '.graph', 'wb') as fo:
        np.array([n, 0], dtype=np.int32).tofile(fo)
        m = 0
        for s in range(0, n, 2048):
            G = F[s:s + 2048] @ F.T
            for i in range(G.shape[0]):
                row = np.nonzero(G[i] > 2 + 1e-9)[0]
                row = row[row != s + i].astype(np.int32)
                np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
    print('edges', m // 2, 'avg deg %.1f' % (m / n), file=sys.stderr)
    # init from 604
    idx = {k: i for i, k in enumerate(keys)}
    init = []
    miss = 0
    for line in open('records/N604/config_exact.txt'):
        v = []
        for t in line.split():
            if ':' in t:
                x, y = t.split(':'); v.append((Fr(x), Fr(y)))
            else:
                v.append((Fr(t), Fr(0)))
        if tuple(v) in idx: init.append(idx[tuple(v)])
        else: miss += 1
    open(a.out + '_init.txt', 'w').write('\n'.join(map(str, init)))
    print('init604 found', len(init), 'missing', miss, file=sys.stderr)

if __name__ == '__main__':
    main()
