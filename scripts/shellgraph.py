#!/usr/bin/env python3
"""Build a candidate shell in R^11 and its conflict graph.

Coordinates are elements of Z[sqrt2]/den given as pairs (p,q) -> (p + q*sqrt2)/den.
A value set is given by --vals "p:q,p:q,..." (nonnegative values; signs added).
All vectors with exactly squared norm 4 (i.e. sum (p+q r2)^2 = 4 den^2, both
rational and sqrt2 parts) are enumerated.  Conflict iff <u,v> > 2 (cos > 1/2),
decided with a 1e-9 tolerance (exact for Z[sqrt2]: an irrational value is never
within 1e-9 of 2 for these small integers; equality occurs only in Q).

Outputs: PREFIX.vec (exact (p,q) coordinates, json), PREFIX.f64 (float array),
PREFIX.graph (binary adjacency for misg).
"""
import argparse, json, itertools, sys
import numpy as np

R2 = 2 ** 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vals', default='1:0,2:0,0:1')
    ap.add_argument('--den', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dim', type=int, default=11)
    ap.add_argument('--maxsupp', type=int, default=11)
    a = ap.parse_args()
    vals = [tuple(map(int, t.split(':'))) for t in a.vals.split(',')]
    sv = set()
    for p, q in vals:
        sv.add((p, q)); sv.add((-p, -q))
    vals = sorted(sv)
    target = 4 * a.den * a.den
    # value squared: (p^2 + 2 q^2) + 2 p q sqrt2
    # enumerate multisets of nonzero absolute values first, then place
    absvals = sorted({(p, q) if (p, q) > (0, 0) else (-p, -q) for p, q in vals})
    sq = {v: (v[0] ** 2 + 2 * v[1] ** 2, 2 * v[0] * v[1]) for v in absvals}
    pats = []
    def rec(i, remr, rems, cur):
        if remr == 0 and rems == 0:
            pats.append(list(cur)); return
        if len(cur) >= min(a.dim, a.maxsupp) or i >= len(absvals):
            return
        v = absvals[i]
        r, s = sq[v]
        # take v some number of times
        k = 0
        while True:
            if r * k > remr:
                break
            rec(i + 1, remr - r * k, rems - s * k, cur + [v] * k)
            k += 1
            if len(cur) + k > a.dim:
                break
    rec(0, target, 0, [])
    vecs = []
    for pat in pats:
        L = len(pat)
        # distinct permutations of pattern placed on supports
        for supp in itertools.combinations(range(a.dim), L):
            for perm in set(itertools.permutations(pat)):
                for signs in itertools.product((1, -1), repeat=L):
                    v = [(0, 0)] * a.dim
                    for idx, (p, q), s in zip(supp, perm, signs):
                        v[idx] = (s * p, s * q)
                    vecs.append(tuple(v))
    vecs = sorted(set(vecs))
    n = len(vecs)
    print('patterns', len(pats), 'vectors', n, file=sys.stderr)
    F = np.array([[(p + q * R2) / a.den for p, q in v] for v in vecs])
    nr = (F * F).sum(1)
    assert np.allclose(nr, 4)
    with open(a.out + '.vec', 'w') as f:
        json.dump({'den': a.den, 'vecs': vecs}, f)
    np.save(a.out + '.npy', F)
    # graph
    with open(a.out + '.graph', 'wb') as f:
        np.array([n, 0], dtype=np.int32).tofile(f)
        B = 4096
        m = 0
        for s in range(0, n, B):
            G = F[s:s + B] @ F.T
            for i in range(G.shape[0]):
                row = np.nonzero(G[i] > 2 + 1e-9)[0]
                row = row[row != s + i].astype(np.int32)
                np.array([len(row)], dtype=np.int32).tofile(f)
                row.tofile(f)
                m += len(row)
    print('edges', m // 2, 'avg deg', m / n, file=sys.stderr)

if __name__ == '__main__':
    main()
