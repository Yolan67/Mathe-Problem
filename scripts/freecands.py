#!/usr/bin/env python3
"""Exact candidates compatible with a sign-closed backbone, values in (Z + Z sqrt2)/den.

Backbone: axes +-2e_s for s in AX and all sign vectors (+-1)^4 on given blocks.
Since the backbone is closed under sign changes, y is compatible iff
  2|y_s| <= 2 for s in AX     and    sum_{i in B} |y_i| <= 2 for every block B.
Candidates: all y with coordinates from +-VALS, |y|^2 = 4 exactly (rational and
sqrt2 parts), compatible with the backbone.

Output: PREFIX.json with exact coordinates as [p,q] pairs (value = (p+q*sqrt2)/den),
PREFIX.npy float array, PREFIX.graph conflict graph (ip > 2) for misg.
"""
import argparse, itertools, json, sys
import numpy as np
R2 = 2 ** 0.5

def le_const(a, b, c):
    """exact test a + b*sqrt2 <= c  for integer a,b,c (b may be negative)"""
    # b*sqrt2 <= c - a
    L = c - a
    if b == 0:
        return L >= 0
    if b > 0:
        return L >= 0 and 2 * b * b <= L * L
    # b < 0: always true if L >= 0, else need 2b^2 >= L^2
    return L >= 0 or 2 * b * b >= L * L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', required=True)
    ap.add_argument('--axes', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--vals', default='1:0,2:0,3:0,4:0,0:1,0:2')
    ap.add_argument('--den', type=int, default=2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nograph', action='store_true')
    a = ap.parse_args()
    den = a.den
    blocks = [tuple(map(int, l.split())) for l in open(a.blocks) if l.strip()]
    AX = set(int(x) for x in a.axes.split(',') if x != '')
    absvals = [(0, 0)] + [tuple(map(int, t.split(':'))) for t in a.vals.split(',')]
    vf = np.array([(p + q * R2) / den for p, q in absvals])
    # norms in units of 1/den^2 : (p^2 + 2q^2) + sqrt2 * 2pq
    nr = np.array([p * p + 2 * q * q for p, q in absvals])
    ns = np.array([2 * p * q for p, q in absvals])
    target = 4 * den * den
    # enumerate absolute patterns per coordinate (indices into absvals) by DFS with pruning
    dim = 11
    out = []
    eps = 1e-9
    blocks_of = [[bi for bi, B in enumerate(blocks) if i in B] for i in range(dim)]
    bsum = [0.0] * len(blocks)
    cur = [0] * dim
    def dfs(i, remr, rems):
        if i == dim:
            if remr == 0 and rems == 0:
                out.append(tuple(cur))
            return
        for k in range(len(absvals)):
            if nr[k] > remr:
                continue
            if i in AX and vf[k] > 1 + eps:
                continue
            ok = True
            for bi in blocks_of[i]:
                if bsum[bi] + vf[k] > 2 + 1e-7:
                    ok = False; break
            if not ok:
                continue
            # remaining norm must be achievable: crude bound skip
            for bi in blocks_of[i]:
                bsum[bi] += vf[k]
            cur[i] = k
            dfs(i + 1, remr - nr[k], rems - ns[k])
            for bi in blocks_of[i]:
                bsum[bi] -= vf[k]
        cur[i] = 0
    dfs(0, target, 0)
    print('abs patterns', len(out), file=sys.stderr)
    # exact recheck of block sums with integer arithmetic, then expand signs
    vecs = []
    for pat in out:
        ok = True
        for B in blocks:
            sa = sum(absvals[pat[i]][0] for i in B)
            sb = sum(absvals[pat[i]][1] for i in B)
            if not le_const(sa, sb, 2 * den):
                ok = False; break
        for s in AX:
            p, q = absvals[pat[s]]
            if not le_const(p, q, den):
                ok = False
        if not ok:
            continue
        supp = [i for i in range(dim) if pat[i] != 0]
        for sg in itertools.product((1, -1), repeat=len(supp)):
            v = [[0, 0] for _ in range(dim)]
            for i, s in zip(supp, sg):
                p, q = absvals[pat[i]]
                v[i] = [s * p, s * q]
            vecs.append(v)
    n = len(vecs)
    print('candidates', n, file=sys.stderr)
    F = np.array([[(p + q * R2) / den for p, q in v] for v in vecs]) if n else np.zeros((0, dim))
    json.dump({'den': den, 'vecs': vecs}, open(a.out + '.json', 'w'))
    np.save(a.out + '.npy', F)
    if a.nograph or n == 0:
        return
    with open(a.out + '.graph', 'wb') as f:
        np.array([n, 0], dtype=np.int32).tofile(f)
        m = 0
        for s in range(0, n, 4096):
            G = F[s:s + 4096] @ F.T
            for i in range(G.shape[0]):
                row = np.nonzero(G[i] > 2 + 1e-9)[0]
                row = row[row != s + i].astype(np.int32)
                np.array([len(row)], dtype=np.int32).tofile(f)
                row.tofile(f)
                m += len(row)
    print('edges', m // 2, 'avg deg %.1f' % (m / n), file=sys.stderr)

if __name__ == '__main__':
    main()
