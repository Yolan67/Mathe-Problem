#!/usr/bin/env python3
"""Multi-frame candidate sets: union of lattice shells in several rotated orthonormal frames.

Frame F (orthogonal 11x11, columns = new basis vectors in standard coordinates):
  F_M,f : 45-degree rotations on the pairs of a matching M of S, and the 3x3 frame f
          (rotated by a signed permutation) on the remaining coordinates T.
Shells per frame: 'r' = sqrt2 * D11 roots (220 vectors (±√2,±√2)), 's' = Z^11 norm-4 shell
(5302 vectors (±2), (±1)^4).  Coordinates are in Q(sqrt2); conflicts (ip > 2) are decided in
float64 with tolerance 1e-9, which is exact for these small algebraic numbers.

usage: multiframe.py OUT [--matchings all|k] [--tvariants k] [--shells F1:s,r FM:r]
Writes OUT.npy, OUT.graph (misg format), OUT_init.txt (indices of the exact 604).
"""
import argparse, itertools, sys
import numpy as np
r2 = 2 ** .5

def perfect_matchings(pts):
    if not pts: yield []; return
    a = pts[0]
    for i in range(1, len(pts)):
        b = pts[i]
        rest = pts[1:i] + pts[i + 1:]
        for m in perfect_matchings(rest):
            yield [(a, b)] + m

def frame(M, T, f):
    F = np.eye(11)
    for (a, b) in M:
        F[:, a] = 0; F[:, b] = 0
        F[a, a] = 1 / r2; F[b, a] = 1 / r2; F[a, b] = 1 / r2; F[b, b] = -1 / r2
    if T is not None:
        T = list(T)
        for i in T:
            F[:, i] = 0
        for j in range(3):
            for i in range(3):
                F[T[i], T[j]] = f[j][i]
    assert np.allclose(F.T @ F, np.eye(11))
    return F

def roots_sqrt2():
    V = []
    for i, j in itertools.combinations(range(11), 2):
        for s, t in itertools.product((1, -1), repeat=2):
            v = np.zeros(11); v[i] = s * r2; v[j] = t * r2; V.append(v)
    return np.array(V)

def shell4():
    V = []
    for i in range(11):
        for s in (2, -2):
            v = np.zeros(11); v[i] = s; V.append(v)
    for B in itertools.combinations(range(11), 4):
        for sg in itertools.product((1, -1), repeat=4):
            v = np.zeros(11); v[list(B)] = sg; V.append(v)
    return np.array(V)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--nmatch', type=int, default=105)
    ap.add_argument('--tvar', type=int, default=1)
    ap.add_argument('--Fms', default='r', help="shells used in rotated frames: r, s or rs")
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    S = list(range(8)); T = [8, 9, 10]
    f0 = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
    # T variants: signed permutations of f0 rows/cols
    tv = [f0]
    allv = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            g = f0[:, list(p)] * np.array(sg)
            if not any(np.allclose(np.abs(g @ h.T), np.eye(3)) for h in allv):
                allv.append(g)
    tv = allv[:a.tvar]
    R = roots_sqrt2(); Sh = shell4()
    cand = [Sh, R]  # frame F1: shell + sqrt2 roots
    Ms = list(perfect_matchings(S))
    order = [0] + list(rng.permutation(range(1, len(Ms))))
    base = [(0, 1), (2, 3), (4, 5), (6, 7)]
    i0 = Ms.index(base)
    order = [i0] + [i for i in order if i != i0]
    for mi in order[:a.nmatch]:
        for f in tv:
            F = frame(Ms[mi], T, f)
            if 'r' in a.Fms: cand.append(R @ F.T)
            if 's' in a.Fms: cand.append(Sh @ F.T)
    V = np.vstack(cand)
    # dedupe
    key = np.round(V * 1e6).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    V = V[np.sort(idx)]
    n = len(V)
    print('candidates', n, file=sys.stderr)
    np.save(a.out + '.npy', V)
    m = 0
    with open(a.out + '.graph', 'wb') as fo:
        np.array([n, 0], dtype=np.int32).tofile(fo)
        for s0 in range(0, n, 4096):
            G = V[s0:s0 + 4096] @ V.T
            for i in range(G.shape[0]):
                row = np.nonzero(G[i] > 2 + 1e-9)[0]
                row = row[row != s0 + i].astype(np.int32)
                np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo); m += len(row)
    print('edges', m // 2, 'avg deg %.0f' % (m / n), file=sys.stderr)
    X = np.loadtxt('records/N604/config_float.txt')
    kv = {tuple(k): i for i, k in enumerate(np.round(V * 1e6).astype(np.int64))}
    init = [kv.get(tuple(k)) for k in np.round(X * 1e6).astype(np.int64)]
    print('604 found', sum(i is not None for i in init), file=sys.stderr)
    open(a.out + '_init.txt', 'w').write('\n'.join(str(i) for i in init if i is not None))

if __name__ == '__main__':
    main()
