#!/usr/bin/env python3
"""Group-orbit search for kissing configurations in R^11 (Ganzhinov-style).

G acts on 12 points (permutation representation); R^11 = sum-zero subspace of R^12.
Orbit types: for a subgroup H, vectors constant on the H-orbits (blocks) of the 12 points;
a generic such x has stabilizer K = {g : g fixes every block setwise} and orbit size |G|/|K|.
A configuration is a union of orbits; representatives are optimised inside Fix(H).

usage: orbits.py types GROUP            -> list orbit types (size, dim, block sizes)
       orbits.py search GROUP N [starts] [maxorb] -> try all combinations summing to N
GROUP: psl211 | pgl211 | psl211pm | pgl211pm  ("pm": also include -I)
"""
import sys, itertools, json, time
import numpy as np

INF = 11
def inv(a): return pow(a, 9, 11)  # a^(p-2)

def mobius(a, b, c, d):
    def f(x):
        if x == INF:
            return INF if c == 0 else (a * inv(c)) % 11
        den = (c * x + d) % 11
        if den == 0:
            return INF
        return ((a * x + b) * inv(den)) % 11
    return tuple(f(x) for x in range(12))

def closure(gens):
    idp = tuple(range(12))
    elems = {idp}
    frontier = [idp]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                k = tuple(h[g[i]] for i in range(12))
                if k not in elems:
                    elems.add(k); new.append(k)
        frontier = new
    return sorted(elems)

def group(name):
    gens = [mobius(1, 1, 0, 1), mobius(0, 10, 1, 0)]  # x+1, -1/x
    if name.startswith('pgl'):
        gens.append(mobius(2, 0, 0, 1))
    G = closure(gens)
    return G

def compose(g, h):  # (g*h)(i) = g(h(i))
    return tuple(g[h[i]] for i in range(12))

def subgroup(gens):
    return closure(gens) if gens else [tuple(range(12))]

def orbit_partition(H):
    seen = [-1] * 12; blocks = []
    for i in range(12):
        if seen[i] >= 0: continue
        orb = {h[i] for h in H}
        for j in orb: seen[j] = len(blocks)
        blocks.append(tuple(sorted(orb)))
    return tuple(sorted(blocks))

def orbit_types(G, rng, ntries=4000):
    Gs = set(G)
    parts = {}
    Gl = list(G)
    cand_gens = [[g] for g in Gl] + [[Gl[rng.integers(len(Gl))], Gl[rng.integers(len(Gl))]] for _ in range(ntries)]
    seen_sub = set()
    for gens in cand_gens:
        H = subgroup(gens)
        key = frozenset(H)
        if key in seen_sub: continue
        seen_sub.add(key)
        P = orbit_partition(H)
        if len(P) < 2: continue
        # K = blockwise stabilizer
        K = [g for g in G if all(tuple(sorted(g[i] for i in B)) == B for B in P)]
        # canonical up to conjugacy: use sorted invariant (|K|, block sizes) + min image under G
        canon = min(tuple(sorted(tuple(sorted(g[i] for i in B)) for B in P)) for g in G)
        if canon in parts: continue
        parts[canon] = (len(G) // len(K), len(P) - 1, P, len(K))
    return parts

def fix_basis(P):
    """orthonormal basis (rows) of span{1_B} ∩ 1^perp in R^12"""
    M = np.zeros((len(P), 12))
    for k, B in enumerate(P):
        M[k, list(B)] = 1
    # project out all-ones
    M = M - M.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = (S > 1e-9).sum()
    return Vt[:r]

def main():
    mode = sys.argv[1]; gname = sys.argv[2]
    rng = np.random.default_rng(1)
    G = group(gname)
    pm = gname.endswith('pm')
    print('group', gname, 'order', len(G) * (2 if pm else 1), file=sys.stderr)
    types = orbit_types(G, rng)
    tl = sorted(types.values(), key=lambda t: (-t[0], t[1]))
    if pm:
        # with -I: orbit of x under G x {±1}; size doubles unless -x in Gx (checked numerically later)
        tl = [(s * 2, d, P, k) for (s, d, P, k) in tl]
    if mode == 'types':
        for s, d, P, k in tl:
            print(s, d, [len(B) for B in P])
        return
    N = int(sys.argv[3]); starts = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    maxorb = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    Garr = np.array(G)                     # g as permutation: (g x)[g[i]] = x[i]
    Ginv = np.argsort(Garr, axis=1)        # (g x)[j] = x[Ginv[g, j]]
    tl = [t for t in tl if t[1] >= 1 and t[0] <= N]
    bases = [fix_basis(t[2]) for t in tl]
    # single-orbit feasibility: best internal max cos
    feas = []
    for (s, d, P, k), Bm in zip(tl, bases):
        best = 9
        for _ in range(300 if d > 1 else 50):
            th = rng.normal(size=Bm.shape[0]); x = th @ Bm; x /= np.linalg.norm(x)
            imgs = x[Ginv]
            if pm: imgs = np.vstack([imgs, -imgs])
            c = imgs @ x
            c = c[c < 1 - 1e-9]
            best = min(best, c.max())
        feas.append(best)
    ok = [i for i in range(len(tl)) if feas[i] <= 0.5 + 1e-9 or tl[i][1] > 1]
    print('orbit types', len(tl), 'usable', len(ok), file=sys.stderr)
    for i in ok:
        print('  size %d dim %d blocks %s best-internal %.4f' % (tl[i][0], tl[i][1], [len(B) for B in tl[i][2]], feas[i]), file=sys.stderr)
    sizes = [tl[i][0] for i in ok]
    combos = []
    for r in range(1, maxorb + 1):
        for comb in itertools.combinations_with_replacement(range(len(ok)), r):
            if sum(sizes[c] for c in comb) == N:
                combos.append([ok[c] for c in comb])
    print('combos', len(combos), file=sys.stderr)
    results = []
    for comb in combos:
        best, bestX = solve(comb, tl, bases, Ginv, pm, rng, starts)
        results.append((best, comb))
        print('combo', [tl[i][0] for i in comb], [tl[i][1] for i in comb], 'best maxcos %.6f' % best, flush=True)
        if best <= 0.5 + 1e-10:
            np.save('/tmp/claude-0/orbit_hit_%s_%d.npy' % (gname, N), bestX)
            print('HIT', flush=True)
    results.sort()
    print('best overall', results[:5])

def solve(comb, tl, bases, Ginv, pm, rng, starts, iters=600):
    m = len(comb)
    B = [bases[i] for i in comb]
    best = 9; bestX = None
    for st in range(starts):
        X = [rng.normal(size=b.shape[0]) for b in B]
        lr = 0.05
        for it in range(iters):
            xs = [th @ b for th, b in zip(X, B)]
            xs = [x / np.linalg.norm(x) for x in xs]
            grads = [np.zeros(12) for _ in range(m)]
            mc = -2
            t = 0.5 - 0.002 if it > iters // 2 else 0.5
            for a in range(m):
                imgs_a = xs[a][Ginv]  # (|G|,12) images g x_a
                if pm: imgs_a = np.vstack([imgs_a, -imgs_a])
                for b in range(a, m):
                    c = imgs_a @ xs[b]
                    if a == b:
                        mask = c < 1 - 1e-7
                    else:
                        mask = np.ones_like(c, bool)
                    cm = c[mask]
                    if cm.size: mc = max(mc, cm.max())
                    viol = np.where(mask & (c > t), c - t, 0)
                    if viol.any():
                        w = 2 * viol
                        grads[b] += w @ imgs_a
                        # d/dx_a <g x_a, x_b> = g^{-1} x_b
                        # approximate symmetric contribution via images of x_b under g^{-1}
                        # (use: <g x_a, x_b> = <x_a, g^-1 x_b>) -> need g^{-1} x_b
                        if a != b:
                            grads[a] += w @ GinvImgs(xs[b], Ginv, pm)
                        else:
                            grads[a] += w @ GinvImgs(xs[b], Ginv, pm)
            if mc < best:
                best = mc; bestX = [x.copy() for x in xs]
            if mc <= 0.5 - 1e-12:
                break
            for k in range(m):
                gproj = B[k] @ grads[k]
                X[k] = X[k] - lr * gproj
            if it % 200 == 199: lr *= 0.5
        if best <= 0.5 + 1e-10:
            break
    return best, bestX

_cache = {}
def GinvImgs(x, Ginv, pm):
    # rows: g^{-1} x for each g (same ordering as Ginv rows) ; g^{-1} as permutation: (g^{-1}x)[i] = x[g[i]]
    G = np.argsort(Ginv, axis=1)
    r = x[G]
    if pm: r = np.vstack([r, -r])
    return r

if __name__ == '__main__':
    main()
