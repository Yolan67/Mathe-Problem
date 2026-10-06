#!/usr/bin/env python3
"""Generic group-orbit search for kissing configurations in R^11 = 1^perp in R^12.

Group = permutation group on 12 points (numpy array of permutations).  A configuration is a
union of G-orbits of representatives x_a; each x_a lives in span{1_B : B in P_a} ∩ 1^perp where
P_a is the orbit partition of the blockwise stabilizer K_a (orbit size |G|/|K_a|).
Optionally (--pm) the group is G x {±1}.

usage: orbits2.py GROUP N [--starts 40] [--maxorb 5] [--pm] [--iters 1500] [--seed 1]
GROUP: psl211 pgl211 m12 m11pt f55 f110 c11 ...
"""
import argparse, itertools, json, sys, time
import numpy as np
sys.path.insert(0, 'scripts')
from orbits import mobius, closure, INF

def load_group(name):
    if name in ('psl211', 'pgl211'):
        gens = [mobius(1, 1, 0, 1), mobius(0, 10, 1, 0)]
        if name == 'pgl211': gens.append(mobius(2, 0, 0, 1))
        G = closure(gens)
    elif name in ('m12', 'm11pt'):
        d = json.load(open('configs/m12_gens.json'))
        G = closure([tuple(g) for g in d['gens']])
        if name == 'm11pt':
            G = [g for g in G if g[INF] == INF]
    elif name == 'f55':
        G = closure([mobius(1, 1, 0, 1), mobius(4, 0, 0, 1)])
    elif name == 'f110':
        G = closure([mobius(1, 1, 0, 1), mobius(2, 0, 0, 1)])
    elif name == 'c11':
        G = closure([mobius(1, 1, 0, 1)])
    elif name == 'psl211fix':  # A5 subgroups etc can be added
        raise SystemExit('unknown')
    else:
        raise SystemExit('unknown group ' + name)
    return np.array(sorted(G), dtype=np.int64)

def orbit_partition_of(Gsub):
    parent = list(range(12))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for g in Gsub:
        for i in range(12):
            a, b = find(i), find(int(g[i]))
            if a != b: parent[a] = b
    blocks = {}
    for i in range(12): blocks.setdefault(find(i), []).append(i)
    return tuple(sorted(tuple(b) for b in blocks.values()))

def blockwise_stab(Gp, P):
    mask = np.ones(len(Gp), bool)
    for B in P:
        mask &= np.isin(Gp[:, list(B)], B).all(1)
    return Gp[mask]

def orbit_types(Gp, rng, nsamp=3000):
    cands = set()
    idx = rng.integers(len(Gp), size=nsamp)
    for i in idx:
        cands.add(orbit_partition_of([Gp[i]]))
    for _ in range(nsamp):
        i, j = rng.integers(len(Gp), size=2)
        cands.add(orbit_partition_of([Gp[i], Gp[j]]))
    types = {}
    for P in cands:
        if len(P) < 2: continue
        K = blockwise_stab(Gp, P)
        Ps = orbit_partition_of(K)
        if len(Ps) < 2: continue
        K = blockwise_stab(Gp, Ps)
        key = (len(Gp) // len(K), tuple(sorted(len(B) for B in Ps)), len(K))
        if key not in types:
            types[key] = Ps
    out = []
    for (s, bs, k), Ps in types.items():
        out.append(dict(size=s, dim=len(Ps) - 1, P=Ps, K=k))
    out.sort(key=lambda t: (-t['size'], -t['dim']))
    return out

def fix_basis(P):
    M = np.zeros((len(P), 12))
    for k, B in enumerate(P): M[k, list(B)] = 1
    M = M - M.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:(S > 1e-9).sum()]

class Problem:
    def __init__(self, Gp, bases, pm):
        self.Gp = Gp; self.Ginv = np.argsort(Gp, axis=1); self.B = bases; self.pm = pm
    def eval(self, thetas, t):
        xs = []
        for th, B in zip(thetas, self.B):
            v = th @ B; n = np.linalg.norm(v); xs.append((v / n, n))
        m = len(xs); grads = [np.zeros(12) for _ in range(m)]
        E = 0.0; mc = -2.0
        for a in range(m):
            xa = xs[a][0]
            pre_a = xa[self.Gp]          # rows: g^{-1} x_a
            for b in range(a, m):
                xb = xs[b][0]
                imgs_b = xb[self.Ginv]   # rows: g x_b
                c = imgs_b @ xa          # <x_a, g x_b>
                if a == b:
                    valid = c < 1 - 1e-6
                    if self.pm: valid &= c > -1 + 1e-6
                else:
                    valid = np.ones_like(c, bool)
                cv = np.where(valid, c, -2.0)
                mc = max(mc, cv.max())
                if self.pm: mc = max(mc, np.where(valid, -c, -2.0).max())
                v = np.where(valid & (c > t), c - t, 0.0)
                w = 2 * v
                if self.pm:
                    v2 = np.where(valid & (-c > t), -c - t, 0.0)
                    w = w - 2 * v2
                    E += (v2 ** 2).sum()
                E += (v ** 2).sum()
                if np.any(w != 0):
                    grads[a] += w @ imgs_b
                    grads[b] += w @ pre_a
        gth = []
        for (x, n), g, B in zip(xs, grads, self.B):
            g = (g - (g @ x) * x) / n
            gth.append(B @ g)
        return E, mc, gth, [x for x, n in xs]

def optimize(prob, dims, rng, starts, iters):
    best = 9; bestx = None
    for st in range(starts):
        th = [rng.normal(size=d) for d in dims]
        m1 = [np.zeros(d) for d in dims]; m2 = [np.zeros(d) for d in dims]
        lr = 0.03
        for it in range(iters):
            t = 0.5 - (0.003 if it > iters * 0.6 else 0.0)
            E, mc, g, xs = prob.eval(th, t)
            if mc < best: best = mc; bestx = xs
            if mc <= 0.5 - 1e-12 or E == 0: break
            for k in range(len(th)):
                m1[k] = 0.9 * m1[k] + 0.1 * g[k]; m2[k] = 0.999 * m2[k] + 0.001 * g[k] ** 2
                th[k] = th[k] - lr * m1[k] / (np.sqrt(m2[k]) + 1e-12)
                th[k] /= np.linalg.norm(th[k])
            if it % 300 == 299: lr *= 0.6
        if best <= 0.5 + 1e-12: break
    return best, bestx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group'); ap.add_argument('N', type=int)
    ap.add_argument('--starts', type=int, default=30); ap.add_argument('--maxorb', type=int, default=5)
    ap.add_argument('--pm', action='store_true'); ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--listonly', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Gp = load_group(a.group)
    T = orbit_types(Gp, rng)
    if a.pm:
        for t in T: t['size'] *= 2   # assume -x not in Gx (checked by internal optimisation)
    print(f'group {a.group} order {len(Gp)} pm={a.pm}: {len(T)} orbit types', flush=True)
    # internal feasibility
    usable = []
    for t in T:
        if t['size'] > a.N: continue
        B = fix_basis(t['P'])
        pr = Problem(Gp, [B], a.pm)
        best, _ = optimize(pr, [B.shape[0]], rng, 6 if B.shape[0] > 1 else 2, 400)
        t['internal'] = best; t['B'] = B
        print('  type size %4d dim %2d blocks %s internal %.5f' % (t['size'], t['dim'], [len(b) for b in t['P']], best), flush=True)
        if best <= 0.5 + 1e-9: usable.append(t)
    sizes = [t['size'] for t in usable]
    combos = []
    for r in range(1, a.maxorb + 1):
        for comb in itertools.combinations_with_replacement(range(len(usable)), r):
            if sum(sizes[c] for c in comb) == a.N: combos.append(comb)
    print(f'usable {len(usable)} combos {len(combos)}', flush=True)
    if a.listonly: return
    res = []
    for comb in combos:
        Bs = [usable[c]['B'] for c in comb]
        pr = Problem(Gp, Bs, a.pm)
        t0 = time.time()
        best, xs = optimize(pr, [B.shape[0] for B in Bs], rng, a.starts, a.iters)
        res.append((best, [usable[c]['size'] for c in comb]))
        print('combo %s dims %s best %.6f (%.0fs)' % ([usable[c]['size'] for c in comb], [usable[c]['dim'] for c in comb], best, time.time() - t0), flush=True)
        if best <= 0.5 + 1e-9:
            fn = f'/tmp/claude-0/orbit_hit_{a.group}_{a.N}_{len(res)}.npy'
            np.save(fn, np.array(xs)); print('HIT saved', fn, flush=True)
    res.sort(key=lambda r: r[0])
    print('BEST', res[:5], flush=True)

if __name__ == '__main__':
    main()
