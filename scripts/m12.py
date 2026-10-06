#!/usr/bin/env python3
"""Construct M12 (and M11) as permutation groups on 12 points via the S(5,6,12) Steiner system.
Hexads: PSL(2,11)-orbit of {inf,1,3,4,5,9}.  Elements are determined by images of 5 points.
Writes configs/m12_gens.json with generators (permutations of 0..11, 11 = infinity)."""
import itertools, json, random
import sys
sys.path.insert(0, 'scripts')
from orbits import mobius, closure, INF

def hexads():
    G = closure([mobius(1, 1, 0, 1), mobius(0, 10, 1, 0)])
    base = frozenset([INF, 1, 3, 4, 5, 9])
    H = {frozenset(g[i] for i in base) for g in G}
    return H

def hexad_through(H5, Hs):
    for h in Hs:
        if H5 <= h: return h
    return None

def extend(xs, ys, Hs):
    sig = dict(zip(xs, ys))
    changed = True
    while changed and len(sig) < 12:
        changed = False
        A = list(sig.keys())
        for five in itertools.combinations(A, 5):
            h = hexad_through(frozenset(five), Hs)
            out = [p for p in h if p not in sig]
            if len(out) != 1: continue
            h2 = hexad_through(frozenset(sig[p] for p in five), Hs)
            img = set(sig.values())
            out2 = [p for p in h2 if p not in img]
            if len(out2) != 1: return None
            sig[out[0]] = out2[0]; changed = True
            break
    if len(sig) < 12:
        rest = [p for p in range(12) if p not in sig]
        imgs = [p for p in range(12) if p not in sig.values()]
        for pr in itertools.permutations(imgs):
            s2 = dict(sig); s2.update(zip(rest, pr))
            perm = tuple(s2[i] for i in range(12))
            if all(frozenset(perm[i] for i in h) in Hs for h in Hs):
                return perm
        return None
    perm = tuple(sig[i] for i in range(12))
    if len(set(perm)) != 12: return None
    for h in Hs:
        if frozenset(perm[i] for i in h) not in Hs: return None
    return perm

def main():
    Hs = hexads()
    assert len(Hs) == 132
    random.seed(1)
    gens = []
    while len(gens) < 3:
        ys = random.sample(range(12), 5)
        p = extend([0, 1, 2, 3, 4], ys, Hs)
        if p: gens.append(p)
    G = closure(gens)
    print('M12 order', len(G))
    json.dump({'gens': gens, 'hexads': [sorted(h) for h in Hs]}, open('configs/m12_gens.json', 'w'))

if __name__ == '__main__':
    main()
