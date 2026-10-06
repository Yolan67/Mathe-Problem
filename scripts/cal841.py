#!/usr/bin/env python3
"""Calibration: can our optimizer reproduce the 840 -> 841 step in R^12?
Builds the 840 code configuration with the three quaternionic factor blocks, removes the 24-cell on
H3 = coords 8..11 and adds 25 random points near H3.  Writes runs/cal841/init_s{seed}.txt (12 dims)."""
import sys, itertools, random
import numpy as np
sys.path.insert(0, 'scripts')
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
def packing_with_factors():
    blocks = [frozenset(c) for c in itertools.combinations(range(12), 4)]
    fac = [frozenset(range(0, 4)), frozenset(range(4, 8)), frozenset(range(8, 12))]
    for att in range(2000):
        random.seed(att)
        cur = set(fac)
        order = blocks[:]; random.shuffle(order)
        for b in order:
            if all(len(b & c) <= 2 for c in cur): cur.add(b)
        for it in range(20000):
            if len(cur) >= 51: return sorted(sorted(b) for b in cur)
            b = random.choice(blocks)
            if b in cur: continue
            cs = [c for c in cur if len(b & c) > 2]
            if len(cs) == 1 and cs[0] not in fac:
                cur.remove(cs[0]); cur.add(b)
                for d in random.sample(blocks, len(blocks)):
                    if d not in cur and all(len(d & c) <= 2 for c in cur): cur.add(d)
            elif not cs: cur.add(b)
P = packing_with_factors()
V = []
for i in range(12):
    for s in (2, -2):
        v = np.zeros(12); v[i] = s; V.append(v)
for B in P:
    for sg in itertools.product((1, -1), repeat=4):
        v = np.zeros(12); v[B] = sg; V.append(v)
V = np.array(V) / 2
G = V @ V.T; np.fill_diagonal(G, -9); assert len(V) == 840 and G.max() <= .5 + 1e-12
np.savetxt('runs/cal841/c840.txt', V, fmt='%.17g')
h3 = (np.abs(V[:, :8]).sum(1) < 1e-12)
W = V[~h3]
print('removed', h3.sum(), 'kept', len(W))
rng = np.random.default_rng(seed)
Y = rng.normal(size=(25, 12)); Y[:, :8] *= eps
Y /= np.linalg.norm(Y, axis=1)[:, None]
np.savetxt(f'runs/cal841/init_s{seed}.txt', np.vstack([W, Y]), fmt='%.17g')
