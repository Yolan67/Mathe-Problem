#!/usr/bin/env python3
"""Fold the 840-point code configuration (R^12 = F1+F2+F3, F3 = coords 8..11) into R^11 with
per-piece parameters.  Pieces = classes of vectors with the same F3-component c (56 classes + the
class c=0).  Piece params: direction in T=R^3 for the image of c (unit vector d, image = |c| mu d),
phase phi on the R^8 part (complex structure on pairs (0,1),(2,3),(4,5),(6,7)), and split angle
(u -> lam u, |c| -> mu |c| with lam^2|u|^2 + mu^2|c|^2 = 4; lam=mu=1 initially).
For classes whose c is 2-dimensional or more, c is mapped by a rotation of span(c) into T
(we simply use |c| times a direction: the class's u-vectors all share the same c, so only |c| matters).
Fitness = MIS size of the union (misg).  Simulated annealing on parameters.
usage: fold840pieces.py seed iters
"""
import sys, os, subprocess, itertools
import numpy as np
sys.path.insert(0, 'scripts')
from build_code582 import find_packing
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(seed)
# 840 with the three factor blocks present: search packing containing {0,1,2,3},{4,5,6,7},{8,9,10,11}
def packing_with_factors():
    import random
    blocks = [frozenset(c) for c in itertools.combinations(range(12), 4)]
    fac = [frozenset(range(0, 4)), frozenset(range(4, 8)), frozenset(range(8, 12))]
    for att in range(2000):
        random.seed(att)
        cur = set(fac)
        order = blocks[:]; random.shuffle(order)
        for b in order:
            if all(len(b & c) <= 2 for c in cur): cur.add(b)
        # local improvement
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
    return None
P = packing_with_factors()
assert P and len(P) == 51, 'packing'
V = []
for i in range(12):
    for s in (2, -2):
        v = np.zeros(12); v[i] = s; V.append(v)
for B in P:
    for sg in itertools.product((1, -1), repeat=4):
        v = np.zeros(12); v[B] = sg; V.append(v)
V = np.array(V)
G12 = V @ V.T; np.fill_diagonal(G12, -9); assert G12.max() <= 2 + 1e-9
U = V[:, :8]; C = V[:, 8:]
keys = [tuple(c) for c in C]
classes = {}
for i, k in enumerate(keys): classes.setdefault(k, []).append(i)
cls = list(classes.keys())
print('classes', len(cls), file=sys.stderr)
J = np.zeros((8, 8))
for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
    J[b, a] = 1; J[a, b] = -1
def build(par):
    Y = np.zeros((len(V), 11))
    for ci, k in enumerate(cls):
        idx = classes[k]
        c = np.array(k); cn = np.linalg.norm(c)
        phi, th, ph, t = par[ci]
        u = U[idx]
        u2 = np.cos(phi) * u + np.sin(phi) * (u @ J.T)
        if cn < 1e-9:
            Y[idx, :8] = u2; continue
        d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        un = np.linalg.norm(u[0])
        # split angle t: total norm 2; base split (un, cn); rotate split by t
        base = np.arctan2(cn, un)
        a = base + t
        Y[idx, :8] = u2 / (un if un > 1e-9 else 1) * 2 * np.cos(a) if un > 1e-9 else 0
        Y[idx, 8:] = 2 * np.sin(a) * d
    return Y
def mis(Y, secs=0.3, s=1, init=None):
    n = len(Y); Gm = Y @ Y.T; np.fill_diagonal(Gm, -9)
    pid = os.getpid()
    with open(f'/tmp/claude-0/f8_{pid}.graph', 'wb') as f:
        np.array([n, 0], dtype=np.int32).tofile(f)
        for i in range(n):
            r = np.nonzero(Gm[i] > 2 + 1e-9)[0].astype(np.int32)
            np.array([len(r)], dtype=np.int32).tofile(f); r.tofile(f)
    cmd = ['./misg', f'/tmp/claude-0/f8_{pid}.graph', '-T', str(secs), '-q', '1', '-s', str(s), '-o', f'/tmp/claude-0/f8_{pid}.sol']
    if init is not None:
        open(f'/tmp/claude-0/f8_{pid}.init', 'w').write('\n'.join(map(str, init))); cmd += ['-i', f'/tmp/claude-0/f8_{pid}.init']
    subprocess.run(cmd, capture_output=True)
    sol = [int(l) for l in open(f'/tmp/claude-0/f8_{pid}.sol')]
    return len(sol), sol
# initial params: direction = normalized projection of c onto first 3 F3 coords (section-like), phase 0, t 0
par = np.zeros((len(cls), 4))
for ci, k in enumerate(cls):
    c = np.array(k, float)[:3]
    if np.linalg.norm(c) < 1e-9: c = rng.normal(size=3)
    c /= np.linalg.norm(c)
    par[ci, 1] = np.arccos(np.clip(c[2], -1, 1)); par[ci, 2] = np.arctan2(c[1], c[0])
Y = build(par); cur, sol = mis(Y, 2.0)
best = cur
print('start', cur, flush=True)
for it in range(ITER):
    p2 = par.copy()
    ci = rng.integers(len(cls))
    p2[ci] += rng.normal(size=4) * np.array([0.5, 0.4, 0.4, 0.15]) * rng.random()
    Y2 = build(p2)
    val, sol2 = mis(Y2, 0.2, it, sol)
    T = 1.5 * (1 - it / ITER) + 0.05
    if val >= cur or rng.random() < np.exp((val - cur) / T):
        par, cur, sol = p2, val, sol2
        if val > best:
            best = val; np.save(f'/tmp/claude-0/fold840_best_{val}_s{seed}.npy', Y2[sol2])
            print('it', it, 'NEW BEST', best, flush=True)
    if it % 100 == 0: print('it', it, 'cur', cur, 'best', best, flush=True)
print('final best', best)
