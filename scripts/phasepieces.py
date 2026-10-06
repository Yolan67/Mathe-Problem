#!/usr/bin/env python3
"""Parametric piece MIS in R^11 = C^4 (+) R^3 (pairs {0,1},{2,3},{4,5},{6,7} = complex coords).

Pieces (base vectors from the 12-dim parent of the 604):
  A    : D8 part (112), w = 0
  L1..L4 : lifted spinor layers (128 each), base T-part = ±e (unit)
  S    : pair vectors x frame (96), T-part norm sqrt2
  Z    : core cuboctahedron (12)
Each piece p has parameters: phase phi_p (u -> e^{i phi} u), rotation R_p in SO(3) for T-part
(3 params: rotation vector), and for layers a direction for the base axis (via R_p e_1) and an
angle t_p controlling the split |u|^2 = 4 cos^2 t ... (kept at natural values unless --scale).
Fitness = maximum independent set (misg) of the union of all piece vectors (ip > 2 conflict).
Annealing over parameters, starting from the 604 (L1..L3,A,S,Z natural; L4 random).
"""
import sys, subprocess, itertools, json, os
import numpy as np
r2 = 2 ** .5
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
X604 = np.loadtxt('records/N604/config_float.txt')
A = X604[:112]; Ls = [X604[112 + 128 * k: 112 + 128 * (k + 1)] for k in range(3)]
S = X604[496:592]; Z = X604[592:604]
# 4th layer: unused transversal triples (one per code block): the remaining clique
d = json.load(open('configs/layer_systems.json'))
cl = [tuple(map(tuple, c)) for c in d['cliques']]
used = set()
blocks = [tuple(map(int, l.split())) for l in open('configs/layer30_blocks.txt')]
for B in blocks[6:]: used.add(tuple(sorted(B[:3])))
alltr = set(t for c in cl for t in c)
rest = sorted(alltr - used)
L4 = []
for t in rest:
    for sg in itertools.product((1, -1), repeat=4):
        v = np.zeros(11); v[list(t)] = sg[:3]; v[8] = sg[3]; L4.append(v)   # base axis e_1 (rotated later)
L4 = np.array(L4)
pieces = {'A': A, 'L1': Ls[0], 'L2': Ls[1], 'L3': Ls[2], 'S': S, 'Z': Z, 'L4': L4}
names = list(pieces)
J = np.zeros((8, 8))
for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
    J[b, a] = 1; J[a, b] = -1      # J e_a = e_b, J e_b = -e_a
def rot(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12: return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
def transform(P, par):
    phi, rv = par[0], par[1:4]
    U = P[:, :8]; W = P[:, 8:]
    U2 = np.cos(phi) * U + np.sin(phi) * (U @ J.T)
    W2 = W @ rot(rv).T
    return np.column_stack([U2, W2])
def build(params):
    return np.vstack([transform(pieces[n], params[n]) for n in names]), np.concatenate([[i] * len(pieces[n]) for i, n in enumerate(names)])
def mis(Y, secs=0.25, seed=1, init=None):
    G = Y @ Y.T; np.fill_diagonal(G, -9)
    n = len(Y)
    with open('/tmp/claude-0/pp_%d.graph' % os.getpid(), 'wb') as f:
        np.array([n, 0], dtype=np.int32).tofile(f)
        for i in range(n):
            r = np.nonzero(G[i] > 2 + 1e-9)[0].astype(np.int32)
            np.array([len(r)], dtype=np.int32).tofile(f); r.tofile(f)
    cmd = ['./misg', '/tmp/claude-0/pp_%d.graph' % os.getpid(), '-T', str(secs), '-q', '1', '-s', str(seed), '-o', '/tmp/claude-0/pp_%d.sol' % os.getpid()]
    if init is not None:
        open('/tmp/claude-0/pp_%d.init' % os.getpid(), 'w').write('\n'.join(map(str, init)))
        cmd += ['-i', '/tmp/claude-0/pp_%d.init' % os.getpid()]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    sol = [int(l) for l in open('/tmp/claude-0/pp_%d.sol' % os.getpid())]
    return len(sol), sol
params = {n: np.zeros(4) for n in names}
params['L4'] = np.concatenate([[rng.uniform(0, 2 * np.pi)], rng.normal(size=3)])
Y, lab = build(params)
best, sol = mis(Y, 1.0)
cur = best
print('start', best, flush=True)
T0 = 2.0
for it in range(ITER):
    newp = {n: params[n].copy() for n in names}
    n = names[rng.integers(len(names))] if rng.random() < 0.5 else 'L4'
    sc = 0.3 * rng.random()
    newp[n] = newp[n] + rng.normal(size=4) * sc
    Y2, lab2 = build(newp)
    # warm start: keep previous solution indices that are still valid
    val, sol2 = mis(Y2, 0.25, seed=it, init=sol)
    T = T0 * (1 - it / ITER) + 0.05
    if val >= cur or rng.random() < np.exp((val - cur) / T):
        params, cur, sol = newp, val, sol2
        if val > best:
            best = val
            np.save('/tmp/claude-0/phasepieces_best_%d.npy' % val, Y2[sol2])
            print('it', it, 'NEW BEST', best, flush=True)
    if it % 50 == 0: print('it', it, 'cur', cur, 'best', best, flush=True)
print('final best', best)
