#!/usr/bin/env python3
"""Symmetry-reduced global optimisation of kissing configurations in R^11.
Configuration = union of generic orbits G.x_i of m representatives (|G|*m points).
Groups (R^11 = C^5 (+) R, complex coordinate j = real coords 2j,2j+1, real line = coord 10):
  c11s   : C11, z -> zeta z (scalar on C^5)
  c11p   : C11, cyclic shift of 11 real coordinates
  f55    : C11 (diag zeta^r, r in QR11 = 1,3,9,5,4) x| C5 (cyclic perm of the 5 complex coords)
  c5:k1,..,k5 : C5 with plane frequencies k_j
  c55:k1,..,k5: C55 with plane frequencies k_j (mod 55)
Objective: penalty sum max(0,c-t)^2 over all pairs with continuation t -> 0.5, then LSE minimax.
usage: symopt.py GROUP m restarts seed outprefix
"""
import sys, time
import numpy as np
grp, m, R, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rng = np.random.default_rng(seed)
def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def planes(angles):
    M = np.zeros((11, 11)); M[10, 10] = 1
    for j, a in enumerate(angles): M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = rot2(a)
    return M
gens = []
if grp == 'c11s': gens = [planes([2 * np.pi / 11] * 5)]
elif grp == 'c11p': gens = [np.roll(np.eye(11), 1, axis=0)]
elif grp == 'f55':
    qr = [1, 3, 9, 5, 4]
    gens.append(planes([2 * np.pi * r / 11 for r in qr]))
    P = np.zeros((11, 11)); P[10, 10] = 1
    for j in range(5):  # coordinate j (residue qr[j]) -> coordinate j+1 (residue 3*qr[j])
        k = (j + 1) % 5
        P[2 * k:2 * k + 2, 2 * j:2 * j + 2] = np.eye(2)
    gens.append(P)
elif grp.startswith('c5:'):
    ks = list(map(int, grp[3:].split(','))); gens = [planes([2 * np.pi * k / 5 for k in ks])]
elif grp.startswith('c55:'):
    ks = list(map(int, grp[4:].split(','))); gens = [planes([2 * np.pi * k / 55 for k in ks])]
# closure
els = [np.eye(11)]
frontier = [np.eye(11)]
while frontier:
    nf = []
    for g in frontier:
        for h in gens:
            x = h @ g
            if not any(np.abs(x - e).max() < 1e-9 for e in els): els.append(x); nf.append(x)
    frontier = nf
Gm = np.array(els); order = len(Gm)
N = order * m
print(f'group {grp} order {order}, m={m}, N={N}', flush=True)
iu = np.triu_indices(N, 1)
def full(Y):
    Xn = Y / np.linalg.norm(Y, axis=1)[:, None]
    return np.einsum('gab,mb->gma', Gm, Xn).reshape(N, 11), Xn
def energy(Y, t, mode, beta):
    X, Xn = full(Y)
    C = X @ X.T
    np.fill_diagonal(C, -2)
    if mode == 0:
        D = np.maximum(C - t, 0)
        E = (D ** 2).sum() / 2
        W = 2 * D
    else:
        mx = C.max()
        Wx = np.exp(beta * (C - mx)) * (C > t)
        Z = Wx.sum() / 2
        E = mx + np.log(Z) / beta
        W = Wx / Z
    GX = W @ X                     # dE/dX (each pair counted via symmetric W)
    GX = GX.reshape(order, m, 11)
    Gn = np.einsum('gab,gma->mb', Gm, GX)   # pull back: sum_g g^T grad
    nrm = np.linalg.norm(Y, axis=1)[:, None]
    Gy = (Gn - (Gn * Xn).sum(1)[:, None] * Xn) / nrm
    return E, Gy, C
def lbfgs(Y, t, mode=0, beta=0, iters=3000, tol=1e-30):
    mem = []
    E, G, C = energy(Y, t, mode, beta)
    for it in range(iters):
        q = -G.ravel().copy(); al = []
        for s, y, r in reversed(mem):
            a = r * (s @ q); al.append(a); q -= a * y
        if mem:
            s, y, r = mem[-1]; q *= (s @ y) / (y @ y)
        else:
            q *= 1e-2 / (np.linalg.norm(q) + 1e-300)
        for (s, y, r), a in zip(mem, reversed(al)):
            b = r * (y @ q); q += s * (a - b)
        d = q.reshape(Y.shape)
        gd = (G * d).sum()
        if gd >= 0: mem = []; d = -G * 1e-2 / (np.linalg.norm(G) + 1e-300); gd = (G * d).sum()
        mx = np.abs(d).max(); step = min(1.0, 0.05 / (mx + 1e-300))
        ok = False
        for ls in range(30):
            Y2 = Y + step * d
            E2, G2, C2 = energy(Y2, t, mode, beta)
            if E2 <= E + 1e-4 * step * gd: ok = True; break
            step *= .5
        if not ok:
            if not mem: break
            mem = []; continue
        s = (Y2 - Y).ravel(); yv = (G2 - G).ravel(); sy = s @ yv
        if sy > 1e-30:
            mem.append((s, yv, 1 / sy))
            if len(mem) > 8: mem.pop(0)
        Y, E, G, C = Y2 / np.linalg.norm(Y2, axis=1)[:, None], E2, G2, C2
        if mode == 0 and E < tol: break
    return Y, E, C
best = 9; t0 = time.time()
for r in range(R):
    Y = rng.normal(size=(m, 11))
    for t in [0.3, 0.4, 0.45, 0.48, 0.5]:
        Y, E, C = lbfgs(Y, t, 0, 0, 1500)
    mc = C.max()
    if mc < best + 0.01:
        for beta in [100, 300, 1000, 3000, 10000, 30000]:
            Y, E, C = lbfgs(Y, 0.3, 1, beta, 800)
        mc = C.max()
    if mc < best:
        best = mc
        X, _ = full(Y)
        np.savetxt(f'{out}_best.txt', X, fmt='%.17g')
    print(f'restart {r} maxcos {mc:.6f} best {best:.6f} ({time.time() - t0:.0f}s)', flush=True)
