#!/usr/bin/env python3
"""Iterated restacking with general automorphisms (src/autgen).  BFS over configurations: for each config and each
coordinate axis c whose height set is within {0,±1/2,±1} (no cross conflicts possible), compute Aut(E) of the
equator E = X ∩ e_c^perp, and form X_g = E ∪ U ∪ g(L).  Keeps configurations with new invariants.
usage: restack2.py START.txt depth"""
import sys, subprocess, os
import numpy as np
start = sys.argv[1]; depth = int(sys.argv[2])
def inv(Y):
    G = Y @ Y.T; np.fill_diagonal(G, -9)
    contacts = int((np.abs(G - .5) < 1e-7).sum() // 2)
    anti = int(sum(1 for y in Y if (np.abs(Y + y).max(1) < 1e-7).any()))
    vals = tuple(np.unique(np.round(G[np.triu_indices(len(Y), 1)], 5), return_counts=True)[1][-6:])
    return (contacts, anti, vals)
def restacks(Y):
    out = []
    dirs = [np.eye(11)[c] for c in range(11)]
    r2 = 2 ** .5
    f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
    for fi in f:
        v = np.zeros(11); v[8:] = fi; dirs.append(v)
    for c, v in enumerate(dirs):
        Q, _ = np.linalg.qr(np.column_stack([v, np.random.default_rng(0).normal(size=(11, 10))]))
        Rm = Q.T if Q[:, 0] @ v > 0 else -Q.T   # rows: v, then basis of v^perp
        Yr = Y @ Rm.T
        h = Yr[:, 0]
        hs = set(np.round(np.abs(h), 6))
        if len(sys.argv) <= 3 and not hs <= {0.0, 0.5, 1.0}: continue
        E = Yr[np.abs(h) < 1e-9]; U = Yr[h > 1e-9]; L = Yr[h < -1e-9]
        if len(L) == 0 or len(E) < 15: continue
        Ec = E[:, 1:]
        fn = f'/tmp/claude-0/rs2_{os.getpid()}'
        np.savetxt(fn + '.txt', Ec, fmt='%.17g')
        subprocess.run(['./autgen', fn + '.txt', '10', fn + '.perms', '300000'], capture_output=True)
        perms = np.loadtxt(fn + '.perms', dtype=int, ndmin=2)
        seen = set()
        Lc = L[:, 1:]
        for p in perms:
            # g maps Ec[i] -> Ec[p[i]]
            g, *_ = np.linalg.lstsq(Ec, Ec[p], rcond=None)   # Ec @ g = Ec[p]  (row-vector convention)
            gL = Lc @ g
            key = frozenset(tuple(r) for r in np.round(gL * 1e6).astype(np.int64))
            if key in seen: continue
            seen.add(key)
            Z = np.column_stack([L[:, 0], gL])
            Ynew = np.vstack([E, U, Z]) @ Rm
            out.append((c, Ynew))
    return out
Y0 = np.loadtxt(start); Y0 /= np.linalg.norm(Y0, axis=1)[:, None]
known = {inv(Y0): Y0}
frontier = [Y0]
for dpt in range(depth):
    nf = []
    for Y in frontier:
        for c, Yn in restacks(Y):
            G = Yn @ Yn.T; np.fill_diagonal(G, -9)
            if G.max() > .5 + 1e-9: continue
            k = inv(Yn)
            if k in known: continue
            known[k] = Yn; nf.append(Yn)
            j = len(known)
            np.savetxt(f'runs/restack/bfs_{j}.txt', Yn, fmt='%.17g')
            print(f'depth {dpt + 1} axis {c}: new config #{j} contacts {k[0]} antipodal {k[1]}', flush=True)
    frontier = nf
    print('depth', dpt + 1, 'known', len(known), flush=True)
