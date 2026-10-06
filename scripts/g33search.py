#!/usr/bin/env python3
"""Search piece configurations (G33/K12 family) in R^11.  usage: g33search.py nR nP poles(0/1) [starts]"""
import sys, numpy as np
sys.argv_saved = sys.argv
import importlib.util
spec = importlib.util.spec_from_file_location('gp', 'scripts/g33pieces.py'); gp = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = ['x']; spec.loader.exec_module(gp); sys.argv = _argv
H = gp.H; sq = gp.sq
nR, nP, poles = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 200
rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
kinds = ['R'] * nR + ['P0'] * nP
def pieces(par):
    out = []
    for k, (phi, t) in zip(kinds, par.reshape(-1, 2)):
        lam = np.cos(t) * (6 / sq[k]) ** .5; hh = 6 ** .5 * np.sin(t)
        out.append((k, phi, lam, hh))
    return out
def F(par):
    return gp.check(pieces(par), poles)
best = 9; bestp = None
for s in range(starts):
    par = np.column_stack([rng.uniform(0, 2 * np.pi, len(kinds)), rng.uniform(-1.2, 1.2, len(kinds))]).ravel()
    if nR: par[0] = 0; par[1] = 0 if rng.random() < 0.5 else par[1]
    f = F(par); step = 0.3
    for it in range(1500):
        q = par + rng.normal(size=par.shape) * step * (rng.random(par.shape) < 0.5)
        fq = F(q)
        if fq <= f: par, f = q, fq
        if it % 150 == 149: step *= 0.6
    if f < best: best, bestp = f, par.copy()
total = 270 * nR + 80 * nP + 2 * poles
print(f'nR={nR} nP={nP} poles={poles} total={total} best worst-ip={best:.6f} (need <= 3)', flush=True)
if best <= 3 + 1e-9:
    np.save(f'/tmp/claude-0/g33_sol_{nR}_{nP}_{poles}.npy', bestp)
    print('FEASIBLE', [tuple(np.round(p[1:], 4)) for p in pieces(bestp)])
