#!/usr/bin/env python3
"""Restacked 604s: for a coordinate axis v = e_c with heights in {0,±1/2,±1}, the equator E = {x : x_c = 0},
upper half U (x_c > 0), lower half L (x_c < 0).  For every signed permutation g of the other 10 coordinates with
g(E) = E, the configuration (E, U, poles, g(L)) is valid.  Enumerate such g (pair-structure-aware search over
all signed perms of the 10 coordinates via sign propagation), keep distinct configurations, report invariants.
usage: restack.py c"""
import sys, itertools, time
import numpy as np
from collections import defaultdict
c = int(sys.argv[1])
X = np.loadtxt('records/N604/config_float.txt'); X /= np.linalg.norm(X, axis=1)[:, None]
h = X[:, c]
print('heights', sorted(set(np.round(h, 6))))
E = X[np.abs(h) < 1e-9]; U = X[h > 1e-9]; L = X[h < -1e-9]
print('E', len(E), 'U', len(U), 'L', len(L))
others = [i for i in range(11) if i != c]
KE = np.round(E * 1e6).astype(np.int64)
keyE = {tuple(r) for r in KE}
absclass = defaultdict(list)
for r in KE: absclass[tuple(np.abs(r))].append(np.sign(r))
# column profiles to restrict permutations
prof = {i: tuple(sorted(np.round(np.abs(E[:, i]), 6))) for i in others}
classes = defaultdict(list)
for i in others: classes[prof[i]].append(i)
print('column classes', [len(v) for v in classes.values()])
t0 = time.time(); gs = []
def perms_respecting():
    groups = list(classes.values())
    for combo in itertools.product(*[itertools.permutations(g) for g in groups]):
        p = list(range(11))
        for g, img in zip(groups, combo):
            for a, b in zip(g, img): p[a] = b
        yield p
cnt = 0
for p in perms_respecting():
    cnt += 1
    P = np.zeros_like(KE); P[:, p] = KE
    cands = [dict()]; covered = set(); ok = True
    for r in P:
        supp = tuple(np.nonzero(r)[0])
        if set(supp) <= covered: continue
        cls = absclass.get(tuple(np.abs(r)))
        if cls is None: ok = False; break
        sr = np.sign(r)
        opts = {tuple((sy * sr)[list(supp)]) for sy in cls}
        new = {}
        for cd in cands:
            for o in opts:
                if all(cd.get(i, v) == v for i, v in zip(supp, o)):
                    d = dict(cd); d.update(zip(supp, o)); new[tuple(sorted(d.items()))] = d
        cands = list(new.values()); covered |= set(supp)
        if not cands: ok = False; break
        if covered >= set(others): break
    if not ok: continue
    for cd in cands:
        s = np.ones(11, int)
        for i in others: s[i] = cd.get(i, 1)
        Q = P * s
        if all(tuple(r) in keyE for r in Q): gs.append((np.array(p), s))
print('perms tried', cnt, 'Aut(E) signed perms:', len(gs), 'time %.0f' % (time.time() - t0))
keyL = {tuple(r) for r in np.round(L * 1e6).astype(np.int64)}
seen = {}
for p, s in gs:
    gL = np.zeros_like(L); gL[:, p] = L; gL *= s
    k = frozenset(tuple(r) for r in np.round(gL * 1e6).astype(np.int64))
    if k in seen: continue
    seen[k] = gL
print('distinct lower layers g(L):', len(seen))
out = []
for j, gL in enumerate(seen.values()):
    Y = np.vstack([E, U, gL])
    G = Y @ Y.T; np.fill_diagonal(G, -9)
    contacts = int((np.abs(G - .5) < 1e-7).sum() // 2)
    anti = sum(1 for y in Y if (np.abs(Y + y).max(1) < 1e-7).any())
    same = len({tuple(r) for r in np.round(gL * 1e6).astype(np.int64)} & keyL)
    out.append((contacts, anti, same, j))
    np.savetxt(f'runs/restack/c{c}_v{j}.txt', Y, fmt='%.17g')
    assert G.max() < .5 + 1e-9
from collections import Counter
print('variants (contacts, antipodal pts, |gL & L|):', Counter((a, b, c2) for a, b, c2, _ in out).most_common(20))
