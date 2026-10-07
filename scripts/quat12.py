#!/usr/bin/env python3
"""R^12 = H^3 quaternionic 2O framework, orbit-MIS under the diagonal LEFT action of a finite group G of unit
quaternions (x1,x2,x3) -> (g x1, g x2, g x3).  Candidates (norm 4): patterns (4,0,0) 2*O; (2,2,0) sqrt2*O x sqrt2*O;
(2,1,1) sqrt2*O x O x O (all slot permutations).  O = 2O (48 unit quaternions).
Writes a weighted orbit graph for misw (weights = orbit sizes).
usage: quat12.py OUT [group: 2T|Q8|2O|1] [patterns: e.g. 400,220,211]"""
import sys, itertools
import numpy as np
out = sys.argv[1]; gname = sys.argv[2] if len(sys.argv) > 2 else '2T'
pats = sys.argv[3].split(',') if len(sys.argv) > 3 else ['400', '220', '211']
r2 = 2 ** .5
def uniq(vs):
    k = np.round(np.array(vs) * 1e6).astype(np.int64); _, i = np.unique(k, axis=0, return_index=True)
    return np.array(vs)[np.sort(i)]
U = [v for v in itertools.product((1, -1, 0), repeat=4) if sum(abs(x) for x in v) == 1]
U += [tuple(s / 2 for s in sg) for sg in itertools.product((1, -1), repeat=4)]
U = uniq(np.array(U, float))
D = uniq(np.array([np.array(p, float) * np.array(sg) / r2 for p in set(itertools.permutations((1, 1, 0, 0)))
                   for sg in itertools.product((1, -1), repeat=4)]))
O = np.vstack([U, D]); assert len(O) == 48
def qmul(a, b):  # a, b: (...,4) real-first quaternions
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0); b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack([a0*b0 - a1*b1 - a2*b2 - a3*b3, a0*b1 + a1*b0 + a2*b3 - a3*b2,
                     a0*b2 - a1*b3 + a2*b0 + a3*b1, a0*b3 + a1*b2 - a2*b1 + a3*b0], -1)
G = {'2T': U, '2O': O, 'Q8': U[np.abs(U).max(1) > 0.9], '1': np.array([[1., 0, 0, 0]])}[gname]
print('group order', len(G), flush=True)
V = []
z = np.zeros(4)
def emb(parts):  # parts: list of 3 quaternions
    return np.concatenate(parts)
if '400' in pats:
    for s in range(3):
        for u in O:
            p = [z, z, z]; p[s] = 2 * u; V.append(emb(p))
if '220' in pats:
    for s, t in [(0, 1), (0, 2), (1, 2)]:
        for a in O:
            for b in O:
                p = [z, z, z]; p[s] = r2 * a; p[t] = r2 * b; V.append(emb(p))
if '211' in pats:
    for s in range(3):
        o = [t for t in range(3) if t != s]
        for a in O:
            for b in O:
                for c in O:
                    p = [z, z, z]; p[s] = r2 * a; p[o[0]] = b; p[o[1]] = c; V.append(emb(p))
V = np.array(V); assert np.allclose((V ** 2).sum(1), 4)
n = len(V); print('candidates', n, flush=True)
key = {tuple(k): i for i, k in enumerate(np.round(V * 1e6).astype(np.int64))}
# orbits under diagonal left multiplication
orb = -np.ones(n, np.int64); reps = []; sizes = []
Q = V.reshape(n, 3, 4)
for i in range(n):
    if orb[i] >= 0: continue
    imgs = qmul(G[:, None, :], Q[i][None, :, :]).reshape(len(G), 12)
    ids = {key[tuple(k)] for k in np.round(imgs * 1e6).astype(np.int64)}
    oid = len(reps)
    for j in ids: orb[j] = oid
    reps.append(i); sizes.append(len(ids))
reps = np.array(reps); sizes = np.array(sizes); no = len(reps)
print('orbits', no, 'size histogram', np.unique(sizes, return_counts=True), flush=True)
# self-compatibility
members = [[] for _ in range(no)]
for j in range(n): members[orb[j]].append(j)
selfok = np.array([((V[m] @ V[m].T)[~np.eye(len(m), dtype=bool)] <= 2 + 1e-9).all() for m in members])
print('self-compatible orbits', selfok.sum(), flush=True)
good = np.nonzero(selfok)[0]
# orbit conflict: rep(A) vs all members of B  (G-invariance makes this sufficient)
R = V[reps[good]]
conf_lists = []
pos = -np.ones(no, np.int64); pos[good] = np.arange(len(good))
for s0 in range(0, len(good), 256):
    M = R[s0:s0 + 256] @ V.T > 2 + 1e-9
    for ii in range(M.shape[0]):
        bad = np.unique(orb[np.nonzero(M[ii])[0]])
        bad = pos[bad]; bad = bad[bad >= 0]; bad = bad[bad != s0 + ii]
        conf_lists.append(bad.astype(np.int32))
ng = len(good); m = sum(len(c) for c in conf_lists)
print('orbit graph n', ng, 'edges', m // 2, flush=True)
with open(out + '.wgraph', 'wb') as fo:
    np.array([ng, 0], dtype=np.int32).tofile(fo)
    for ii in range(ng):
        np.array([sizes[good[ii]], len(conf_lists[ii])], dtype=np.int32).tofile(fo); conf_lists[ii].tofile(fo)
np.save(out + '_V.npy', V); np.save(out + '_orb.npy', orb); np.save(out + '_good.npy', good)
