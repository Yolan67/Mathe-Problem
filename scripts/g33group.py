#!/usr/bin/env python3
"""Generate G33 (order 51840) as real 10x10 matrices from reflections in the 270 roots; find small orbits."""
import numpy as np, sys
roots = np.load('/tmp/claude-0/g33_roots_c.npy')      # in C^6 (v-perp)
v = np.load('/tmp/claude-0/k12_v.npy'); vn = v / np.linalg.norm(v)
M = np.eye(6, dtype=complex) - np.outer(vn, vn.conj())
U, s, Wh = np.linalg.svd(M); Bc = U[:, :5]
R5 = roots @ Bc.conj()                                  # 270 x 5 complex
def realmat(Mc):  # complex 5x5 acting on C^5 -> real 10x10 on (Re, Im)
    A, B = Mc.real, Mc.imag
    return np.block([[A, -B], [B, A]])
# reflections of order 2: r(x) = x - 2 <x,a> a / |a|^2  (Hermitian)
lines = []
for a in R5:
    an = a / np.linalg.norm(a)
    if not any(abs(abs(np.vdot(an, b)) - 1) < 1e-9 for b in lines): lines.append(an)
print('reflection lines', len(lines), file=sys.stderr)
gens = [realmat(np.eye(5) - 2 * np.outer(a, a.conj())) for a in lines]
def key(Mr): return tuple(np.round(Mr * 1e5).astype(np.int64).ravel())
G = [np.eye(10)]; seen = {key(G[0])}; frontier = [G[0]]
while frontier and len(G) < 200000:
    new = []
    for Mx in frontier:
        for g in gens:
            P = g @ Mx; k = key(P)
            if k not in seen: seen.add(k); G.append(P); new.append(P)
    frontier = new
print('group order (from 6 reflections)', len(G), file=sys.stderr)
G = np.array(G)
np.save('/tmp/claude-0/g33_G10.npy', G.astype(np.float32))
