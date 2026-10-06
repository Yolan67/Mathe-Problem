#!/usr/bin/env python3
"""Pieces construction in R^11 = C^5 (+) R based on K12 / G33.
Base sets in C^5 = v^perp: R (270 roots, |x|^2=6), P0,P1,P2 (80 each, |x|^2=4.5).
A piece = (base, phase phi, lam, h): points (lam*e^{i phi} x, h), with lam^2 |x|^2 + h^2 = 6.
Optional poles (0, ±sqrt6).  Constraint: real ip <= 3 for all distinct points.
Max over a piece pair = max over Hermitian ip values c of lam lam' Re(e^{i dphi} c) + h h'."""
import itertools, sys, numpy as np
V = np.load('/tmp/claude-0/k12_min.npy'); v = np.load('/tmp/claude-0/k12_v.npy')
h = V @ v.conj()
bases = {'R': [x for x, c in zip(V, h) if abs(c) < 1e-9]}
cl = {}
for x, c in zip(V, h):
    if abs(abs(c) - 3) < 1e-9:
        k = int(round(np.degrees(np.angle(c)) % 360 / 60)) % 3
        xp = x - (c / 6) * v
        cl.setdefault(k, {})
        cl[k][tuple(np.round(xp, 9))] = xp
for k in range(3): bases['P%d' % k] = list(cl[k].values())
names = ['R', 'P0', 'P1', 'P2']
B = {n: np.array(bases[n]) for n in names}
sq = {n: float(np.round((np.abs(B[n][0]) ** 2).sum(), 6)) for n in names}
print({n: (len(B[n]), sq[n]) for n in names}, file=sys.stderr)
# Hermitian ip value sets between bases (excluding identical vectors for same base)
H = {}; Hin = {}
for a in names:
    for b in names:
        G = B[a] @ B[b].conj().T
        vals = G.ravel()
        H[(a, b)] = np.unique(np.round(vals, 9))          # all pairs (distinct pieces)
        if a == b:
            Hin[a] = np.unique(np.round(vals[np.abs(vals - sq[a]) > 1e-9], 9))  # within one piece
        print(a, b, len(H[(a, b)]), 'max |c| %.3f' % np.abs(H[(a, b)]).max(), file=sys.stderr)
np.save('/tmp/claude-0/g33_H.npy', H, allow_pickle=True)

def piece_pair_max(p, q):
    (a, fa, la, ha), (b, fb, lb, hb) = p, q
    c = H[(a, b)]
    return (la * lb * (np.exp(1j * (fa - fb)) * c).real).max() + ha * hb

def check(pieces, poles):
    worst = -9
    for i in range(len(pieces)):
        a, fa, la, ha = pieces[i]
        # internal: same piece distinct points
        c = Hin[a]
        worst = max(worst, la * la * c.real.max() + ha * ha)
        for j in range(i + 1, len(pieces)):
            worst = max(worst, piece_pair_max(pieces[i], pieces[j]))
        if poles:
            worst = max(worst, abs(ha) * 6 ** .5)
    return worst

if __name__ == '__main__':
    # sanity: Ganzhinov-like: R at h=0, P0 at ±sqrt1.5
    pcs = [('R', 0, 1, 0), ('P0', 0, 1, 1.5 ** .5), ('P0', 0, 1, -1.5 ** .5)]
    print('section 432 worst', check(pcs, True))
