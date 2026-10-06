#!/usr/bin/env python3
"""Compute G33 orbit point sets (deduped with tolerance) for small orbit types; save as complex C^5 arrays."""
import numpy as np, sys
G = np.load('/tmp/claude-0/g33_G10.npy').astype(np.float64)
# re-orthogonalise group matrices (float32 storage) via polar decomposition
U, s, Vt = np.linalg.svd(G); G = U @ Vt
T = np.load('/tmp/claude-0/g33_orbittypes.npy', allow_pickle=True).item()
rng = np.random.default_rng(3)
out = {}
for (sz, dim), W in sorted(T.items()):
    if 51840 % sz or sz > 1100: continue
    x = rng.normal(size=W.shape[0]) @ W; x /= np.linalg.norm(x)
    Y = G @ x
    # dedupe with tolerance via sorting on rounded coords at 1e-3 then exact check
    pts = []
    keys = {}
    for y in Y:
        k = tuple(np.round(y, 3))
        if k in keys: continue
        keys[k] = 1; pts.append(y)
    P = np.array(pts)
    # merge near duplicates (distance < 1e-6)
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    keep = []; used = np.zeros(len(P), bool)
    for i in range(len(P)):
        if used[i]: continue
        used |= D[i] < 1e-8; keep.append(i)
    P = P[keep]
    Z = P[:, :5] + 1j * P[:, 5:]
    G2 = (Z @ Z.conj().T).real; np.fill_diagonal(G2, -9)
    print('type', sz, 'dim', dim, '-> points', len(Z), 'internal max cos %.4f' % G2.max(), file=sys.stderr)
    out[len(Z)] = Z
np.save('/tmp/claude-0/g33_bases.npy', out, allow_pickle=True)
