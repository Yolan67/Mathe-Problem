#!/usr/bin/env python3
"""Scan block-diagonal rotations R between the D11-root frame F2 and the Z^11 norm-4 shell frame F1.
score(R) = 220 + MIS(shell vectors v with R v compatible with sqrt2*D11, conflicts ip>2).
usage: rscan.py seed n_samples"""
import sys, itertools, subprocess, os, random
import numpy as np
r2 = 2 ** .5; r3 = 3 ** .5; phi = (1 + 5 ** .5) / 2
seed = int(sys.argv[1]); NS = int(sys.argv[2])
rng = random.Random(seed)
# shell
S = []
for i in range(11):
    for s in (2, -2):
        v = np.zeros(11); v[i] = s; S.append(v)
for B in itertools.combinations(range(11), 4):
    for sg in itertools.product((1, -1), repeat=4):
        v = np.zeros(11); v[list(B)] = sg; S.append(v)
S = np.array(S)
def rot2(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
f = np.array([[1 / r2, .5, .5], [1 / r2, -.5, -.5], [0, 1 / r2, -1 / r2]])
R3q = np.array([[1, 2, 2], [2, 1, -2], [2, -2, 1]]) / 3
H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
def qL(q):
    a, b, c, d = q
    return np.array([[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]])
def ico3():
    # rotation taking z to icosahedral vertex direction (0,1,phi)/norm
    v = np.array([0, 1, phi]); v /= np.linalg.norm(v)
    a = np.cross([0, 0, 1], v); s = np.linalg.norm(a); c = v[2]; a /= s
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K
blocks = {
    1: [('id1', np.eye(1))],
    2: [('r45', rot2(np.pi / 4)), ('r30', rot2(np.pi / 6)), ('r60', rot2(np.pi / 3)), ('id2', np.eye(2)), ('rat2', np.array([[3, -4], [4, 3]]) / 5)],
    3: [('f', f), ('rat3', R3q), ('ico3', ico3()), ('r45x1', np.block([[rot2(np.pi / 4), np.zeros((2, 1))], [np.zeros((1, 2)), np.eye(1)]])), ('id3', np.eye(3)),
        ('f2', f @ R3q)],
    4: [('H4', H4), ('qL1i', qL([1 / r2, 1 / r2, 0, 0])), ('qLico', qL([phi / 2, .5, 1 / (2 * phi), 0])), ('qLh', qL([.5, .5, .5, .5])), ('id4', np.eye(4)),
        ('r45x2', np.kron(np.eye(2), rot2(np.pi / 4)))],
}
parts = [[2, 2, 2, 2, 3], [4, 4, 3], [2, 2, 4, 3], [3, 3, 3, 2], [4, 4, 2, 1], [2, 2, 2, 2, 2, 1], [4, 3, 2, 2], [3, 3, 4, 1], [4, 4, 1, 1, 1], [3, 3, 3, 1, 1]]
def build(spec):
    R = np.zeros((11, 11)); o = 0; names = []
    for sz, (nm, M) in spec:
        R[o:o + sz, o:o + sz] = M; o += sz; names.append(nm)
    return R, names
def mis(V):
    n = len(V)
    if n == 0: return 0
    G = V @ V.T; np.fill_diagonal(G, -99)
    fn = f'/tmp/claude-0/rs_{os.getpid()}'
    with open(fn + '.graph', 'wb') as fo:
        np.array([n, 0], dtype=np.int32).tofile(fo)
        for i in range(n):
            row = np.nonzero(G[i] > 2 + 1e-9)[0].astype(np.int32)
            np.array([len(row)], dtype=np.int32).tofile(fo); row.tofile(fo)
    subprocess.run(['./misg', fn + '.graph', '-o', fn + '.sol', '-T', '2', '-q', '1', '-s', '1'], capture_output=True)
    return sum(1 for _ in open(fn + '.sol'))
best = 0
seen = set()
for it in range(NS):
    part = parts[rng.randrange(len(parts))] if it else [2, 2, 2, 2, 3]
    spec = [(sz, blocks[sz][rng.randrange(len(blocks[sz]))] if it else ((sz, ('r45', rot2(np.pi / 4))) if sz == 2 else (sz, ('f', f)))[1]) for sz in part]
    R, names = build(spec)
    key = tuple(names)
    if key in seen: continue
    seen.add(key)
    W = S @ R.T                     # shell vectors expressed in the D11 frame
    top2 = np.sort(np.abs(W), 1)[:, -2:].sum(1)
    C = W[top2 <= r2 + 1e-9]
    sc = 220 + mis(C)
    if sc > best: best = sc
    print(f'{sc} {len(C)} {"-".join(names)}  best {best}', flush=True)
