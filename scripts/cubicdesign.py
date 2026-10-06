#!/usr/bin/env python3
"""Design: cubic graph H on 8 S-points, proper 3-edge-colouring by square planes {xy,xz,yz};
pair-items: each H-edge P with its square (4 w's (±1,±1,0)-type) x 4 sign patterns = 16 vectors.
Triples t with line k in {x,y,z}: if t contains an H-edge P, k must be normal of colour(P).
Blocks: 4-sets with no used triple, pairwise |∩|<=2.  Core: ±2e_k (6).  Axes ±2e_s (16).
Maximise #triples + #blocks (weight 16 each) via MIS (misg) over items."""
import itertools, subprocess, sys, json
import numpy as np

def cubic_graphs_8():
    # enumerate cubic graphs on 8 labelled vertices with a proper 3-edge-colouring (perfect matchings M1,M2,M3)
    V = list(range(8))
    def pms(pts):
        if not pts: yield []; return
        a = pts[0]
        for i in range(1, len(pts)):
            rest = pts[1:i] + pts[i+1:]
            for m in pms(rest): yield [(a, pts[i])] + m
    allpm = [frozenset(frozenset(e) for e in m) for m in pms(V)]
    seen = set(); out = []
    for m1 in allpm[:1]:   # WLOG first colour class fixed
        for m2 in allpm:
            if m1 & m2: continue
            for m3 in allpm:
                if m3 & m1 or m3 & m2: continue
                key = (m1, m2, m3)
                out.append(key)
    return out

def solve(col, secs=3, seed=1):
    # col: dict edge(frozenset)->colour 0,1,2 (0:xy->line z(2), 1:xz->line y(1), 2:yz->line x(0))
    normal = {0: 2, 1: 1, 2: 0}
    items = []
    for t in itertools.combinations(range(8), 3):
        hedges = [frozenset(p) for p in itertools.combinations(t, 2) if frozenset(p) in col]
        lines = {normal[col[e]] for e in hedges}
        if len(lines) > 1: continue
        cand = list(lines) if lines else [0, 1, 2]
        for k in cand: items.append(('T', t, k))
    for B in itertools.combinations(range(8), 4): items.append(('B', B, None))
    n = len(items)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            c = len(set(a[1]) & set(b[1]))
            bad = False
            if a[0] == 'T' and b[0] == 'T':
                if c == 3: bad = True
                elif c == 2 and a[2] == b[2]: bad = True
                elif c == 2:
                    pass  # different lines (axes) are orthogonal
            elif a[0] == 'B' and b[0] == 'B':
                bad = c > 2
            else:
                bad = c == 3   # triple inside block
            if bad: adj[i].append(j); adj[j].append(i)
    with open('/tmp/claude-0/cd.graph', 'wb') as f:
        np.array([n, 0], dtype=np.int32).tofile(f)
        for i in range(n): np.array([len(adj[i])] + adj[i], dtype=np.int32).tofile(f)
    r = subprocess.run(['./misg', '/tmp/claude-0/cd.graph', '-T', str(secs), '-q', '1', '-s', str(seed), '-o', '/tmp/claude-0/cd.sol'], capture_output=True, text=True)
    best = int(r.stdout.split()[-1])
    sol = [items[int(l)] for l in open('/tmp/claude-0/cd.sol')]
    return best, sol

def main():
    gr = cubic_graphs_8()
    print('coloured cubic graphs (labelled, first class fixed):', len(gr), file=sys.stderr)
    import random
    random.seed(1)
    best_all = 0
    seen_iso = set()
    for (m1, m2, m3) in gr:
        col = {}
        for c, m in enumerate((m1, m2, m3)):
            for e in m: col[e] = c
        # crude isomorphism filter: sorted degree of triangles / 4-cycles
        edges = list(col)
        tri = sum(1 for t in itertools.combinations(range(8), 3) if all(frozenset(p) in col for p in itertools.combinations(t, 2)))
        sq = 0
        key = (tri,)
        best, sol = solve(col, secs=2)
        nT = sum(1 for s in sol if s[0] == 'T'); nB = len(sol) - nT
        total = 16 * best + 192 + 6 + 16
        if best > best_all or random.random() < 0.002:
            print('triangles', tri, 'triples', nT, 'blocks', nB, 'sum', best, '=> total', total, flush=True)
        if best > best_all:
            best_all = best
            json.dump({'col': [[sorted(e), c] for e, c in col.items()], 'sol': [[s[0], list(s[1]), s[2]] for s in sol]}, open('/tmp/claude-0/cd_best.json', 'w'))
    print('BEST sum', best_all, 'total', 16 * best_all + 214)

if __name__ == '__main__':
    main()
