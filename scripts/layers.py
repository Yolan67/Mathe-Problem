#!/usr/bin/env python3
"""Backbone of the 594..600 constructions (as described in the literature):
pairs P0={0,1},P1={2,3},P2={4,5},P3={6,7}; 6 blocks = unions of two pairs;
3 layers k=8,9,10, each 8 blocks T∪{k}, T a transversal triple (<=1 element per
pair), triples within a layer pairwise meeting in <=1 point, all 24 distinct.
Enumerate all such layer systems (up to search), write blocks."""
import itertools, random, sys
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
trans = []
for miss in range(4):
    others = [p for i, p in enumerate(pairs) if i != miss]
    for bits in itertools.product((0, 1), repeat=3):
        trans.append(tuple(sorted(o[b] for o, b in zip(others, bits))))
assert len(trans) == 32
def ok(a, b): return len(set(a) & set(b)) <= 1
# all 8-subsets of trans pairwise ok (cliques of size 8 in compat graph)
comp = {t: [u for u in trans if u != t and ok(t, u)] for t in trans}
cliques = []
def ext(cl, cand):
    if len(cl) == 8:
        cliques.append(tuple(sorted(cl))); return
    for i, u in enumerate(cand):
        if not cl or u > cl[-1]:
            ext(cl + [u], [w for w in cand[i+1:] if ok(u, w)])
ext([], sorted(trans))
cliques = sorted(set(cliques))
print('8-cliques (layers):', len(cliques))
# triples of disjoint layers
sols = []
for a, b, c in itertools.combinations(range(len(cliques)), 3):
    A, B, C = set(cliques[a]), set(cliques[b]), set(cliques[c])
    if A & B or A & C or B & C: continue
    sols.append((a, b, c))
print('layer systems:', len(sols))
if sols:
    a, b, c = sols[0]
    blocks = [tuple(sorted(p + q)) for p, q in itertools.combinations(pairs, 2)]
    for k, L in zip((8, 9, 10), (cliques[a], cliques[b], cliques[c])):
        for t in L: blocks.append(tuple(sorted(t + (k,))))
    with open('configs/layer30_blocks.txt', 'w') as f:
        for B in blocks: f.write(' '.join(map(str, B)) + '\n')
    print('written', len(blocks))
    # print layers
    for k, L in zip((8, 9, 10), (cliques[a], cliques[b], cliques[c])):
        print(k, L)
import json
json.dump({'cliques': cliques, 'sols': sols}, open('configs/layer_systems.json', 'w'))
