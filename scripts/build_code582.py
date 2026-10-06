#!/usr/bin/env python3
"""Construct the classical 582-point kissing configuration in R^11.

{±2 e_i} (22 points)  ∪  {(±1)^4 on B : B in a packing of 4-subsets of {0..10}
with pairwise intersections <= 2} (16 * 35 = 560 points).  A(11,4,4) = 35.
All vectors have norm 4 and pairwise inner products <= 2.
"""
import itertools, random, sys

def find_packing(n=11, k=4, target=35, seed=0):
    rng = random.Random(seed)
    blocks = [frozenset(c) for c in itertools.combinations(range(n), k)]
    conf = {b: [c for c in blocks if c != b and len(b & c) > k - 2] for b in blocks}
    for attempt in range(10000):
        # randomized greedy + simple local search (remove 1, add 2)
        cur = set()
        order = blocks[:]
        rng.shuffle(order)
        for b in order:
            if all(len(b & c) <= k - 2 for c in cur):
                cur.add(b)
        for it in range(20000):
            if len(cur) >= target:
                return sorted(sorted(b) for b in cur)
            # pick a random non-member with exactly one conflict, swap
            b = rng.choice(blocks)
            if b in cur:
                continue
            cs = [c for c in conf[b] if c in cur]
            if len(cs) == 1:
                cur.remove(cs[0])
                cur.add(b)
                # try to greedily add
                for d in rng.sample(blocks, len(blocks)):
                    if d not in cur and all(len(d & c) <= k - 2 for c in cur):
                        cur.add(d)
            elif len(cs) == 0:
                cur.add(b)
    return None

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'configs/code582.txt'
    P = find_packing()
    assert P is not None and len(P) == 35
    pts = []
    for i in range(11):
        for s in (2, -2):
            v = [0] * 11; v[i] = s; pts.append(v)
    for B in P:
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * 11
            for idx, s in zip(B, signs):
                v[idx] = s
            pts.append(v)
    with open(out, 'w') as f:
        for v in pts:
            f.write(' '.join(map(str, v)) + '\n')
    with open(out.replace('.txt', '_blocks.txt'), 'w') as f:
        for B in P:
            f.write(' '.join(map(str, B)) + '\n')
    print(len(pts), 'points written to', out)

if __name__ == '__main__':
    main()
