#!/usr/bin/env python3
"""Max packing of 4-subsets of {0..10} (pairwise |B∩B'|<=2) with |B ∩ T| <= tmax, T={8,9,10}.
Writes graph for misg and decodes best solution."""
import itertools, numpy as np, subprocess, sys
tmax = int(sys.argv[1]) if len(sys.argv) > 1 else 1
T = {8, 9, 10}
blocks = [B for B in itertools.combinations(range(11), 4) if len(set(B) & T) <= tmax]
n = len(blocks)
with open('/tmp/claude-0/tpack.graph', 'wb') as f:
    np.array([n, 0], dtype=np.int32).tofile(f)
    for i, B in enumerate(blocks):
        nb = [j for j, C in enumerate(blocks) if j != i and len(set(B) & set(C)) > 2]
        np.array([len(nb)] + nb, dtype=np.int32).tofile(f)
subprocess.run(['./misg', '/tmp/claude-0/tpack.graph', '-o', '/tmp/claude-0/tpack.sol', '-T', '10', '-q', '1'])
sol = [blocks[int(l)] for l in open('/tmp/claude-0/tpack.sol')]
print(len(sol))
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    with open(out, 'w') as f:
        for B in sol: f.write(' '.join(map(str, B)) + '\n')
from collections import Counter
print(Counter(len(set(B) & T) for B in sol))
