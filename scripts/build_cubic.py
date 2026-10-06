#!/usr/bin/env python3
"""Build the configuration of the cubic-graph design from /tmp/claude-0/cd_best.json (or given json)."""
import itertools, json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/cd_best.json'))
out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/claude-0/cubic_cfg.txt'
pts = []
for s in range(8):
    for g in (2, -2):
        v = [0] * 11; v[s] = g; pts.append(v)
for kind, sup, k in d['sol']:
    if kind == 'B':
        for sg in itertools.product((1, -1), repeat=4):
            v = [0] * 11
            for a, s in zip(sup, sg): v[a] = s
            pts.append(v)
    else:
        for sg in itertools.product((1, -1), repeat=4):
            v = [0] * 11
            for a, s in zip(list(sup) + [8 + k], sg): v[a] = s
            pts.append(v)
planes = {0: (8, 9), 1: (8, 10), 2: (9, 10)}
for e, c in d['col']:
    a, b = e
    p, q = planes[c]
    for sa, sb, sp, sq in itertools.product((1, -1), repeat=4):
        v = [0] * 11; v[a] = sa; v[b] = sb; v[p] = sp; v[q] = sq; pts.append(v)
for k in (8, 9, 10):
    for g in (2, -2):
        v = [0] * 11; v[k] = g; pts.append(v)
with open(out, 'w') as f:
    for v in pts: f.write(' '.join(map(str, v)) + '\n')
print(len(pts), 'points ->', out)
