#!/usr/bin/env python3
"""exactify.py FLOAT_IN NFIX OUT [--den 1000000000]
First NFIX points: snapped to exact integers (norm-4 scale); asserts closeness.
Remaining points: scaled to norm 2 and rounded to rationals with denominator DEN.
Output is a rational configuration for validate/verify.py (exact)."""
import sys
import numpy as np
from fractions import Fraction as Fr
args = sys.argv[1:]
den = 10**9
if '--den' in args:
    i = args.index('--den'); den = int(args[i+1]); del args[i:i+2]
X = np.loadtxt(args[0]); nfix = int(args[1]); out = args[2]
X = 2 * X / np.linalg.norm(X, axis=1, keepdims=True)
with open(out, 'w') as f:
    for i, v in enumerate(X):
        if i < nfix:
            r = np.rint(v)
            assert np.abs(r - v).max() < 1e-9, (i, v)
            f.write(' '.join(str(int(t)) for t in r) + '\n')
        else:
            f.write(' '.join(str(Fr(int(round(t * den)), den)) for t in v) + '\n')
print('wrote', len(X), 'points to', out)
