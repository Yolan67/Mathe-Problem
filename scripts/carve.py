#!/usr/bin/env python3
"""carve.py IN OUT cos_threshold c0 c1 ... c10
Remove all points x with <x/|x|, c/|c|> > cos_threshold.  Kept points are written
(in original coordinates) to OUT; prints number kept/removed."""
import sys, numpy as np
X = np.loadtxt(sys.argv[1])
th = float(sys.argv[3])
c = np.array([float(t) for t in sys.argv[4:]]); c /= np.linalg.norm(c)
U = X / np.linalg.norm(X, axis=1, keepdims=True)
keep = U @ c <= th
np.savetxt(sys.argv[2], X[keep], fmt='%.17g')
print(int(keep.sum()), int((~keep).sum()))
