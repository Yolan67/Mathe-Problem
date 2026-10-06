#!/usr/bin/env python3
"""Given a feasible piece configuration (points X, ip<=3), scan extra partial pieces:
base in {R,P0}, phase phi, angle t; compatible vectors = those with ip<=3+tol to all of X;
then MIS among compatible (internal conflicts).  Reports best counts."""
import sys, subprocess, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location('gp', 'scripts/g33pieces.py'); gp = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ['x']; spec.loader.exec_module(gp); sys.argv = _a
X = np.load(sys.argv[1])
nphi = int(sys.argv[2]) if len(sys.argv) > 2 else 72
nt = int(sys.argv[3]) if len(sys.argv) > 3 else 61
v = np.load('/tmp/claude-0/k12_v.npy'); vn = v / np.linalg.norm(v)
M = np.eye(6, dtype=complex) - np.outer(vn, vn.conj())
U, s, Wh = np.linalg.svd(M); Bc = U[:, :5]
best = []
for base in ['P0', 'R']:
    Z = gp.B[base] @ Bc.conj()          # (n,5) complex
    for phi in np.linspace(0, 2 * np.pi, nphi, endpoint=False):
        Zp = np.exp(1j * phi) * Z
        Y0 = np.concatenate([Zp.real, Zp.imag], axis=1)
        for t in np.linspace(-1.5, 1.5, nt):
            lam = np.cos(t) * (6 / gp.sq[base]) ** .5; hh = 6 ** .5 * np.sin(t)
            Y = np.concatenate([lam * Y0, np.full((len(Y0), 1), hh)], axis=1)
            mx = (Y @ X.T).max(1)
            ok = np.nonzero(mx <= 3 + 1e-9)[0]
            if len(ok) == 0: continue
            Yo = Y[ok]; Gi = Yo @ Yo.T; np.fill_diagonal(Gi, -9)
            # greedy + small exact-ish: count MIS via misg if needed
            conf = Gi > 3 + 1e-9
            if not conf.any():
                k = len(ok)
            else:
                n = len(ok)
                with open('/tmp/claude-0/pp.graph', 'wb') as f:
                    np.array([n, 0], dtype=np.int32).tofile(f)
                    for i in range(n):
                        r = np.nonzero(conf[i])[0].astype(np.int32)
                        np.array([len(r)], dtype=np.int32).tofile(f); r.tofile(f)
                out = subprocess.run(['./misg', '/tmp/claude-0/pp.graph', '-T', '0.2', '-q', '1', '-o', '/tmp/claude-0/pp.sol'], capture_output=True, text=True).stdout
                k = int(out.split()[-1])
            best.append((k, base, round(phi, 4), round(t, 4), len(ok)))
best.sort(reverse=True)
print('top extra partial pieces (count, base, phi, t, compatible):')
for b in best[:15]: print(b)
