#!/usr/bin/env python3
"""Infinitesimal flexibility of a kissing configuration.
Equality flexes: null space of the rigidity matrix (contacts kept exactly, tangent velocities).
Strict flexes: LP max total opening of contacts.  usage: flex.py config.txt [tol]"""
import sys, time
import numpy as np
X = np.loadtxt(sys.argv[1]); X /= np.linalg.norm(X, axis=1)[:, None]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-7
N, d = X.shape
G = X @ X.T; np.fill_diagonal(G, -9)
I, J = np.nonzero(np.triu(G > .5 - tol, 1))
print('N', N, 'contacts', len(I), flush=True)
# tangent bases
P = np.zeros((N, d, d - 1))
for i in range(N):
    q, _ = np.linalg.qr(np.column_stack([X[i], np.random.default_rng(i).normal(size=(d, d - 1))]))
    P[i] = q[:, 1:]
nv = N * (d - 1)
U = np.einsum('ck,ckm->cm', X[J], P[I]); W = np.einsum('ck,ckm->cm', X[I], P[J])
M = np.zeros((N, d - 1, N, d - 1))
t = time.time()
np.add.at(M, (I, slice(None), I, slice(None)), np.einsum('ca,cb->cab', U, U))
np.add.at(M, (J, slice(None), J, slice(None)), np.einsum('ca,cb->cab', W, W))
np.add.at(M, (I, slice(None), J, slice(None)), np.einsum('ca,cb->cab', U, W))
np.add.at(M, (J, slice(None), I, slice(None)), np.einsum('ca,cb->cab', W, U))
M = M.reshape(nv, nv)
w, V = np.linalg.eigh(M)
np.save('/tmp/claude-0/flex_eig.npy', w); np.save('/tmp/claude-0/flex_vec.npy', V[:, :200])
print('eig time %.1f' % (time.time() - t), 'smallest eigenvalues', np.round(w[:80], 8).tolist(), flush=True)
print('null dim (eig<1e-9):', (w < 1e-9).sum(), ' rotations:', d * (d - 1) // 2)
