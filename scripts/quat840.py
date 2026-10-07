#!/usr/bin/env python3
"""Rebuild the quaternionic 840-point kissing arrangement of Takhanov (arXiv 2609.09179,
one-column.tex, Sections 2-4) EXACTLY in Q(sqrt2), following the paper's recipe literally:
  - Hamilton quaternions, <p,q> = Re(p conj q)                       (tex l.131-136)
  - Q8, C = 2T, omega, tau                                            (l.144-201)
  - K = Q8/{+-1} = F_2^2 with q_(0,0)=1,q_(1,0)=i,q_(0,1)=j,q_(1,1)=k (l.694-701, l.1509)
  - L(u) = {q_u, -q_u}, B_eta = {u+v+w=eta}, hat B_eta              (l.704-743)
  - (r1,r2,r3)[S] = {(r1 a, r2 b, r3 c)} LEFT multiplication          (l.747-754)
  - T_{1,r} = (tau w^r, w^r, w^r)[hat B_0]                            (l.756-777)
  - T_{2,r} = (w^r, tau w^r, tau w^r)[hat B_{delta_r}], delta=(j,i,k) (l.778-805)
  - E_1,E_2,E_3, M_1 = (a/sqrt2, b/2, c/2), M_2 = (a/2, b/sqrt2, c/2) (l.836-871)
Then verifies everything with exact integer arithmetic in Z[sqrt2] (coordinates scaled by 4).
Writes exact coordinates to X840_exact.txt: each line = 12 entries 'a,b' meaning (a + b*sqrt2)/4.
"""
import sys, itertools, collections
from fractions import Fraction as Fr
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "configs/quat840_exact.txt"

class S:
    """element a + b*sqrt2 of Q(sqrt2), a,b rational"""
    __slots__ = ("a", "b")
    def __init__(self, a, b=0):
        self.a = Fr(a); self.b = Fr(b)
    @staticmethod
    def c(o):
        return o if isinstance(o, S) else S(o)
    def __add__(s, o): o = S.c(o); return S(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __sub__(s, o): o = S.c(o); return S(s.a - o.a, s.b - o.b)
    def __neg__(s): return S(-s.a, -s.b)
    def __mul__(s, o): o = S.c(o); return S(s.a * o.a + 2 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def __eq__(s, o): o = S.c(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def __repr__(s): return f"({s.a}{'+' if s.b >= 0 else '-'}{abs(s.b)}r2)"

ZERO, ONE, HALF = S(0), S(1), S(Fr(1, 2))
INV_SQRT2 = S(0, Fr(1, 2))          # 1/sqrt2 = sqrt2/2

def qmul(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return (a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2)
def qneg(p): return tuple(-x for x in p)
def qconj(p): return (p[0], -p[1], -p[2], -p[3])
def qscale(s, p): return tuple(S.c(s) * x for x in p)
def qip(p, q):  # <p,q> = Re(p conj q)
    return qmul(p, qconj(q))[0]

one = (ONE, ZERO, ZERO, ZERO); I = (ZERO, ONE, ZERO, ZERO); J = (ZERO, ZERO, ONE, ZERO); K_ = (ZERO, ZERO, ZERO, ONE)
# sanity: Hamilton rules
assert qmul(I, J) == K_ and qmul(J, K_) == I and qmul(K_, I) == J and qmul(I, I) == qneg(one)
assert qmul(qmul(I, J), K_) == qneg(one)

Q8 = [one, qneg(one), I, qneg(I), J, qneg(J), K_, qneg(K_)]
half_units = [tuple(S(Fr(e, 2)) for e in eps) for eps in itertools.product((1, -1), repeat=4)]
C = Q8 + half_units
omega = (HALF, HALF, HALF, HALF)
tau = (INV_SQRT2, INV_SQRT2, ZERO, ZERO)
w = [one, omega, qmul(omega, omega)]
# checks from the paper (l.212-216, l.356, l.358-364)
assert w[2] == (S(Fr(-1, 2)), HALF, HALF, HALF)
assert qmul(w[2], omega) == qneg(one)
assert qmul(tau, tau) == I
tauinv = qconj(tau)
assert qmul(qmul(tau, I), tauinv) == I and qmul(qmul(tau, J), tauinv) == K_ and qmul(qmul(tau, K_), tauinv) == qneg(J)
assert qmul(qmul(omega, I), qconj(omega)) == J and qmul(qmul(omega, J), qconj(omega)) == K_ and qmul(qmul(omega, K_), qconj(omega)) == I

Cr = [[qmul(w[r], q) for q in Q8] for r in range(3)]      # C_r = w^r Q8
Dr = [[qmul(qmul(tau, w[r]), q) for q in Q8] for r in range(3)]  # D_r = tau w^r Q8
Cset = set(C)
assert set(Cr[0]) | set(Cr[1]) | set(Cr[2]) == Cset and sum(map(len, map(set, Cr))) == 24
# C_1: eps product = +1, C_2: eps product = -1 (l.296-327)
def epsprod(q):
    p = 1
    for x in q: p *= (1 if x.a > 0 else -1)
    return p
assert all(epsprod(q) == 1 for q in Cr[1]) and all(epsprod(q) == -1 for q in Cr[2])
# D = {(e_r u_r + e_s u_s)/sqrt2} (l.172-186)
Dpaper = set()
for r_, s_ in itertools.combinations(range(4), 2):
    for er, es in itertools.product((1, -1), repeat=2):
        v = [ZERO] * 4; v[r_] = S(0, Fr(er, 2)); v[s_] = S(0, Fr(es, 2)); Dpaper.add(tuple(v))
D = set(Dr[0]) | set(Dr[1]) | set(Dr[2])
assert D == Dpaper and len(D) == 24
assert D == {qmul(tau, c) for c in C} == {qmul(c, tau) for c in C}
# supports of D_0, D_1, D_2 (l.420-566)
def supp(q): return frozenset(i for i in range(4) if q[i] != ZERO)
assert {supp(q) for q in Dr[0]} == {frozenset({0, 1}), frozenset({2, 3})}
assert {supp(q) for q in Dr[1]} == {frozenset({0, 2}), frozenset({1, 3})}
assert {supp(q) for q in Dr[2]} == {frozenset({0, 3}), frozenset({1, 2})}
# group closure of 2O = C u D
G = Cset | D
assert all(qmul(x, y) in G for x in G for y in G) and len(G) == 48

# K = F_2^2 ; q_(0,0)=1, q_(1,0)=i, q_(0,1)=j, q_(1,1)=k  (l.1509)
Kel = [(0, 0), (1, 0), (0, 1), (1, 1)]
qrep = {(0, 0): one, (1, 0): I, (0, 1): J, (1, 1): K_}
def kadd(u, v): return (u[0] ^ v[0], u[1] ^ v[1])
def L(u): return [qrep[u], qneg(qrep[u])]
def Bhat(eta):
    out = []
    for u in Kel:
        for v in Kel:
            wv = kadd(kadd(u, v), eta)        # u+v+w = eta  <=>  w = eta+u+v
            for a in L(u):
                for b in L(v):
                    for c in L(wv):
                        out.append((a, b, c))
    assert len(out) == 128
    return out
delta = [(0, 1), (1, 0), (1, 1)]   # delta_0 = jbar, delta_1 = ibar, delta_2 = kbar (l.780, l.1550-1554)

def act(r1, r2, r3, Sset):  # left multiplication
    return [(qmul(r1, a), qmul(r2, b), qmul(r3, c)) for (a, b, c) in Sset]

T1 = []; T2 = []
for r in range(3):
    t1 = act(qmul(tau, w[r]), w[r], w[r], Bhat((0, 0)))
    t2 = act(w[r], qmul(tau, w[r]), qmul(tau, w[r]), Bhat(delta[r]))
    assert all(a in Dr[r] and b in Cr[r] and c in Cr[r] for a, b, c in t1)   # T_{1,r} in D_r x C_r x C_r
    assert all(a in Cr[r] and b in Dr[r] and c in Dr[r] for a, b, c in t2)   # T_{2,r} in C_r x D_r x D_r
    T1 += t1; T2 += t2
assert len(set(T1)) == 384 and len(set(T2)) == 384

Z4 = (ZERO,) * 4
pts = []; fam = []
for c in C: pts.append(c + Z4 + Z4); fam.append("E1")
for c in C: pts.append(Z4 + c + Z4); fam.append("E2")
for c in C: pts.append(Z4 + Z4 + c); fam.append("E3")
for a, b, c in T1: pts.append(qscale(INV_SQRT2, a) + qscale(HALF, b) + qscale(HALF, c)); fam.append("M1")
for a, b, c in T2: pts.append(qscale(HALF, a) + qscale(INV_SQRT2, b) + qscale(HALF, c)); fam.append("M2")
assert len(pts) == 840 and len(set(pts)) == 840

# integer encoding: 4*x_i = A + B*sqrt2 with A,B integers
A = np.zeros((840, 12), dtype=np.int64); B = np.zeros((840, 12), dtype=np.int64)
for n, p in enumerate(pts):
    for i, x in enumerate(p):
        a4, b4 = 4 * x.a, 4 * x.b
        assert a4.denominator == 1 and b4.denominator == 1
        A[n, i] = int(a4); B[n, i] = int(b4)
# Gram: 16<x,y> = P + Q sqrt2
P = A @ A.T + 2 * (B @ B.T)
Q = A @ B.T + B @ A.T
assert np.all(np.diag(P) == 16) and np.all(np.diag(Q) == 0)      # unit norms
def le(a, b, c):  # exact test a + b*sqrt2 <= c  (integers)
    d = a - c
    return np.where(b == 0, d <= 0,
           np.where(b > 0, (d <= 0) & (2 * b * b <= d * d),
                           (d <= 0) | (d * d <= 2 * b * b)))
off = ~np.eye(840, dtype=bool)
ok = le(P, Q, 8) | ~off
print("all distinct pairs <x,y> <= 1/2 exactly:", bool(ok.all()))
# antipodal
S_pts = set(pts)
print("antipodal:", all(tuple(-x for x in p) in S_pts for p in pts))
# inner-product profile
iu = np.triu_indices(840, 1)
prof = collections.Counter(zip(P[iu].tolist(), Q[iu].tolist()))
print("inner-product profile  (value = (P + Q sqrt2)/16):")
for (pp, qq), cnt in sorted(prof.items(), key=lambda kv: kv[0][0] / 16 + kv[0][1] * 2 ** 0.5 / 16):
    print(f"  ({pp:+d} {qq:+d}sqrt2)/16 = {pp/16 + qq*2**0.5/16:+.6f} : {cnt}")
print("total pairs:", sum(prof.values()))
# max inner product value
# frame operator (exact): 16 * S = sum (A+B r2)(A+B r2)^T
F_rat = A.T @ A + 2 * (B.T @ B); F_irr = A.T @ B + B.T @ A
print("frame operator *16 rational part diag:", np.diag(F_rat).tolist())
print("frame operator off-diagonal zero:", bool(np.all(F_rat - np.diag(np.diag(F_rat)) == 0)), bool(np.all(F_irr == 0)))
print("frame operator diag / 16:", (np.diag(F_rat) / 16).tolist())
# squared-norm distribution per H-factor, per family
for f in ("M1", "M2"):
    idx = [n for n in range(840) if fam[n] == f]
    s = set()
    for n in idx:
        p = pts[n]
        s.add(tuple(sum((x * x for x in p[4*h:4*h+4]), ZERO) for h in range(3)))
    print(f, "per-factor squared norms:", s)
# coordinate value sets per factor
for f in ("E1", "M1", "M2"):
    idx = [n for n in range(840) if fam[n] == f]
    for h in range(3):
        vals = sorted({(int(A[n, i]), int(B[n, i])) for n in idx for i in range(4*h, 4*h+4)})
        print(f, "factor", h + 1, "4*coord values (a,b) for a+b*sqrt2:", vals)
with open(OUT, "w") as fh:
    fh.write("# 840-point quaternionic kissing arrangement (Takhanov, arXiv 2609.09179). Each row: family, then 12 coords; 'a,b' means (a + b*sqrt(2))/4\n")
    for n in range(840):
        fh.write(fam[n] + " " + " ".join(f"{A[n,i]},{B[n,i]}" for i in range(12)) + "\n")
np.save(OUT.replace(".txt", "_A.npy"), A); np.save(OUT.replace(".txt", "_B.npy"), B)
print("wrote", OUT)
