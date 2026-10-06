// cyclic.c -- harmonic (cyclic group) codes in R^11 = 5 planes + 1 line.
// Point t (t = 0..N-1): x_t = (sqrt(p0) chi(t), sqrt(p_j) (cos 2pi k_j t/N, sin 2pi k_j t/N))_j
// chi = 1 (trivial) or (-1)^t (sign, N even).  <x_t,x_s> = p0 chi(d) + sum_j p_j cos(2 pi k_j d/N), d=t-s.
// For fixed frequencies the best weights solve the LP  min s : p>=0, sum p = 1, A(d) p <= s.
// We solve it by a simple dense simplex on the dual-friendly form; search frequencies by
// random restarts + local improvement.  Reports best max-cos for given N.
// usage: cyclic N [seconds] [seed] [sign 0/1] [nplanes]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static unsigned long long rs = 88172645463325252ULL;
static unsigned long long rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double ru(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

#define MAXV 8
// LP: variables p_0..p_{m-1} >= 0, sum = 1, minimize max_d (A_d . p).
// Solve via iterative approach: minimax over simplex with few variables: use the fact that the
// optimum is attained at a vertex of the arrangement; use a simple primal simplex on
//   min s  s.t.  A p - s 1 <= 0 ,  1.p = 1 , p >= 0, s free.
// Implemented as a dense tableau with Bland's rule (sizes: ~N/2 rows, <= 8 columns).
static int ND;  // number of constraints (d values)
static double *A;  // ND x m
static double lp_solve(int m, double *pout) {
  // Use dual: max t s.t. sum_d y_d A_d(i) + ... ; simpler: solve with "minimax by subgradient + refine"
  // Here: exact via enumerating active sets is expensive; use a robust iterative LP: multiplicative
  // weights on constraints (Freund–Schapire) then polish with few vertex checks.
  // Game: row player chooses p in simplex minimizing max_d A_d.p ; equivalent zero-sum game value.
  int T = 4000;
  double *w = malloc(ND * sizeof(double));
  double *pa = calloc(m, sizeof(double));
  for (int d = 0; d < ND; d++) w[d] = 1.0;
  double eta = 0.05;
  double best = 1e9; double pb[MAXV];
  double ub_val;
  for (int it = 0; it < T; it++) {
    // column player (constraints) mixed strategy w ; row player best response: choose i minimizing sum_d w_d A_d(i)
    double W = 0; for (int d = 0; d < ND; d++) W += w[d];
    int bi = 0; double bv = 1e300;
    for (int i = 0; i < m; i++) { double s = 0; for (int d = 0; d < ND; d++) s += w[d] * A[d * m + i]; if (s < bv) { bv = s; bi = i; } }
    pa[bi] += 1.0;
    // evaluate averaged p
    double p[MAXV]; double tot = it + 1;
    for (int i = 0; i < m; i++) p[i] = pa[i] / tot;
    double mx = -1e300;
    for (int d = 0; d < ND; d++) { double s = 0; for (int i = 0; i < m; i++) s += A[d * m + i] * p[i]; if (s > mx) mx = s; }
    if (mx < best) { best = mx; memcpy(pb, p, sizeof(double) * m); }
    // update weights: constraints with high value under pure strategy bi get more weight
    for (int d = 0; d < ND; d++) w[d] *= exp(eta * A[d * m + bi]);
    double mw = 0; for (int d = 0; d < ND; d++) if (w[d] > mw) mw = w[d];
    for (int d = 0; d < ND; d++) w[d] /= mw;
  }
  ub_val = best;
  // local polish: coordinate pair moves
  double p[MAXV]; memcpy(p, pb, sizeof(double) * m);
  double step = 0.02;
  for (int r = 0; r < 3000; r++) {
    int i = rnd() % m, j = rnd() % m; if (i == j) continue;
    double delta = step * (ru() * 2 - 1);
    if (p[i] + delta < 0 || p[j] - delta < 0) continue;
    p[i] += delta; p[j] -= delta;
    double mx = -1e300;
    for (int d = 0; d < ND; d++) { double s = 0; for (int k = 0; k < m; k++) s += A[d * m + k] * p[k]; if (s > mx) mx = s; }
    if (mx < ub_val) { ub_val = mx; memcpy(pb, p, sizeof(double) * m); } else { p[i] -= delta; p[j] += delta; }
    if (r % 500 == 499) step *= 0.5;
  }
  memcpy(pout, pb, sizeof(double) * m);
  free(w); free(pa);
  return ub_val;
}
int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  double T = argc > 2 ? atof(argv[2]) : 60;
  rs ^= (argc > 3 ? strtoull(argv[3], 0, 10) : 1) * 0x9E3779B97F4A7C15ULL;
  int sign = argc > 4 ? atoi(argv[4]) : 0;
  int np = argc > 5 ? atoi(argv[5]) : 5;
  int m = np + 1;
  ND = N / 2;
  A = malloc((size_t)ND * m * sizeof(double));
  int k[MAXV], kb[MAXV]; double best = 9, pb[MAXV];
  double t0 = now();
  long evals = 0;
  while (now() - t0 < T) {
    for (int j = 0; j < np; j++) k[j] = 1 + rnd() % (N / 2);
    double cur = 9; double p[MAXV];
    for (int iter = 0; iter < 200; iter++) {
      int kk[MAXV]; memcpy(kk, k, sizeof k);
      if (iter > 0) { int j = rnd() % np; kk[j] = 1 + rnd() % (N / 2); }
      for (int d = 1; d <= ND; d++) {
        A[(d - 1) * m + 0] = sign ? ((d & 1) ? -1.0 : 1.0) : 1.0;
        for (int j = 0; j < np; j++) A[(d - 1) * m + 1 + j] = cos(2 * M_PI * (double)kk[j] * d / N);
      }
      double v = lp_solve(m, p); evals++;
      if (v < cur) { cur = v; memcpy(k, kk, sizeof k); }
      if (cur < best) { best = cur; memcpy(kb, k, sizeof k); memcpy(pb, p, sizeof(double) * m);
        printf("N=%d best maxcos %.6f freqs", N, best); for (int j = 0; j < np; j++) printf(" %d", kb[j]); printf(" weights"); for (int j = 0; j < m; j++) printf(" %.4f", pb[j]); printf("  (%.0fs, %ld LPs)\n", now() - t0, evals); fflush(stdout); }
    }
  }
  printf("final N=%d best %.6f\n", N, best);
  return 0;
}
