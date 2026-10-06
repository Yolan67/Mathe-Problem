// kiss.c -- search engine for kissing configurations on S^{d-1}
//
// Energy: E(X) = sum_{i<j} max(0, <x_i,x_j> - t)^2 with unit vectors x_i,
// minimized with L-BFGS on unnormalized coordinates (x = y/|y|) using a
// Verlet neighbour list.  Search for N+1 points by basin hopping starting from
// a valid N-point configuration plus a point in its deepest hole.
//
// Modes:
//   kiss relax -i IN -o OUT [-m margin]
//   kiss holes -i IN [-k count]
//   kiss grow  -i IN -o PREFIX [-a add] [-s seed] [-T seconds] [-m margin]
//
// The output of this program is NOT a proof of validity; every candidate is
// re-checked with validate/verify.py in exact arithmetic.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define D 11
#define S 12  // stride

static int g_dim = D;
static int g_nfix = 0;  // first g_nfix points are fixed

// ---------------------------------------------------------------- rng
static uint64_t rs[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(void) {
  uint64_t s0 = rs[0], s1 = rs[1], r = s0 + s1;
  s1 ^= s0;
  rs[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  rs[1] = rotl(s1, 36);
  return r;
}
static double runif(void) { return (rnext() >> 11) * (1.0 / 9007199254740992.0); }
static double rgauss(void) {
  double u = runif(), v = runif();
  if (u < 1e-300) u = 1e-300;
  return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}
static void rseed(uint64_t s) {
  rs[0] = s * 0x9E3779B97F4A7C15ULL + 1;
  rs[1] = (s ^ 0xDEADBEEFCAFEULL) * 0xBF58476D1CE4E5B9ULL + 7;
  for (int i = 0; i < 20; i++) rnext();
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// ---------------------------------------------------------------- vectors
static inline double dot(const double *a, const double *b) {
  double s = 0;
  for (int k = 0; k < S; k++) s += a[k] * b[k];
  return s;
}
static inline void normalize(double *a) {
  double n = sqrt(dot(a, a));
  for (int k = 0; k < S; k++) a[k] /= n;
}
static void rand_unit(double *a) {
  for (int k = 0; k < S; k++) a[k] = k < g_dim ? rgauss() : 0;
  normalize(a);
}

// ---------------------------------------------------------------- io
static double *load(const char *fn, int *N) {
  FILE *f = fopen(fn, "r");
  if (!f) { perror(fn); exit(1); }
  int cap = 1024, n = 0;
  double *X = calloc((size_t)cap * S, sizeof(double));
  char line[1 << 16];
  while (fgets(line, sizeof line, f)) {
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == 0) continue;
    if (n == cap) { cap *= 2; X = realloc(X, (size_t)cap * S * sizeof(double)); }
    double *x = X + (size_t)n * S;
    memset(x, 0, S * sizeof(double));
    for (int k = 0; k < g_dim; k++) x[k] = strtod(p, &p);
    normalize(x);
    n++;
  }
  fclose(f);
  *N = n;
  return X;
}
static void save(const char *fn, const double *X, int N) {
  FILE *f = fopen(fn, "w");
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < g_dim; k++) fprintf(f, "%.17g%c", X[i * S + k], k == g_dim - 1 ? '\n' : ' ');
  }
  fclose(f);
}

// ---------------------------------------------------------------- stats
static double maxcos(const double *X, int N, int *pi, int *pj) {
  double m = -2;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) {
      double c = dot(X + i * S, X + j * S);
      if (c > m) { m = c; if (pi) { *pi = i; *pj = j; } }
    }
  return m;
}

// ---------------------------------------------------------------- energy
typedef struct {
  int N;
  double t;      // threshold
  double skin;
  int *pi, *pj;  // neighbour list
  int np, cap;
  double *Xref;  // positions at last rebuild (normalized)
  int nbuild;
} ctx_t;

static void nl_build(ctx_t *c, const double *X) {
  int N = c->N;
  c->np = 0;
  double th = c->t - c->skin;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      if (dot(X + i * S, X + j * S) > th) {
        if (c->np == c->cap) {
          c->cap = c->cap ? 2 * c->cap : 1 << 16;
          c->pi = realloc(c->pi, c->cap * sizeof(int));
          c->pj = realloc(c->pj, c->cap * sizeof(int));
        }
        c->pi[c->np] = i;
        c->pj[c->np] = j;
        c->np++;
      }
  memcpy(c->Xref, X, (size_t)N * S * sizeof(double));
  c->nbuild++;
}
static int nl_check(ctx_t *c, const double *X) {
  // returns 1 if rebuild needed
  double m1 = 0, m2 = 0;
  for (int i = 0; i < c->N; i++) {
    double d2 = 0;
    for (int k = 0; k < S; k++) { double d = X[i * S + k] - c->Xref[i * S + k]; d2 += d * d; }
    if (d2 > m1) { m2 = m1; m1 = d2; } else if (d2 > m2) m2 = d2;
  }
  return sqrt(m1) + sqrt(m2) > c->skin;
}

// E and gradient wrt unnormalized Y (rows), X = normalized Y written to Xn
static double eval(ctx_t *c, const double *Y, double *Xn, double *G) {
  int N = c->N;
  static double *nrm = NULL; static int nc = 0;
  if (nc < N) { nrm = realloc(nrm, N * sizeof(double)); nc = N; }
  for (int i = 0; i < N; i++) {
    double n = sqrt(dot(Y + i * S, Y + i * S));
    nrm[i] = n;
    for (int k = 0; k < S; k++) Xn[i * S + k] = Y[i * S + k] / n;
  }
  if (nl_check(c, Xn)) nl_build(c, Xn);
  memset(G, 0, (size_t)N * S * sizeof(double));
  double E = 0, t = c->t;
  for (int p = 0; p < c->np; p++) {
    int i = c->pi[p], j = c->pj[p];
    const double *xi = Xn + i * S, *xj = Xn + j * S;
    double cc = dot(xi, xj);
    if (cc > t) {
      double d = cc - t;
      E += d * d;
      double w = 2 * d;
      double *gi = G + i * S, *gj = G + j * S;
      for (int k = 0; k < S; k++) { gi[k] += w * xj[k]; gj[k] += w * xi[k]; }
    }
  }
  memset(G, 0, (size_t)g_nfix * S * sizeof(double));
  // project to tangent space and scale by 1/|y|
  for (int i = g_nfix; i < N; i++) {
    double *g = G + i * S;
    const double *x = Xn + i * S;
    double gx = dot(g, x);
    for (int k = 0; k < S; k++) g[k] = (g[k] - gx * x[k]) / nrm[i];
  }
  return E;
}

// ---------------------------------------------------------------- L-BFGS
#define LM 8
typedef struct { double E; int it; double gnorm; } relax_res;

static relax_res relax(ctx_t *c, double *Y, int maxit, double Etol) {
  int N = c->N, n = N * S;
  double *Xn = malloc(n * sizeof(double)), *G = malloc(n * sizeof(double));
  double *Y2 = malloc(n * sizeof(double)), *G2 = malloc(n * sizeof(double));
  double *dir = malloc(n * sizeof(double));
  double *sv = malloc((size_t)LM * n * sizeof(double)), *yv = malloc((size_t)LM * n * sizeof(double));
  double rho[LM], al[LM];
  int head = 0, mem = 0;
  // renormalize
  for (int i = 0; i < N; i++) normalize(Y + i * S);
  nl_build(c, Y);
  double E = eval(c, Y, Xn, G);
  int it;
  double Eprev_check = E;
  for (it = 0; it < maxit && E > Etol; it++) {
    // two-loop
    for (int k = 0; k < n; k++) dir[k] = -G[k];
    int idx[LM];
    for (int m = 0; m < mem; m++) idx[m] = (head - 1 - m + LM) % LM;
    for (int m = 0; m < mem; m++) {
      int q = idx[m];
      double a = 0;
      for (int k = 0; k < n; k++) a += sv[(size_t)q * n + k] * dir[k];
      a *= rho[q];
      al[q] = a;
      for (int k = 0; k < n; k++) dir[k] -= a * yv[(size_t)q * n + k];
    }
    if (mem > 0) {
      int q = idx[0];
      double sy = 0, yy = 0;
      for (int k = 0; k < n; k++) { sy += sv[(size_t)q * n + k] * yv[(size_t)q * n + k]; yy += yv[(size_t)q * n + k] * yv[(size_t)q * n + k]; }
      double gam = sy / yy;
      for (int k = 0; k < n; k++) dir[k] *= gam;
    } else {
      double gn = 0;
      for (int k = 0; k < n; k++) gn += G[k] * G[k];
      gn = sqrt(gn);
      double sc = 1e-2 / (gn + 1e-300);
      for (int k = 0; k < n; k++) dir[k] *= sc;
    }
    for (int m = mem - 1; m >= 0; m--) {
      int q = idx[m];
      double b = 0;
      for (int k = 0; k < n; k++) b += yv[(size_t)q * n + k] * dir[k];
      b *= rho[q];
      for (int k = 0; k < n; k++) dir[k] += sv[(size_t)q * n + k] * (al[q] - b);
    }
    double gd = 0;
    for (int k = 0; k < n; k++) gd += G[k] * dir[k];
    if (gd >= 0) {  // not descent: reset
      mem = 0;
      double gn = 0;
      for (int k = 0; k < n; k++) gn += G[k] * G[k];
      gn = sqrt(gn);
      for (int k = 0; k < n; k++) dir[k] = -G[k] * 1e-2 / (gn + 1e-300);
      gd = -gn * 1e-2;
    }
    // limit max displacement per point
    double md = 0;
    for (int i = 0; i < N; i++) {
      double d2 = 0;
      for (int k = 0; k < S; k++) d2 += dir[i * S + k] * dir[i * S + k];
      if (d2 > md) md = d2;
    }
    md = sqrt(md);
    double step = 1.0;
    if (md > 0.05) step = 0.05 / md;
    double E2;
    int ok = 0;
    for (int ls = 0; ls < 30; ls++) {
      for (int k = 0; k < n; k++) Y2[k] = Y[k] + step * dir[k];
      E2 = eval(c, Y2, Xn, G2);
      if (E2 <= E + 1e-4 * step * gd) { ok = 1; break; }
      step *= 0.5;
    }
    if (!ok) {
      if (mem == 0) break;
      mem = 0;
      continue;
    }
    // update memory
    double sy = 0;
    double *s = sv + (size_t)head * n, *y = yv + (size_t)head * n;
    for (int k = 0; k < n; k++) { s[k] = Y2[k] - Y[k]; y[k] = G2[k] - G[k]; sy += s[k] * y[k]; }
    if (sy > 1e-30) {
      rho[head] = 1.0 / sy;
      head = (head + 1) % LM;
      if (mem < LM) mem++;
    }
    memcpy(Y, Y2, n * sizeof(double));
    memcpy(G, G2, n * sizeof(double));
    E = E2;
    // renormalize rows occasionally
    if ((it & 255) == 255) {
      int bad = 0;
      for (int i = 0; i < N; i++) {
        double q = dot(Y + i * S, Y + i * S);
        if (q > 4 || q < 0.25) bad = 1;
      }
      if (bad) { for (int i = 0; i < N; i++) normalize(Y + i * S); mem = 0; E = eval(c, Y, Xn, G); }
    }
    // stagnation check
    if ((it % 500) == 499) {
      if (E > 0.999 * Eprev_check && E > 1e-10) { it++; break; }
      Eprev_check = E;
    }
  }
  for (int i = 0; i < N; i++) normalize(Y + i * S);
  relax_res r;
  r.E = E;
  r.it = it;
  r.gnorm = 0;
  free(Xn); free(G); free(Y2); free(G2); free(dir); free(sv); free(yv);
  return r;
}

// ---------------------------------------------------------------- holes
// find point y minimizing max_i <y, x_i>, x_i i in [0,N) excluding 'skip' (-1 none)
static double hole_opt(const double *X, int N, int skip, double *y, int iters) {
  double beta = 60;
  double g[S];
  double *w = malloc(N * sizeof(double));
  double lr = 0.02;
  double best = 2, yb[S];
  for (int it = 0; it < iters; it++) {
    double m = -2;
    for (int i = 0; i < N; i++) {
      if (i == skip) { w[i] = -1e9; continue; }
      w[i] = dot(y, X + i * S);
      if (w[i] > m) m = w[i];
    }
    if (m < best) { best = m; memcpy(yb, y, sizeof yb); }
    double Z = 0;
    for (int i = 0; i < N; i++) { w[i] = i == skip ? 0 : exp(beta * (w[i] - m)); Z += w[i]; }
    memset(g, 0, sizeof g);
    for (int i = 0; i < N; i++) if (w[i] > 1e-12) { double a = w[i] / Z; for (int k = 0; k < S; k++) g[k] += a * X[i * S + k]; }
    double gy = dot(g, y);
    for (int k = 0; k < S; k++) g[k] -= gy * y[k];
    for (int k = 0; k < S; k++) y[k] -= lr * g[k];
    normalize(y);
    if (it % 50 == 49) { beta *= 1.5; if (beta > 3000) beta = 3000; lr *= 0.7; }
  }
  memcpy(y, yb, sizeof yb);
  free(w);
  return best;
}

static double deepest_hole(const double *X, int N, int skip, double *yout, int samples, int keep) {
  double *cand = malloc((size_t)samples * S * sizeof(double));
  double *val = malloc(samples * sizeof(double));
  for (int s = 0; s < samples; s++) {
    rand_unit(cand + s * S);
    double m = -2;
    for (int i = 0; i < N; i++) {
      if (i == skip) continue;
      double c = dot(cand + s * S, X + i * S);
      if (c > m) m = c;
    }
    val[s] = m;
  }
  // pick 'keep' best by partial selection
  double best = 3;
  for (int r = 0; r < keep; r++) {
    int bi = 0;
    for (int s = 1; s < samples; s++) if (val[s] < val[bi]) bi = s;
    double y[S];
    memcpy(y, cand + bi * S, sizeof y);
    val[bi] = 10;
    double v = hole_opt(X, N, skip, y, 400);
    if (v < best) { best = v; memcpy(yout, y, sizeof y); }
  }
  free(cand); free(val);
  return best;
}

// ---------------------------------------------------------------- main
static const char *argval(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

static void ctx_init(ctx_t *c, int N, double t) {
  memset(c, 0, sizeof *c);
  c->N = N;
  c->t = t;
  c->skin = 0.12;
  c->Xref = malloc((size_t)N * S * sizeof(double));
}
static void ctx_free(ctx_t *c) { free(c->pi); free(c->pj); free(c->Xref); }

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: kiss relax|holes|grow ...\n"); return 1; }
  const char *mode = argv[1];
  const char *in = argval(argc, argv, "-i", NULL);
  const char *out = argval(argc, argv, "-o", "out");
  double margin = atof(argval(argc, argv, "-m", "0"));
  uint64_t seed = strtoull(argval(argc, argv, "-s", "1"), NULL, 10);
  g_dim = atoi(argval(argc, argv, "-d", "11"));
  g_nfix = atoi(argval(argc, argv, "-f", "0"));
  rseed(seed);
  int N;
  double *X = load(in, &N);
  int a, b;
  double mc = maxcos(X, N, &a, &b);
  printf("loaded %d points, maxcos %.15f (%d,%d)\n", N, mc, a, b);
  fflush(stdout);

  if (!strcmp(mode, "relax")) {
    ctx_t c;
    ctx_init(&c, N, 0.5 - margin);
    double t0 = now();
    relax_res r = relax(&c, X, 200000, 1e-30);
    mc = maxcos(X, N, NULL, NULL);
    printf("relax: E=%.3e it=%d maxcos=%.15f time %.2fs pairs %d builds %d\n", r.E, r.it, mc, now() - t0, c.np, c.nbuild);
    save(out, X, N);
    return 0;
  }
  if (!strcmp(mode, "holes")) {
    int k = atoi(argval(argc, argv, "-k", "10"));
    FILE *fo = fopen(out, "w");
    for (int r = 0; r < k; r++) {
      double y[S];
      double v = deepest_hole(X, N, -1, y, 20000, 10);
      printf("hole %d: maxcos %.10f\n", r, v);
      for (int q = 0; q < g_dim; q++) fprintf(fo, "%.17g%c", y[q], q == g_dim - 1 ? '\n' : ' ');
      fflush(stdout);
    }
    fclose(fo);
    return 0;
  }
  if (!strcmp(mode, "grow")) {
    int add = atoi(argval(argc, argv, "-a", "1"));
    double T = atof(argval(argc, argv, "-T", "3600"));
    double temp0 = atof(argval(argc, argv, "-temp", "0"));
    int keepgoing = atoi(argval(argc, argv, "-c", "1"));
    double t0 = now();
    int cap = N + 1000;
    X = realloc(X, (size_t)cap * S * sizeof(double));
    memset(X + (size_t)N * S, 0, (size_t)(cap - N) * S * sizeof(double));
    for (int r = 0; r < add; r++) {
      double y[S];
      double v = deepest_hole(X, N, -1, y, 20000, 10);
      memcpy(X + (size_t)N * S, y, sizeof y);
      N++;
      printf("added point in hole with maxcos %.6f -> N=%d\n", v, N);
    }
    double *Xc = malloc((size_t)cap * S * sizeof(double));
    double *Xb = malloc((size_t)cap * S * sizeof(double));
    ctx_t c;
    ctx_init(&c, N, 0.5 - margin);
    relax_res r = relax(&c, X, 20000, 1e-26);
    double E = r.E, Eb = E;
    memcpy(Xb, X, (size_t)N * S * sizeof(double));
    printf("initial relax E=%.4e\n", E);
    long iter = 0, acc = 0;
    double sig_lo = 1e-3, sig_hi = 5e-2;
    double temp = temp0;
    double *le = malloc(cap * sizeof(double));
    while (now() - t0 < T) {
      if (E < 1e-26) {
        mc = maxcos(X, N, NULL, NULL);
        char fn[512];
        snprintf(fn, sizeof fn, "%s_N%d_s%llu.txt", out, N, (unsigned long long)seed);
        save(fn, X, N);
        printf("SUCCESS N=%d maxcos=%.15f iter=%ld time=%.1f -> %s\n", N, mc, iter, now() - t0, fn);
        fflush(stdout);
        if (!keepgoing) break;
        double y[S];
        double v = deepest_hole(X, N, -1, y, 20000, 10);
        memcpy(X + (size_t)N * S, y, sizeof y);
        N++;
        ctx_free(&c);
        ctx_init(&c, N, 0.5 - margin);
        r = relax(&c, X, 20000, 1e-26);
        E = r.E;
        Eb = E;
        memcpy(Xb, X, (size_t)N * S * sizeof(double));
        printf("grow to N=%d, hole %.6f, E=%.4e\n", N, v, E);
        continue;
      }
      memcpy(Xc, X, (size_t)N * S * sizeof(double));
      double u = runif();
      int mv;
      if (u < 0.35) {
        // relocate worst-energy point(s) to deepest hole among others
        mv = 0;
        memset(le, 0, N * sizeof(double));
        for (int i = 0; i < N; i++)
          for (int j = i + 1; j < N; j++) {
            double cc = dot(X + i * S, X + j * S);
            if (cc > c.t) { double d = (cc - c.t) * (cc - c.t); le[i] += d; le[j] += d; }
          }
        // roulette selection
        double tot = 0;
        for (int i = 0; i < N; i++) tot += le[i];
        for (int i = 0; i < g_nfix; i++) { tot -= le[i]; le[i] = 0; }
        double rr = runif() * tot;
        int sel = N - 1;
        for (int i = g_nfix; i < N; i++) { rr -= le[i]; if (rr <= 0) { sel = i; break; } }
        double y[S];
        deepest_hole(X, N, sel, y, 4000, 4);
        memcpy(X + sel * S, y, sizeof y);
      } else if (u < 0.55) {
        // relocate a random point to a random hole
        mv = 1;
        int sel = g_nfix + rnext() % (N - g_nfix);
        double y[S];
        deepest_hole(X, N, sel, y, 2000, 3);
        memcpy(X + sel * S, y, sizeof y);
      } else if (u < 0.85) {
        mv = 2;
        double sg = sig_lo * pow(sig_hi / sig_lo, runif());
        for (int i = g_nfix; i < N; i++) {
          for (int k = 0; k < g_dim; k++) X[i * S + k] += sg * rgauss() / sqrt(g_dim);
          normalize(X + i * S);
        }
      } else {
        // local shake around a random point with overlap
        mv = 3;
        int cen = g_nfix + rnext() % (N - g_nfix);
        double sg = 0.05 + 0.1 * runif();
        for (int i = g_nfix; i < N; i++) {
          if (dot(X + i * S, X + cen * S) > 0.3) {
            for (int k = 0; k < g_dim; k++) X[i * S + k] += sg * rgauss() / sqrt(g_dim);
            normalize(X + i * S);
          }
        }
      }
      r = relax(&c, X, 20000, 1e-26);
      iter++;
      int accept = r.E < E;
      if (!accept && temp > 0) accept = runif() < exp(-(r.E - E) / (temp * E));
      if (accept) {
        acc++;
        E = r.E;
        if (E < Eb) { Eb = E; memcpy(Xb, X, (size_t)N * S * sizeof(double)); }
      } else {
        memcpy(X, Xc, (size_t)N * S * sizeof(double));
      }
      if (iter % 20 == 0) {
        printf("it %ld N=%d E=%.4e best=%.4e acc=%ld mv=%d t=%.0f\n", iter, N, E, Eb, acc, mv, now() - t0);
        fflush(stdout);
      }
    }
    char fn[512];
    snprintf(fn, sizeof fn, "%s_last_N%d_s%llu.txt", out, N, (unsigned long long)seed);
    save(fn, Xb, N);
    printf("done N=%d bestE=%.4e\n", N, Eb);
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 1;
}
