// misg.c -- maximum independent set by iterated local search (ARW 2012)
// on an explicit graph.  Graph file (binary, little endian int32):
//   n, m, then for each vertex: deg, neighbours...
// usage: misg GRAPH [-i init_indices.txt] [-o out.txt] [-s seed] [-T sec]
//                   [-target K] [-pm prob_multi]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int n;
static int **adj, *deg;
static uint8_t *inS;
static int *tight, *Sl, *Spos, Sn;
static int64_t *sumS;
static int *Fl, *Fpos, Fn;
static int *Q, Qn; static uint8_t *inQ;
static long *tabu; static long itg = 0;
static uint8_t *adjbits = NULL;  // optional dense bit matrix for small n

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rs[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(void) { uint64_t s0 = rs[0], s1 = rs[1], r = s0 + s1; s1 ^= s0; rs[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); rs[1] = rotl(s1, 36); return r; }
static double runif(void) { return (rnext() >> 11) * (1.0 / 9007199254740992.0); }

static inline int adjacent(int a, int b) {
  if (adjbits) { size_t k = (size_t)a * n + b; return (adjbits[k >> 3] >> (k & 7)) & 1; }
  // binary search in sorted list of smaller degree
  int *L = adj[a], d = deg[a];
  if (deg[b] < d) { L = adj[b]; d = deg[b]; b = a; }
  int lo = 0, hi = d - 1;
  while (lo <= hi) { int mid = (lo + hi) >> 1; if (L[mid] == b) return 1; if (L[mid] < b) lo = mid + 1; else hi = mid - 1; }
  return 0;
}
static inline void free_add(int v) { if (Fpos[v] < 0) { Fpos[v] = Fn; Fl[Fn++] = v; } }
static inline void free_del(int v) { int p = Fpos[v]; if (p >= 0) { int w = Fl[--Fn]; Fl[p] = w; Fpos[w] = p; Fpos[v] = -1; } }
static inline void q_push(int x) { if (!inQ[x]) { inQ[x] = 1; Q[Qn++] = x; } }
static void S_add(int v) {
  inS[v] = 1; Spos[v] = Sn; Sl[Sn++] = v; free_del(v);
  for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; sumS[u] += v; if (tight[u]++ == 0) free_del(u); }
  q_push(v);
}
static void S_del(int v) {
  inS[v] = 0; int p = Spos[v]; int w = Sl[--Sn]; Sl[p] = w; Spos[w] = p; Spos[v] = -1;
  for (int i = 0; i < deg[v]; i++) {
    int u = adj[v][i]; sumS[u] -= v; int t = --tight[u];
    if (t == 0 && !inS[u]) free_add(u); else if (t == 1 && !inS[u]) q_push((int)sumS[u]);
  }
  if (tight[v] == 0) free_add(v);
}
static void add_free(void) {
  while (Fn > 0) {
    int v = -1;
    for (int tr = 0; tr < 32 && Fn > 0; tr++) { int c = Fl[rnext() % Fn]; if (tabu[c] <= itg) { v = c; break; } }
    if (v < 0) { for (int i = 0; i < Fn; i++) if (tabu[Fl[i]] <= itg) { v = Fl[i]; break; } }
    if (v < 0) break;
    S_add(v);
  }
}
static int *Lb; static int Lc = 0;
static int try_swap(int x) {
  int m = 0;
  for (int i = 0; i < deg[x]; i++) {
    int u = adj[x][i];
    if (tight[u] == 1 && !inS[u] && tabu[u] <= itg) { if (m == Lc) { Lc = Lc ? 2 * Lc : 1024; Lb = realloc(Lb, Lc * sizeof(int)); } Lb[m++] = u; }
  }
  if (m < 2) return 0;
  int st = rnext() % m;
  for (int a0 = 0; a0 < m; a0++) {
    int a = Lb[(st + a0) % m];
    for (int b0 = a0 + 1; b0 < m; b0++) {
      int b = Lb[(st + b0) % m];
      if (!adjacent(a, b)) { S_del(x); S_add(a); S_add(b); return 1; }
    }
  }
  return 0;
}
static void local_search(void) {
  add_free();
  while (Qn > 0) {
    int j = rnext() % Qn; int x = Q[j]; Q[j] = Q[--Qn]; inQ[x] = 0;
    if (!inS[x]) continue;
    if (try_swap(x)) add_free();
  }
}
static const char *argval(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

int main(int argc, char **argv) {
  const char *gf = argv[1];
  const char *init = argval(argc, argv, "-i", NULL);
  const char *out = argval(argc, argv, "-o", "misg_out.txt");
  uint64_t seed = strtoull(argval(argc, argv, "-s", "1"), NULL, 10);
  double T = atof(argval(argc, argv, "-T", "60"));
  int target = atoi(argval(argc, argv, "-target", "1000000000"));
  double pm = atof(argval(argc, argv, "-pm", "0.05"));
  int quiet = atoi(argval(argc, argv, "-q", "0"));
  rs[0] = seed * 0x9E3779B97F4A7C15ULL + 3; rs[1] = seed ^ 0xABCDEF12345ULL; for (int i = 0; i < 10; i++) rnext();
  FILE *f = fopen(gf, "rb"); if (!f) { perror(gf); return 1; }
  int32_t hdr[2]; if (fread(hdr, 4, 2, f) != 2) return 1;
  n = hdr[0];
  adj = malloc(n * sizeof(int *)); deg = malloc(n * sizeof(int));
  long m2 = 0;
  for (int v = 0; v < n; v++) {
    int32_t d; if (fread(&d, 4, 1, f) != 1) return 1;
    deg[v] = d; adj[v] = malloc((d ? d : 1) * sizeof(int));
    if (fread(adj[v], 4, d, f) != (size_t)d) return 1;
    m2 += d;
  }
  fclose(f);
  if ((size_t)n * n / 8 < 400000000ULL) {
    adjbits = calloc((size_t)n * n / 8 + 1, 1);
    for (int v = 0; v < n; v++) for (int i = 0; i < deg[v]; i++) { size_t k = (size_t)v * n + adj[v][i]; adjbits[k >> 3] |= 1 << (k & 7); }
  }
  if (!quiet) printf("graph n=%d m=%ld\n", n, m2 / 2);
  inS = calloc(n, 1); tight = calloc(n, sizeof(int)); sumS = calloc(n, sizeof(int64_t));
  Sl = malloc(n * sizeof(int)); Spos = malloc(n * sizeof(int)); Fl = malloc(n * sizeof(int)); Fpos = malloc(n * sizeof(int));
  Q = malloc(n * sizeof(int)); inQ = calloc(n, 1); tabu = calloc(n, sizeof(long));
  Sn = 0; Fn = n; Qn = 0;
  for (int i = 0; i < n; i++) { Spos[i] = -1; Fpos[i] = i; Fl[i] = i; }
  if (init) {
    FILE *g = fopen(init, "r"); int x;
    while (fscanf(g, "%d", &x) == 1) if (x >= 0 && x < n && !inS[x] && tight[x] == 0) S_add(x);
    fclose(g);
  }
  local_search();
  int best = Sn; int *bestS = malloc(n * sizeof(int)); memcpy(bestS, Sl, Sn * sizeof(int));
  double t0 = now();
  if (!quiet) { printf("initial %d\n", best); fflush(stdout); }
  int cur = Sn; long lastrep = 0;
  int *snap = malloc(n * sizeof(int));
  while (now() - t0 < T && best < target) {
    itg++;
    int snapn = Sn; memcpy(snap, Sl, Sn * sizeof(int));
    int k = 1; if (runif() < pm) k = 2 + rnext() % 3;
    for (int r = 0; r < k; r++) {
      int u = -1;
      // choose a non-solution vertex, biased to low tightness
      for (int tr = 0; tr < 8; tr++) {
        int c = rnext() % n; if (inS[c]) continue;
        if (u < 0 || tight[c] < tight[u]) u = c;
      }
      if (u < 0) continue;
      for (int i = 0; i < deg[u]; i++) { int w = adj[u][i]; if (inS[w]) { S_del(w); tabu[w] = itg + 1; } }
      S_add(u);
    }
    local_search();
    int nn = Sn, acc;
    if (nn >= cur) acc = 1; else { int d = cur - nn, ds = best - nn; acc = runif() < 1.0 / (1.0 + d * (double)ds * 2); }
    if (acc) {
      cur = nn;
      if (cur > best) { best = cur; memcpy(bestS, Sl, Sn * sizeof(int)); if (!quiet) { printf("best %d it %ld t=%.1f\n", best, itg, now() - t0); fflush(stdout); }
        FILE *o = fopen(out, "w"); for (int i = 0; i < best; i++) fprintf(o, "%d\n", bestS[i]); fclose(o); }
    } else {
      for (int i = 0; i < snapn; i++) inS[snap[i]] |= 2;
      int *rem = malloc(Sn * sizeof(int)), rn = 0;
      for (int i = 0; i < Sn; i++) if (!(inS[Sl[i]] & 2)) rem[rn++] = Sl[i];
      for (int i = 0; i < snapn; i++) inS[snap[i]] &= 1;
      for (int i = 0; i < rn; i++) S_del(rem[i]);
      for (int i = 0; i < snapn; i++) if (!inS[snap[i]]) S_add(snap[i]);
      free(rem); cur = Sn;
    }
    if (!quiet && itg - lastrep >= 200000) { lastrep = itg; printf("it %ld cur %d best %d t=%.0f\n", itg, cur, best, now() - t0); fflush(stdout); }
  }
  FILE *o = fopen(out, "w"); for (int i = 0; i < best; i++) fprintf(o, "%d\n", bestS[i]); fclose(o);
  printf("final best %d\n", best);
  return 0;
}
