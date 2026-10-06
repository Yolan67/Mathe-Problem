// misw.c -- maximum weight independent set by iterated local search
// (perturbation = forced insertion, improvement = (w,1)-swaps + greedy fill).
// Graph file: n, m(int32), then per vertex: weight(int32), deg, neighbours.
// usage: misw GRAPH [-i init] [-o out] [-s seed] [-T sec]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int n, **adj, *deg, *wt;
static uint8_t *inS;
static long *nbw;  // total weight of solution neighbours
static int *tightc; // number of solution neighbours
static long cur = 0;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rs[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(void) { uint64_t s0 = rs[0], s1 = rs[1], r = s0 + s1; s1 ^= s0; rs[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); rs[1] = rotl(s1, 36); return r; }
static double runif(void) { return (rnext() >> 11) * (1.0 / 9007199254740992.0); }
static void add(int v) { inS[v] = 1; cur += wt[v]; for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; nbw[u] += wt[v]; tightc[u]++; } }
static void del(int v) { inS[v] = 0; cur -= wt[v]; for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; nbw[u] -= wt[v]; tightc[u]--; } }
static int *order;
static long *tabu; static long it = 0;
static int improve(void) {
  // repeat: (w,1)-swaps : insert v if wt[v] > nbw[v], removing neighbours
  int any = 0, changed = 1;
  while (changed) {
    changed = 0;
    for (int i = n - 1; i > 0; i--) { int j = rnext() % (i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
    for (int k = 0; k < n; k++) {
      int v = order[k];
      if (inS[v] || tabu[v] > it) continue;
      if (wt[v] > nbw[v]) {
        for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; if (inS[u]) del(u); }
        add(v);
        changed = 1; any = 1;
      }
    }
  }
  return any;
}
static const char *argval(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
int main(int argc, char **argv) {
  const char *init = argval(argc, argv, "-i", NULL), *out = argval(argc, argv, "-o", "misw.sol");
  uint64_t seed = strtoull(argval(argc, argv, "-s", "1"), NULL, 10);
  double T = atof(argval(argc, argv, "-T", "60"));
  rs[0] = seed * 0x9E3779B97F4A7C15ULL + 11; rs[1] = seed ^ 0x5555AAAA1234ULL; for (int i = 0; i < 10; i++) rnext();
  FILE *f = fopen(argv[1], "rb"); int32_t h[2]; if (fread(h, 4, 2, f) != 2) return 1; n = h[0];
  adj = malloc(n * sizeof(int *)); deg = malloc(n * sizeof(int)); wt = malloc(n * sizeof(int));
  for (int v = 0; v < n; v++) { int32_t a[2]; if (fread(a, 4, 2, f) != 2) return 1; wt[v] = a[0]; deg[v] = a[1]; adj[v] = malloc((a[1] + 1) * sizeof(int)); if (fread(adj[v], 4, a[1], f) != (size_t)a[1]) return 1; }
  fclose(f);
  inS = calloc(n, 1); nbw = calloc(n, sizeof(long)); tightc = calloc(n, sizeof(int)); order = malloc(n * sizeof(int)); tabu = calloc(n, sizeof(long));
  for (int i = 0; i < n; i++) order[i] = i;
  if (init) { FILE *g = fopen(init, "r"); int x; while (fscanf(g, "%d", &x) == 1) if (!inS[x] && tightc[x] == 0) add(x); fclose(g); }
  improve();
  long best = cur; uint8_t *bestS = malloc(n); memcpy(bestS, inS, n);
  printf("initial %ld\n", best); fflush(stdout);
  double t0 = now();
  uint8_t *snap = malloc(n);
  while (now() - t0 < T) {
    it++;
    memcpy(snap, inS, n); long snapw = cur;
    int k = 1 + (runif() < 0.3) + (runif() < 0.1);
    for (int r = 0; r < k; r++) {
      int v = rnext() % n;
      if (inS[v]) continue;
      for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; if (inS[u]) { del(u); tabu[u] = it + 2; } }
      add(v);
    }
    improve();
    if (cur > best) {
      best = cur; memcpy(bestS, inS, n);
      printf("best %ld it %ld t=%.1f\n", best, it, now() - t0); fflush(stdout);
    }
    if (cur < snapw && runif() > 0.02) {
      // revert
      for (int v = 0; v < n; v++) if (inS[v] && !snap[v]) del(v);
      for (int v = 0; v < n; v++) if (!inS[v] && snap[v]) add(v);
    }
  }
  FILE *o = fopen(out, "w"); for (int v = 0; v < n; v++) if (bestS[v]) fprintf(o, "%d\n", v); fclose(o);
  printf("final best %ld\n", best);
  return 0;
}
