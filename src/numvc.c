// numvc.c -- NuMVC (Cai, Su, Luo, Sattar 2013) for minimum vertex cover;
// maximum independent set = complement.  Graph file in misg format.
// usage: numvc GRAPH [-T sec] [-s seed] [-o out] [-target k (MIS size)] [-i init_indep_set]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int n, m;
static int *eu, *ev;          // edges
static int *adjE, *adjOff;    // incident edges per vertex (CSR), and neighbours
static int *nbr;
static int64_t *w;            // edge weights
static int64_t *dscore;
static uint8_t *inC, *conf;
static int64_t *tstamp;
static int *unc, *uncPos, nunc;
static int *Cl, *Cpos, Cn;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rs[2];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(void) { uint64_t s0 = rs[0], s1 = rs[1], r = s0 + s1; s1 ^= s0; rs[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); rs[1] = rotl(s1, 36); return r; }

static inline void unc_add(int e) { uncPos[e] = nunc; unc[nunc++] = e; }
static inline void unc_del(int e) { int p = uncPos[e]; int f = unc[--nunc]; unc[p] = f; uncPos[f] = p; uncPos[e] = -1; }
static void addC(int v) {
  inC[v] = 1; Cpos[v] = Cn; Cl[Cn++] = v;
  dscore[v] = -dscore[v];
  for (int k = adjOff[v]; k < adjOff[v + 1]; k++) {
    int e = adjE[k], u = nbr[k];
    if (!inC[u]) { dscore[u] -= w[e]; unc_del(e); conf[u] = 1; }
    else dscore[u] += w[e];
  }
}
static void remC(int v) {
  inC[v] = 0; int p = Cpos[v]; int f = Cl[--Cn]; Cl[p] = f; Cpos[f] = p; Cpos[v] = -1;
  dscore[v] = -dscore[v];
  conf[v] = 0;
  for (int k = adjOff[v]; k < adjOff[v + 1]; k++) {
    int e = adjE[k], u = nbr[k];
    if (!inC[u]) { dscore[u] += w[e]; unc_add(e); conf[u] = 1; }
    else dscore[u] -= w[e];
  }
}
static const char *argval(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

int main(int argc, char **argv) {
  double T = atof(argval(argc, argv, "-T", "60"));
  uint64_t seed = strtoull(argval(argc, argv, "-s", "1"), NULL, 10);
  const char *out = argval(argc, argv, "-o", "numvc.sol");
  const char *init = argval(argc, argv, "-i", NULL);
  int target = atoi(argval(argc, argv, "-target", "0"));
  rs[0] = seed * 0x9E3779B97F4A7C15ULL + 5; rs[1] = seed ^ 0xC0FFEE1234ULL; for (int i = 0; i < 10; i++) rnext();
  FILE *f = fopen(argv[1], "rb"); int32_t h[2]; if (fread(h, 4, 2, f) != 2) return 1; n = h[0];
  int **adj = malloc(n * sizeof(int *)); int *deg = malloc(n * sizeof(int));
  long m2 = 0;
  for (int v = 0; v < n; v++) { int32_t d; if (fread(&d, 4, 1, f) != 1) return 1; deg[v] = d; adj[v] = malloc((d + 1) * sizeof(int)); if (fread(adj[v], 4, d, f) != (size_t)d) return 1; m2 += d; }
  fclose(f);
  m = (int)(m2 / 2);
  eu = malloc(m * sizeof(int)); ev = malloc(m * sizeof(int));
  adjOff = malloc((n + 1) * sizeof(int)); adjE = malloc(m2 * sizeof(int)); nbr = malloc(m2 * sizeof(int));
  int *fill = calloc(n, sizeof(int));
  adjOff[0] = 0; for (int v = 0; v < n; v++) adjOff[v + 1] = adjOff[v] + deg[v];
  int ec = 0;
  for (int v = 0; v < n; v++) for (int i = 0; i < deg[v]; i++) { int u = adj[v][i]; if (u > v) { eu[ec] = v; ev[ec] = u; adjE[adjOff[v] + fill[v]] = ec; nbr[adjOff[v] + fill[v]++] = u; adjE[adjOff[u] + fill[u]] = ec; nbr[adjOff[u] + fill[u]++] = v; ec++; } }
  w = malloc(m * sizeof(int64_t)); for (int e = 0; e < m; e++) w[e] = 1;
  dscore = calloc(n, sizeof(int64_t)); inC = calloc(n, 1); conf = malloc(n); tstamp = calloc(n, sizeof(int64_t));
  unc = malloc(m * sizeof(int)); uncPos = malloc(m * sizeof(int)); nunc = 0;
  Cl = malloc(n * sizeof(int)); Cpos = malloc(n * sizeof(int)); Cn = 0;
  for (int v = 0; v < n; v++) { conf[v] = 1; Cpos[v] = -1; }
  for (int e = 0; e < m; e++) { uncPos[e] = -1; unc_add(e); dscore[eu[e]] += 1; dscore[ev[e]] += 1; }
  // initial cover: complement of init independent set or greedy
  uint8_t *ind = calloc(n, 1);
  if (init) { FILE *g = fopen(init, "r"); int x; while (fscanf(g, "%d", &x) == 1) ind[x] = 1; fclose(g); }
  else {
    // greedy MIS: min degree first
    int *ord = malloc(n * sizeof(int)); for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = rnext() % (i + 1); int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    uint8_t *blocked = calloc(n, 1);
    for (int i = 0; i < n; i++) { int v = ord[i]; if (!blocked[v]) { ind[v] = 1; for (int k = 0; k < deg[v]; k++) blocked[adj[v][k]] = 1; } }
  }
  for (int v = 0; v < n; v++) if (!ind[v]) addC(v);
  // ensure cover
  while (nunc > 0) { int e = unc[0]; addC(eu[e]); }
  int bestC = Cn; uint8_t *bestInC = malloc(n); memcpy(bestInC, inC, n);
  printf("initial MIS %d\n", n - bestC); fflush(stdout);
  double t0 = now();
  int64_t step = 0;
  double gamma = 0.5 * n, rho = 0.3;
  int64_t wsum = m;
  // remove one vertex
  #define BEST_DSCORE_IN_C(res) { int64_t bd = INT64_MIN; res = Cl[0]; for (int i = 0; i < Cn; i++) { int v = Cl[i]; if (dscore[v] > bd || (dscore[v] == bd && tstamp[v] < tstamp[res])) { bd = dscore[v]; res = v; } } }
  int v; BEST_DSCORE_IN_C(v); remC(v); tstamp[v] = step;
  while (1) {
    if (nunc == 0) {
      if (Cn < bestC) {
        bestC = Cn; memcpy(bestInC, inC, n);
        printf("MIS %d step %lld t=%.1f\n", n - bestC, (long long)step, now() - t0); fflush(stdout);
        FILE *o = fopen(out, "w"); for (int u = 0; u < n; u++) if (!bestInC[u]) fprintf(o, "%d\n", u); fclose(o);
        if (target && n - bestC >= target) break;
      }
      int u; BEST_DSCORE_IN_C(u); remC(u); tstamp[u] = step;
      continue;
    }
    if ((step & 1023) == 0 && now() - t0 > T) break;
    step++;
    // choose u in C with highest dscore (sampled for speed: scan subset)
    int u = Cl[0]; int64_t bd = INT64_MIN;
    int samples = Cn < 64 ? Cn : 64;
    for (int s = 0; s < samples; s++) { int c = Cl[rnext() % Cn]; if (dscore[c] > bd || (dscore[c] == bd && tstamp[c] < tstamp[u])) { bd = dscore[c]; u = c; } }
    remC(u); conf[u] = 0; tstamp[u] = step;
    // choose random uncovered edge, add endpoint with conf and higher dscore
    int e = unc[rnext() % nunc];
    int a = eu[e], b = ev[e], x;
    if (!conf[a]) x = b; else if (!conf[b]) x = a;
    else x = (dscore[a] > dscore[b] || (dscore[a] == dscore[b] && tstamp[a] < tstamp[b])) ? a : b;
    addC(x); tstamp[x] = step;
    // weights
    for (int i = 0; i < nunc; i++) { int ee = unc[i]; w[ee]++; dscore[eu[ee]]++; dscore[ev[ee]]++; }
    wsum += nunc;
    if (wsum > gamma * m) {
      // forgetting: w = floor(rho*w), recompute dscores
      wsum = 0;
      for (int ee = 0; ee < m; ee++) { w[ee] = (int64_t)(rho * w[ee]); if (w[ee] < 1) w[ee] = 1; wsum += w[ee]; }
      for (int vv = 0; vv < n; vv++) dscore[vv] = 0;
      for (int ee = 0; ee < m; ee++) {
        int p = eu[ee], q = ev[ee];
        if (inC[p] && !inC[q]) dscore[p] -= w[ee];
        else if (!inC[p] && inC[q]) dscore[q] -= w[ee];
        else if (!inC[p] && !inC[q]) { dscore[p] += w[ee]; dscore[q] += w[ee]; }
      }
    }
  }
  printf("final MIS %d\n", n - bestC);
  FILE *o = fopen(out, "w"); for (int u = 0; u < n; u++) if (!bestInC[u]) fprintf(o, "%d\n", u); fclose(o);
  return 0;
}
