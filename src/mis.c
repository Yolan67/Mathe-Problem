// mis.c -- kissing configurations inside an integer shell {v in Z^11 : |v|^2 = R}
//
// Two shell vectors u,v conflict iff 2<u,v> > R (i.e. cos > 1/2).  A kissing
// configuration is an independent set of the conflict graph.  The graph is
// implicit (millions of vertices); neighbourhoods are computed on demand with
// a vectorised scan over structure-of-arrays int8 coordinates and cached.
//
// Search: iterated local search of Andrade, Resende & Werneck (ARW 2012):
// perturbation by forced insertion + (1,2)-swap local search.
//
// usage: mis -R 16 [-i init.txt] [-o prefix] [-s seed] [-T seconds]
//            [-types list]   (restrict candidate types, e.g. "4,2222,31111111")
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define D 11
static int R = 16, HALF;  // conflict iff dot > HALF  (dot*2 > R)
static int M = 0;
static int8_t *A;          // AoS M x 16
static int8_t *C[D];       // SoA
static int8_t *acc;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
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

// ------------------------------------------------------------ candidates
static int cap = 0;
static const char *typefilter = NULL;
static int type_ok(const int *v) {
  if (!typefilter) return 1;
  int a[D], n = 0;
  for (int k = 0; k < D; k++) if (v[k]) a[n++] = abs(v[k]);
  // sort descending
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (a[j] > a[i]) { int t = a[i]; a[i] = a[j]; a[j] = t; }
  char buf[64];
  int p = 0;
  for (int i = 0; i < n; i++) p += sprintf(buf + p, "%d", a[i]);
  // match whole token in comma separated list
  const char *s = typefilter;
  size_t L = strlen(buf);
  while (*s) {
    const char *e = strchr(s, ',');
    size_t l = e ? (size_t)(e - s) : strlen(s);
    if (l == L && !strncmp(s, buf, l)) return 1;
    if (!e) break;
    s = e + 1;
  }
  return 0;
}
static void add_cand(const int *v) {
  if (!type_ok(v)) return;
  if (M == cap) { cap = cap ? 2 * cap : 1 << 20; A = realloc(A, (size_t)cap * 16); }
  int8_t *a = A + (size_t)M * 16;
  memset(a, 0, 16);
  for (int k = 0; k < D; k++) a[k] = (int8_t)v[k];
  M++;
}
static void gen(int *v, int k, int rem) {
  if (k == D) { if (rem == 0) add_cand(v); return; }
  int m = (int)floor(sqrt((double)rem) + 1e-9);
  for (int x = -m; x <= m; x++) {
    v[k] = x;
    gen(v, k + 1, rem - x * x);
  }
  v[k] = 0;
}
static void build_soa(void) {
  for (int k = 0; k < D; k++) {
    C[k] = aligned_alloc(64, ((size_t)M + 64) / 64 * 64 + 64);
    memset(C[k], 0, ((size_t)M + 64) / 64 * 64 + 64);
    for (int i = 0; i < M; i++) C[k][i] = A[(size_t)i * 16 + k];
  }
  acc = aligned_alloc(64, ((size_t)M + 64) / 64 * 64 + 64);
}
static inline int dotA(int u, int v) {
  const int8_t *a = A + (size_t)u * 16, *b = A + (size_t)v * 16;
  int s = 0;
  for (int k = 0; k < D; k++) s += a[k] * b[k];
  return s;
}

// hash for lookup of vectors
static int *htab; static size_t hsz;
static uint64_t hvec(const int8_t *a) {
  uint64_t h = 1469598103934665603ULL;
  for (int k = 0; k < D; k++) { h ^= (uint8_t)a[k]; h *= 1099511628211ULL; }
  return h;
}
static void hbuild(void) {
  hsz = 1; while (hsz < (size_t)M * 2) hsz <<= 1;
  htab = malloc(hsz * sizeof(int));
  memset(htab, -1, hsz * sizeof(int));
  for (int i = 0; i < M; i++) {
    size_t h = hvec(A + (size_t)i * 16) & (hsz - 1);
    while (htab[h] >= 0) h = (h + 1) & (hsz - 1);
    htab[h] = i;
  }
}
static int hfind(const int8_t *a) {
  size_t h = hvec(a) & (hsz - 1);
  while (htab[h] >= 0) {
    if (!memcmp(A + (size_t)htab[h] * 16, a, D)) return htab[h];
    h = (h + 1) & (hsz - 1);
  }
  return -1;
}

// ------------------------------------------------------------ neighbourhoods
typedef struct { int *v; int n; } nlist;
static nlist *NB;  // cache, indexed by vertex (lazy)
static long nb_computed = 0;
static size_t nb_mem = 0;

static void compute_nb(int x, nlist *out) {
  const int8_t *xv = A + (size_t)x * 16;
  size_t Mp = ((size_t)M + 63) / 64 * 64;
  memset(acc, 0, Mp);
  for (int k = 0; k < D; k++) {
    int8_t c = xv[k];
    if (!c) continue;
    const int8_t *ck = C[k];
    __m512i cc = _mm512_set1_epi16(c);
    for (size_t i = 0; i < Mp; i += 64) {
      __m512i a = _mm512_load_si512((const void *)(acc + i));
      __m512i b = _mm512_load_si512((const void *)(ck + i));
      // widen multiply: split into lo/hi 16-bit lanes
      __m512i blo = _mm512_srai_epi16(_mm512_slli_epi16(b, 8), 8);
      __m512i bhi = _mm512_srai_epi16(b, 8);
      __m512i plo = _mm512_mullo_epi16(blo, cc);
      __m512i phi = _mm512_mullo_epi16(bhi, cc);
      __m512i p = _mm512_or_si512(_mm512_and_si512(plo, _mm512_set1_epi16(0x00ff)), _mm512_slli_epi16(phi, 8));
      a = _mm512_add_epi8(a, p);
      _mm512_store_si512((void *)(acc + i), a);
    }
  }
  int cnt = 0, capn = 1024;
  int *lst = malloc(capn * sizeof(int));
  __m512i th = _mm512_set1_epi8((int8_t)HALF);
  for (size_t i = 0; i < Mp; i += 64) {
    __m512i a = _mm512_load_si512((const void *)(acc + i));
    uint64_t m = _mm512_cmpgt_epi8_mask(a, th);
    while (m) {
      int b = __builtin_ctzll(m);
      m &= m - 1;
      int v = (int)(i + b);
      if (v >= M || v == x) continue;
      if (cnt == capn) { capn *= 2; lst = realloc(lst, capn * sizeof(int)); }
      lst[cnt++] = v;
    }
  }
  out->v = realloc(lst, (cnt ? cnt : 1) * sizeof(int));
  out->n = cnt;
  nb_computed++;
  nb_mem += cnt * sizeof(int);
}
static inline nlist *nb(int x) {
  if (!NB[x].v) compute_nb(x, &NB[x]);
  return &NB[x];
}
static void nb_evict_all_but_solution(const uint8_t *inS) {
  for (int i = 0; i < M; i++)
    if (NB[i].v && !inS[i]) { nb_mem -= NB[i].n * sizeof(int); free(NB[i].v); NB[i].v = NULL; NB[i].n = 0; }
}

// ------------------------------------------------------------ solution state
static uint8_t *inS;
static uint16_t *tight;
static int64_t *sumS;  // sum of indices of solution neighbours
static int *Q, Qn = 0; static uint8_t *inQ;  // work queue of solution vertices
static int *Sl, *Spos, Sn = 0;
static int *Fl, *Fpos, Fn = 0;  // free vertices (tight==0, not in S)
static long tabu_until_dummy;
static int *tabu;  // iteration until which vertex cannot be inserted
static long it_global = 0;

static inline void free_add(int v) { if (Fpos[v] < 0) { Fpos[v] = Fn; Fl[Fn++] = v; } }
static inline void free_del(int v) {
  int p = Fpos[v];
  if (p >= 0) { int w = Fl[--Fn]; Fl[p] = w; Fpos[w] = p; Fpos[v] = -1; }
}
static inline void q_push(int x) { if (!inQ[x]) { inQ[x] = 1; Q[Qn++] = x; } }
static void S_add(int v) {
  inS[v] = 1;
  Spos[v] = Sn; Sl[Sn++] = v;
  free_del(v);
  nlist *L = nb(v);
  for (int i = 0; i < L->n; i++) {
    int u = L->v[i];
    sumS[u] += v;
    if (tight[u]++ == 0) free_del(u);
  }
  q_push(v);
}
static void S_del(int v) {
  inS[v] = 0;
  int p = Spos[v]; int w = Sl[--Sn]; Sl[p] = w; Spos[w] = p; Spos[v] = -1;
  nlist *L = nb(v);
  for (int i = 0; i < L->n; i++) {
    int u = L->v[i];
    sumS[u] -= v;
    int t = --tight[u];
    if (t == 0 && !inS[u]) free_add(u);
    else if (t == 1 && !inS[u]) q_push((int)sumS[u]);
  }
  if (tight[v] == 0) free_add(v);
}

// ------------------------------------------------------------ local search
// add free vertices (respecting tabu) ; returns number added
static int add_free(void) {
  int added = 0;
  while (Fn > 0) {
    // pick a random free vertex that is not tabu
    int tries = 0, v = -1;
    while (tries < 64 && Fn > 0) {
      int c = Fl[rnext() % Fn];
      if (tabu[c] <= it_global) { v = c; break; }
      tries++;
    }
    if (v < 0) {
      for (int i = 0; i < Fn; i++) if (tabu[Fl[i]] <= it_global) { v = Fl[i]; break; }
      if (v < 0) break;
    }
    S_add(v);
    added++;
  }
  return added;
}

// try a (1,2)-swap around solution vertex x; returns 1 on success
static int *Lbuf; static int Lcap = 0;
static int try_swap(int x) {
  nlist *L = nb(x);
  int m = 0;
  for (int i = 0; i < L->n; i++) {
    int u = L->v[i];
    if (tight[u] == 1 && !inS[u] && tabu[u] <= it_global) {
      if (m == Lcap) { Lcap = Lcap ? 2 * Lcap : 4096; Lbuf = realloc(Lbuf, Lcap * sizeof(int)); }
      Lbuf[m++] = u;
    }
  }
  if (m < 2) return 0;
  // random order search for non-adjacent pair
  int start = rnext() % m;
  for (int a0 = 0; a0 < m; a0++) {
    int a = Lbuf[(start + a0) % m];
    for (int b0 = a0 + 1; b0 < m; b0++) {
      int b = Lbuf[(start + b0) % m];
      if (2 * dotA(a, b) <= R) {
        S_del(x);
        S_add(a);
        S_add(b);
        return 1;
      }
    }
  }
  return 0;
}
static void local_search(void) {
  add_free();
  while (Qn > 0) {
    int j = rnext() % Qn;
    int x = Q[j]; Q[j] = Q[--Qn]; inQ[x] = 0;
    if (!inS[x]) continue;
    if (try_swap(x)) add_free();
  }
}
static void queue_all(void) { for (int i = 0; i < Sn; i++) q_push(Sl[i]); }

// ------------------------------------------------------------ io
static void save_sol(const char *fn) {
  FILE *f = fopen(fn, "w");
  for (int i = 0; i < Sn; i++) {
    const int8_t *a = A + (size_t)Sl[i] * 16;
    for (int k = 0; k < D; k++) fprintf(f, "%d%c", a[k], k == D - 1 ? '\n' : ' ');
  }
  fclose(f);
}
static const char *argval(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

int main(int argc, char **argv) {
  R = atoi(argval(argc, argv, "-R", "16"));
  HALF = R / 2;  // conflict iff dot > R/2 ; for odd R dot > floor(R/2)
  const char *init = argval(argc, argv, "-i", NULL);
  const char *out = argval(argc, argv, "-o", "mis");
  uint64_t seed = strtoull(argval(argc, argv, "-s", "1"), NULL, 10);
  double T = atof(argval(argc, argv, "-T", "600"));
  typefilter = argval(argc, argv, "-types", NULL);
  double pmulti = atof(argval(argc, argv, "-pm", "0.1"));
  rs[0] = seed * 0x9E3779B97F4A7C15ULL + 1; rs[1] = seed ^ 0x1234567ULL;
  for (int i = 0; i < 10; i++) rnext();
  double t0 = now();
  int v[D] = {0};
  gen(v, 0, R);
  printf("R=%d candidates M=%d (%.1fs)\n", R, M, now() - t0);
  build_soa();
  hbuild();
  NB = calloc(M, sizeof(nlist));
  inS = calloc(M, 1);
  tight = calloc(M, sizeof(uint16_t));
  sumS = calloc(M, sizeof(int64_t));
  Q = malloc(M * sizeof(int)); inQ = calloc(M, 1);
  Sl = malloc(M * sizeof(int)); Spos = malloc(M * sizeof(int));
  Fl = malloc(M * sizeof(int)); Fpos = malloc(M * sizeof(int));
  tabu = calloc(M, sizeof(int));
  for (int i = 0; i < M; i++) { Spos[i] = -1; Fpos[i] = i; Fl[i] = i; }
  Fn = M;
  double tn = now();
  nlist tmp; compute_nb(0, &tmp);
  printf("nb scan time %.2f ms, deg(0)=%d\n", (now() - tn) * 1e3, tmp.n);
  free(tmp.v);
  if (init) {
    FILE *f = fopen(init, "r");
    char line[4096];
    int nin = 0, miss = 0;
    while (fgets(line, sizeof line, f)) {
      int8_t a[16] = {0};
      char *p = line;
      int ok = 1;
      for (int k = 0; k < D; k++) { char *q; long x = strtol(p, &q, 10); if (q == p) { ok = 0; break; } a[k] = (int8_t)x; p = q; }
      if (!ok) continue;
      int id = hfind(a);
      if (id < 0) { miss++; continue; }
      if (tight[id] == 0 && !inS[id]) { S_add(id); nin++; }
      else miss++;
    }
    fclose(f);
    printf("init: %d inserted, %d missing/conflicting\n", nin, miss);
  }
  queue_all();
  local_search();
  int best = Sn;
  char fn[512];
  snprintf(fn, sizeof fn, "%s_best_s%llu.txt", out, (unsigned long long)seed);
  save_sol(fn);
  printf("after LS: %d (%.1fs)\n", Sn, now() - t0);
  fflush(stdout);
  // undo log for ILS
  int *logv = malloc(sizeof(int) * 1000000); int8_t *logop = malloc(1000000);
  long last_report = 0;
  int cur = Sn;
  while (now() - t0 < T) {
    it_global++;
    // record state via log: we log every S_add/S_del during this iteration
    // simple approach: snapshot S list
    int *snap = malloc(Sn * sizeof(int));
    int snapn = Sn;
    memcpy(snap, Sl, Sn * sizeof(int));
    // perturbation: force-insert k vertices
    int k = 1;
    if (runif() < pmulti) k = 2 + (rnext() % 3);
    for (int r = 0; r < k; r++) {
      // choose a random non-solution vertex near the solution: pick random x in S, random neighbor u of x
      int u = -1;
      for (int tr = 0; tr < 50; tr++) {
        int x = Sl[rnext() % Sn];
        nlist *L = nb(x);
        int c = L->v[rnext() % L->n];
        if (inS[c]) continue;
        if (u < 0 || tight[c] < tight[u]) u = c;
        if (tight[u] <= 2 && runif() < 0.5) break;
      }
      if (u < 0 || inS[u]) continue;
      // remove conflicting
      nlist *Lu = nb(u);
      for (int i = 0; i < Lu->n; i++) {
        int w = Lu->v[i];
        if (inS[w]) { S_del(w); tabu[w] = it_global + 1; }
      }
      S_add(u);
    }
    local_search();
    int newn = Sn;
    int accept;
    if (newn >= cur) accept = 1;
    else {
      int d = cur - newn, ds = best - newn;
      accept = runif() < 1.0 / (1.0 + d * (double)ds * 4);
    }
    if (accept) {
      cur = newn;
      if (cur > best) {
        best = cur;
        save_sol(fn);
        printf("NEW BEST %d at it %ld t=%.1fs\n", best, it_global, now() - t0);
        fflush(stdout);
      }
    } else {
      // revert: remove all, add snapshot
      // efficient revert: compute differences
      for (int i = 0; i < snapn; i++) inS[snap[i]] |= 2;  // mark
      int *rem = malloc(Sn * sizeof(int)); int rn = 0;
      for (int i = 0; i < Sn; i++) if (!(inS[Sl[i]] & 2)) rem[rn++] = Sl[i];
      for (int i = 0; i < snapn; i++) inS[snap[i]] &= 1;
      for (int i = 0; i < rn; i++) S_del(rem[i]);
      for (int i = 0; i < snapn; i++) if (!inS[snap[i]]) S_add(snap[i]);
      free(rem);
      cur = Sn;
    }
    free(snap);
    if (it_global - last_report >= 200) {
      last_report = it_global;
      printf("it %ld cur %d best %d nbcache %.0fMB comp %ld t=%.0fs\n", it_global, cur, best, nb_mem / 1e6, nb_computed, now() - t0);
      fflush(stdout);
      if (nb_mem > 6e9) nb_evict_all_but_solution(inS);
    }
  }
  printf("final best %d\n", best);
  return 0;
}
