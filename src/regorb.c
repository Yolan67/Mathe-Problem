// regorb.c -- orbits of D11-region candidates under a group of signed permutations, and the
// weighted orbit compatibility graph (misw format).  Candidates (sqrt8-scaled F2 coordinates):
// vectors with entries in {0,±1,±h} (h = 1/sqrt2), norm 8, >= 5 entries ±1  (all compatible with D11).
// usage: regorb GENFILE OUTPREFIX     GENFILE lines: p0..p10 s0..s10  (x'_{p_i} = s_i x_i)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NC 11
typedef struct { int8_t p[NC], s[NC]; } gel;
static gel *G; static int ng = 0, gcap = 0;
static int gel_eq(const gel *a, const gel *b) { return !memcmp(a, b, sizeof(gel)); }
static gel gmul(const gel *a, const gel *b) { // (a*b)(x) = a(b(x))
  gel c; for (int i = 0; i < NC; i++) { int j = b->p[i]; c.p[i] = a->p[j]; c.s[i] = b->s[i] * a->s[j]; } return c; }
static uint64_t gkey(const gel *g) { uint64_t k = 0; for (int i = 0; i < NC; i++) k = k * 16 + g->p[i]; for (int i = 0; i < NC; i++) k = k * 2 + (g->s[i] < 0); return k; }
#define HS (1 << 23)
static uint64_t *ht = NULL;
static void gadd(gel g) {
  if (!ht) { ht = calloc(HS, sizeof(uint64_t)); }
  uint64_t k = gkey(&g) + 1, h = (k * 0x9E3779B97F4A7C15ULL) >> 41;
  while (ht[h & (HS - 1)]) { if (ht[h & (HS - 1)] == k) return; h++; }
  ht[h & (HS - 1)] = k;
  if (ng == gcap) { gcap = gcap ? 2 * gcap : 1024; G = realloc(G, gcap * sizeof(gel)); } G[ng++] = g; }
// values: code 0:0, 1:+1, 2:-1, 3:+h, 4:-h
static const double VAL[5] = {0, 1, -1, 0.70710678118654752, -0.70710678118654752};
static int neg(int c) { return c == 0 ? 0 : (c == 1 ? 2 : c == 2 ? 1 : c == 3 ? 4 : 3); }
static int32_t *idx;   // base-5 code -> candidate index
static int8_t (*C)[NC]; static int nc = 0, ccap = 0;
static uint32_t enc(const int8_t *v) { uint32_t k = 0; for (int i = NC - 1; i >= 0; i--) k = k * 5 + v[i]; return k; }
static void gen(int pos, int n1, int nh, int8_t *v) {
  if (pos == NC) { if (n1 == 0 && nh == 0) { if (nc == ccap) { ccap = ccap ? 2 * ccap : 1 << 20; C = realloc(C, ccap * NC); } memcpy(C[nc], v, NC); idx[enc(v)] = nc; nc++; } return; }
  int rem = NC - pos; if (n1 + nh > rem) return;
  v[pos] = 0; gen(pos + 1, n1, nh, v);
  if (n1) { v[pos] = 1; gen(pos + 1, n1 - 1, nh, v); v[pos] = 2; gen(pos + 1, n1 - 1, nh, v); }
  if (nh) { v[pos] = 3; gen(pos + 1, n1, nh - 1, v); v[pos] = 4; gen(pos + 1, n1, nh - 1, v); }
  v[pos] = 0;
}
static void act(const gel *g, const int8_t *v, int8_t *w) { for (int i = 0; i < NC; i++) { int c = v[i]; w[g->p[i]] = g->s[i] > 0 ? c : neg(c); } }
static double ip(const int8_t *a, const int8_t *b) { double s = 0; for (int i = 0; i < NC; i++) s += VAL[a[i]] * VAL[b[i]]; return s; }
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "r"); const char *out = argv[2];
  gel gens[4096]; int ngen = 0;
  while (1) { gel g; int ok = 1; for (int i = 0; i < NC && ok; i++) { int x; if (fscanf(f, "%d", &x) != 1) ok = 0; g.p[i] = x; }
    for (int i = 0; i < NC && ok; i++) { int x; if (fscanf(f, "%d", &x) != 1) ok = 0; g.s[i] = x; } if (!ok) break; gens[ngen++] = g; }
  fclose(f);
  gel e; for (int i = 0; i < NC; i++) { e.p[i] = i; e.s[i] = 1; } gadd(e);
  for (int i = 0; i < ng; i++) for (int j = 0; j < ngen; j++) { gel h = gmul(&gens[j], &G[i]); gadd(h); if (ng > 2000000) { fprintf(stderr, "group too big\n"); return 1; } }
  fprintf(stderr, "group order %d\n", ng);
  idx = malloc(sizeof(int32_t) * 48828125); memset(idx, 0xff, sizeof(int32_t) * 48828125);
  int8_t v[NC] = {0};
  int types[4][2] = {{8, 0}, {7, 2}, {6, 4}, {5, 6}};
  for (int t = 0; t < 4; t++) gen(0, types[t][0], types[t][1], v);
  fprintf(stderr, "candidates %d\n", nc);
  int32_t *orb = malloc(nc * sizeof(int32_t)); for (int i = 0; i < nc; i++) orb[i] = -1;
  int norb = 0; int *rep = NULL, *osz = NULL, *ostart = NULL; int32_t *mem = malloc(nc * sizeof(int32_t)); int nm = 0;
  int8_t w[NC];
  for (int i = 0; i < nc; i++) if (orb[i] < 0) {
    rep = realloc(rep, (norb + 1) * sizeof(int)); osz = realloc(osz, (norb + 1) * sizeof(int)); ostart = realloc(ostart, (norb + 1) * sizeof(int));
    rep[norb] = i; ostart[norb] = nm; int sz = 0;
    for (int g = 0; g < ng; g++) { act(&G[g], C[i], w); int j = idx[enc(w)]; if (j < 0) { fprintf(stderr, "closure error\n"); return 1; } if (orb[j] < 0) { orb[j] = norb; mem[nm++] = j; sz++; } }
    osz[norb] = sz; norb++;
  }
  fprintf(stderr, "orbits %d\n", norb);
  // self-compatible orbits
  int *ok = calloc(norb, sizeof(int)); int nok = 0;
  for (int o = 0; o < norb; o++) { int good = 1; const int8_t *r = C[rep[o]];
    for (int k = 0; k < osz[o] && good; k++) { int j = mem[ostart[o] + k]; if (j == rep[o]) continue; if (ip(r, C[j]) > 4 + 1e-9) good = 0; }
    ok[o] = good; nok += good; }
  fprintf(stderr, "self-compatible orbits %d\n", nok);
  int *vid = malloc(norb * sizeof(int)), *ov = malloc(nok * sizeof(int)); int nv = 0;
  for (int o = 0; o < norb; o++) { vid[o] = -1; if (ok[o]) { vid[o] = nv; ov[nv++] = o; } }
  // adjacency
  int **adj = calloc(nv, sizeof(int *)); int *deg = calloc(nv, sizeof(int)), *cap = calloc(nv, sizeof(int));
  long m = 0;
  for (int a = 0; a < nv; a++) {
    const int8_t *ra = C[rep[ov[a]]];
    for (int b = a + 1; b < nv; b++) {
      int ob = ov[b], conf = 0;
      for (int k = 0; k < osz[ob] && !conf; k++) if (ip(ra, C[mem[ostart[ob] + k]]) > 4 + 1e-9) conf = 1;
      if (conf) {
        if (deg[a] == cap[a]) { cap[a] = cap[a] ? 2 * cap[a] : 16; adj[a] = realloc(adj[a], cap[a] * sizeof(int)); }
        if (deg[b] == cap[b]) { cap[b] = cap[b] ? 2 * cap[b] : 16; adj[b] = realloc(adj[b], cap[b] * sizeof(int)); }
        adj[a][deg[a]++] = b; adj[b][deg[b]++] = a; m++;
      }
    }
  }
  fprintf(stderr, "orbit graph: %d vertices, %ld edges\n", nv, m);
  char fn[512]; snprintf(fn, sizeof fn, "%s.wgraph", out);
  FILE *o = fopen(fn, "wb"); int32_t hdr[2] = {nv, (int32_t)m}; fwrite(hdr, 4, 2, o);
  for (int a = 0; a < nv; a++) { int32_t h2[2] = {osz[ov[a]], deg[a]}; fwrite(h2, 4, 2, o); fwrite(adj[a], 4, deg[a], o); }
  fclose(o);
  snprintf(fn, sizeof fn, "%s.reps", out); o = fopen(fn, "w");
  for (int a = 0; a < nv; a++) { const int8_t *r = C[rep[ov[a]]]; fprintf(o, "%d", osz[ov[a]]); for (int i = 0; i < NC; i++) fprintf(o, " %.17g", VAL[r[i]]); fprintf(o, "\n"); }
  fclose(o);
  snprintf(fn, sizeof fn, "%s.members", out); o = fopen(fn, "w");
  for (int a = 0; a < nv; a++) { int ob = ov[a]; for (int k = 0; k < osz[ob]; k++) { const int8_t *r = C[mem[ostart[ob] + k]]; fprintf(o, "%d", a); for (int i = 0; i < NC; i++) fprintf(o, " %d", r[i]); fprintf(o, "\n"); } }
  fclose(o);
  return 0;
}
