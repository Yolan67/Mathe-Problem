// corb.c -- orbit-MIS on the complete candidate set C = {a + b/sqrt2 : |a|^2+|b|^2/2 = 4, a.b = 0}
// under a group of signed permutations of the 11 coordinates.  Candidates read from CANDFILE (int8 a[11], b[11]).
// Writes OUT.wgraph (misw format: weight = orbit size) and OUT.members (orbit id + a,b per member).
// usage: corb GENFILE CANDFILE OUT [maxorbit]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NC 11
typedef struct { int8_t p[NC], s[NC]; } gel;
static gel *G; static int ng = 0, gcap = 0;
static gel gmul(const gel *a, const gel *b) { gel c; for (int i = 0; i < NC; i++) { int j = b->p[i]; c.p[i] = a->p[j]; c.s[i] = b->s[i] * a->s[j]; } return c; }
static uint64_t gkey(const gel *g) { uint64_t k = 0; for (int i = 0; i < NC; i++) k = k * 16 + g->p[i]; for (int i = 0; i < NC; i++) k = k * 2 + (g->s[i] < 0); return k; }
#define HS (1 << 23)
static uint64_t *ht = NULL;
static void gadd(gel g) {
  if (!ht) ht = calloc(HS, sizeof(uint64_t));
  uint64_t k = gkey(&g) + 1, h = (k * 0x9E3779B97F4A7C15ULL) >> 41;
  while (ht[h & (HS - 1)]) { if (ht[h & (HS - 1)] == k) return; h++; }
  ht[h & (HS - 1)] = k;
  if (ng == gcap) { gcap = gcap ? 2 * gcap : 1024; G = realloc(G, gcap * sizeof(gel)); } G[ng++] = g; }
typedef struct { int8_t a[NC], b[NC]; } cv;
static cv *C; static int nc;
static uint64_t ckey(const cv *v) { uint64_t k = 0; for (int i = 0; i < NC; i++) k = k * 25 + (uint64_t)((v->a[i] + 2) * 5 + (v->b[i] + 2)); return k; }
static uint64_t *ck; static int32_t *cidx; static uint64_t CHS;
static void cput(uint64_t k, int i) { uint64_t h = (k * 0x9E3779B97F4A7C15ULL) & (CHS - 1); while (ck[h] != UINT64_MAX) h = (h + 1) & (CHS - 1); ck[h] = k; cidx[h] = i; }
static int cget(uint64_t k) { uint64_t h = (k * 0x9E3779B97F4A7C15ULL) & (CHS - 1); while (ck[h] != UINT64_MAX) { if (ck[h] == k) return cidx[h]; h = (h + 1) & (CHS - 1); } return -1; }
static void act(const gel *g, const cv *v, cv *w) { for (int i = 0; i < NC; i++) { w->a[g->p[i]] = g->s[i] * v->a[i]; w->b[g->p[i]] = g->s[i] * v->b[i]; } }
static double ip(const cv *x, const cv *y) { long r = 0, q = 0, h = 0; for (int i = 0; i < NC; i++) { r += x->a[i] * y->a[i]; h += x->b[i] * y->b[i]; q += x->a[i] * y->b[i] + x->b[i] * y->a[i]; } return r + h * 0.5 + q * 0.70710678118654752; }
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "r"); int maxorb = argc > 4 ? atoi(argv[4]) : 1 << 30;
  static gel gens[4096]; int ngen = 0;
  while (1) { gel g; int ok = 1; for (int i = 0; i < NC && ok; i++) { int x; if (fscanf(f, "%d", &x) != 1) ok = 0; g.p[i] = x; }
    for (int i = 0; i < NC && ok; i++) { int x; if (fscanf(f, "%d", &x) != 1) ok = 0; g.s[i] = x; } if (!ok) break; gens[ngen++] = g; }
  fclose(f);
  gel e; for (int i = 0; i < NC; i++) { e.p[i] = i; e.s[i] = 1; } gadd(e);
  for (int i = 0; i < ng; i++) for (int j = 0; j < ngen; j++) gadd(gmul(&gens[j], &G[i]));
  fprintf(stderr, "group order %d\n", ng);
  f = fopen(argv[2], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  nc = sz / sizeof(cv); C = malloc(sz); if (fread(C, sizeof(cv), nc, f) != (size_t)nc) return 1; fclose(f);
  CHS = 1; while (CHS < 4 * (uint64_t)nc) CHS <<= 1;
  ck = malloc(CHS * 8); memset(ck, 0xff, CHS * 8); cidx = malloc(CHS * 4);
  for (int i = 0; i < nc; i++) cput(ckey(&C[i]), i);
  fprintf(stderr, "candidates %d\n", nc);
  int32_t *orb = malloc(nc * 4); for (int i = 0; i < nc; i++) orb[i] = -1;
  int norb = 0, *rep = NULL, *osz = NULL, *ost = NULL; int32_t *mem = malloc(nc * 4); int nm = 0; cv w;
  for (int i = 0; i < nc; i++) if (orb[i] < 0) {
    rep = realloc(rep, (norb + 1) * sizeof(int)); osz = realloc(osz, (norb + 1) * sizeof(int)); ost = realloc(ost, (norb + 1) * sizeof(int));
    rep[norb] = i; ost[norb] = nm; int s = 0;
    for (int g = 0; g < ng; g++) { act(&G[g], &C[i], &w); int j = cget(ckey(&w)); if (j < 0) { fprintf(stderr, "not closed\n"); return 1; } if (orb[j] < 0) { orb[j] = norb; mem[nm++] = j; s++; } }
    osz[norb++] = s;
  }
  fprintf(stderr, "orbits %d\n", norb);
  int *ok = calloc(norb, sizeof(int)), nok = 0;
  for (int o = 0; o < norb; o++) { if (osz[o] > maxorb) continue; int good = 1;
    for (int k = 0; k < osz[o] && good; k++) { int j = mem[ost[o] + k]; if (j != rep[o] && ip(&C[rep[o]], &C[j]) > 2 + 1e-9) good = 0; }
    ok[o] = good; nok += good; }
  fprintf(stderr, "self-compatible orbits %d\n", nok);
  int *ov = malloc(nok * sizeof(int)), nv = 0; for (int o = 0; o < norb; o++) if (ok[o]) ov[nv++] = o;
  int **adj = calloc(nv, sizeof(int *)), *deg = calloc(nv, sizeof(int)), *cap = calloc(nv, sizeof(int)); long m = 0;
  for (int a = 0; a < nv; a++) for (int b = a + 1; b < nv; b++) {
    int ob = ov[b], conf = 0;
    for (int k = 0; k < osz[ob] && !conf; k++) if (ip(&C[rep[ov[a]]], &C[mem[ost[ob] + k]]) > 2 + 1e-9) conf = 1;
    if (conf) {
      if (deg[a] == cap[a]) { cap[a] = cap[a] ? 2 * cap[a] : 16; adj[a] = realloc(adj[a], cap[a] * sizeof(int)); }
      if (deg[b] == cap[b]) { cap[b] = cap[b] ? 2 * cap[b] : 16; adj[b] = realloc(adj[b], cap[b] * sizeof(int)); }
      adj[a][deg[a]++] = b; adj[b][deg[b]++] = a; m++; }
  }
  fprintf(stderr, "orbit graph %d vertices %ld edges\n", nv, m);
  char fn[512]; snprintf(fn, sizeof fn, "%s.wgraph", argv[3]); FILE *o = fopen(fn, "wb"); int32_t hd[2] = {nv, (int32_t)m}; fwrite(hd, 4, 2, o);
  for (int a = 0; a < nv; a++) { int32_t h2[2] = {osz[ov[a]], deg[a]}; fwrite(h2, 4, 2, o); fwrite(adj[a], 4, deg[a], o); } fclose(o);
  snprintf(fn, sizeof fn, "%s.members", argv[3]); o = fopen(fn, "w");
  for (int a = 0; a < nv; a++) { int ob = ov[a]; for (int k = 0; k < osz[ob]; k++) { cv *x = &C[mem[ost[ob] + k]]; fprintf(o, "%d", a); for (int i = 0; i < NC; i++) fprintf(o, " %d", x->a[i]); for (int i = 0; i < NC; i++) fprintf(o, " %d", x->b[i]); fprintf(o, "\n"); } }
  fclose(o);
  return 0;
}
