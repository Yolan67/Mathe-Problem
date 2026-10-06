// autgen.c -- all orthogonal maps preserving a finite point set X (N x d, unit vectors), by backtracking
// over images of a basis with Gram-matrix pruning and colour refinement.  Output: permutations (one per line).
// usage: autgen POINTS.txt d OUT.perms [maxsol]
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int N, d; static double *X; static int64_t *K; static int *col; static long maxsol = 50000000;
static int *B, *img; static long nsol = 0; static FILE *fo;
static uint64_t *hk; static int *hv; static uint64_t HS;
static uint64_t vkey(const double *x) { uint64_t h = 1469598103934665603ULL; for (int k = 0; k < d; k++) { int64_t v = llround(x[k] * 1e6); h ^= (uint64_t)v; h *= 1099511628211ULL; } return h; }
static void hput(uint64_t k, int i) { uint64_t h = k & (HS - 1); while (hv[h] >= 0) h = (h + 1) & (HS - 1); hk[h] = k; hv[h] = i; }
static int hget(uint64_t k) { uint64_t h = k & (HS - 1); while (hv[h] >= 0) { if (hk[h] == k) return hv[h]; h = (h + 1) & (HS - 1); } return -1; }
static double *Minv;   // inverse of basis matrix (d x d), rows = basis vectors
static int inv(double *A, double *Ai, int n) { // Gauss-Jordan
  double *M = malloc(sizeof(double) * n * 2 * n);
  for (int i = 0; i < n; i++) for (int j = 0; j < 2 * n; j++) M[i * 2 * n + j] = j < n ? A[i * n + j] : (j - n == i);
  for (int c = 0; c < n; c++) { int p = c; for (int r = c + 1; r < n; r++) if (fabs(M[r * 2 * n + c]) > fabs(M[p * 2 * n + c])) p = r;
    if (fabs(M[p * 2 * n + c]) < 1e-12) { free(M); return 0; }
    for (int j = 0; j < 2 * n; j++) { double t = M[c * 2 * n + j]; M[c * 2 * n + j] = M[p * 2 * n + j]; M[p * 2 * n + j] = t; }
    double v = M[c * 2 * n + c]; for (int j = 0; j < 2 * n; j++) M[c * 2 * n + j] /= v;
    for (int r = 0; r < n; r++) if (r != c) { double f = M[r * 2 * n + c]; if (f != 0) for (int j = 0; j < 2 * n; j++) M[r * 2 * n + j] -= f * M[c * 2 * n + j]; } }
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Ai[i * n + j] = M[i * 2 * n + n + j];
  free(M); return 1; }
static void leaf(void) {
  // map g with g(B_j) = X[img_j]:  g = Y^T * (Bm^T)^{-1}  where Bm rows = basis vectors; g x = sum_j c_j X[img_j], c = (Bm^T)^{-1} x
  double g[16 * 16];
  // c-coefficients for x: solve Bm^T c = x  ->  c = Minv^T x  (Minv = (Bm)^{-1}, so (Bm^T)^{-1} = Minv^T)
  for (int a = 0; a < d; a++) for (int b = 0; b < d; b++) { double s = 0; for (int j = 0; j < d; j++) s += X[img[j] * d + a] * Minv[b * d + j]; g[a * d + b] = s; }
  int *perm = malloc(N * sizeof(int)); double y[16];
  for (int i = 0; i < N; i++) {
    for (int a = 0; a < d; a++) { double s = 0; for (int b = 0; b < d; b++) s += g[a * d + b] * X[i * d + b]; y[a] = s; }
    int j = hget(vkey(y)); if (j < 0) { free(perm); return; } perm[i] = j;
  }
  for (int i = 0; i < N; i++) fprintf(fo, "%d%c", perm[i], i == N - 1 ? '\n' : ' ');
  nsol++; free(perm);
}
static void rec(int k) {
  if (nsol >= maxsol) return;
  if (k == d) { leaf(); return; }
  int b = B[k];
  for (int c = 0; c < N; c++) {
    if (col[c] != col[b]) continue;
    int ok = 1;
    for (int j = 0; j < k && ok; j++) if (K[(long)c * N + img[j]] != K[(long)b * N + B[j]] || c == img[j]) ok = 0;
    if (!ok) continue;
    img[k] = c; rec(k + 1);
  }
}
static int cmp64(const void *a, const void *b) { int64_t x = *(int64_t *)a, y = *(int64_t *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
  d = atoi(argv[2]); if (argc > 4) maxsol = atol(argv[4]);
  FILE *f = fopen(argv[1], "r"); int cap = 1024; X = malloc(sizeof(double) * cap * d); N = 0;
  while (1) { if (N == cap) { cap *= 2; X = realloc(X, sizeof(double) * cap * d); } int ok = 1; for (int k = 0; k < d; k++) if (fscanf(f, "%lf", &X[N * d + k]) != 1) ok = 0; if (!ok) break;
    double s = 0; for (int k = 0; k < d; k++) s += X[N * d + k] * X[N * d + k]; s = sqrt(s); for (int k = 0; k < d; k++) X[N * d + k] /= s; N++; }
  fclose(f);
  K = malloc(sizeof(int64_t) * N * N);
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < d; k++) s += X[i * d + k] * X[j * d + k]; K[(long)i * N + j] = llround(s * 1e6); }
  HS = 1; while (HS < 4 * (uint64_t)N) HS <<= 1; hk = malloc(HS * 8); hv = malloc(HS * 4); for (uint64_t i = 0; i < HS; i++) hv[i] = -1;
  for (int i = 0; i < N; i++) hput(vkey(X + i * d), i);
  // colour refinement
  col = calloc(N, sizeof(int)); int64_t *row = malloc(sizeof(int64_t) * N); uint64_t *sig = malloc(8 * N);
  for (int it = 0; it < 4; it++) {
    for (int i = 0; i < N; i++) { for (int j = 0; j < N; j++) row[j] = K[(long)i * N + j] * 1000003LL + col[j]; qsort(row, N, 8, cmp64);
      uint64_t h = 1469598103934665603ULL ^ (uint64_t)col[i]; for (int j = 0; j < N; j++) { h ^= (uint64_t)row[j]; h *= 1099511628211ULL; } sig[i] = h; }
    int nc = 0; for (int i = 0; i < N; i++) { int c = -1; for (int j = 0; j < i; j++) if (sig[j] == sig[i]) { c = col[j]; break; } if (c < 0) c = nc++; col[i] = c; }
    // note: col values compared only for equality
    fprintf(stderr, "refinement %d: %d classes\n", it, nc);
  }
  // basis: greedily from smallest classes
  int *cnt = calloc(N, sizeof(int)); for (int i = 0; i < N; i++) cnt[col[i]]++;
  int *order = malloc(N * sizeof(int)); for (int i = 0; i < N; i++) order[i] = i;
  for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) if (cnt[col[order[j]]] < cnt[col[order[i]]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
  B = malloc(d * sizeof(int)); img = malloc(d * sizeof(int)); int nb = 0;
  double *Bm = malloc(sizeof(double) * d * d); Minv = malloc(sizeof(double) * d * d);
  for (int t = 0; t < N && nb < d; t++) {
    int i = order[t]; B[nb] = i;
    // rank check via Gram-Schmidt
    double v[16]; for (int k = 0; k < d; k++) v[k] = X[i * d + k];
    double tmp[16 * 16];
    for (int j = 0; j < nb; j++) for (int k = 0; k < d; k++) tmp[j * d + k] = X[B[j] * d + k];
    // orthogonalize v against span via least squares using current Bm rows (simple modified GS on copy)
    double Q[16 * 16]; int nq = 0;
    for (int j = 0; j <= nb; j++) { double w[16]; for (int k = 0; k < d; k++) w[k] = j < nb ? tmp[j * d + k] : v[k];
      for (int q = 0; q < nq; q++) { double s = 0; for (int k = 0; k < d; k++) s += w[k] * Q[q * d + k]; for (int k = 0; k < d; k++) w[k] -= s * Q[q * d + k]; }
      double nn = 0; for (int k = 0; k < d; k++) nn += w[k] * w[k]; nn = sqrt(nn);
      if (nn > 1e-8) { for (int k = 0; k < d; k++) Q[nq * d + k] = w[k] / nn; nq++; } }
    if (nq == nb + 1) nb++;
  }
  for (int j = 0; j < d; j++) for (int k = 0; k < d; k++) Bm[j * d + k] = X[B[j] * d + k];
  if (!inv(Bm, Minv, d)) { fprintf(stderr, "singular basis\n"); return 1; }
  // Minv = Bm^{-1}: Bm Minv = I.  For g: need c with x = sum_j c_j B_j = Bm^T c -> c = (Bm^T)^{-1} x = Minv^T x
  fo = fopen(argv[3], "w");
  rec(0);
  fclose(fo);
  fprintf(stderr, "automorphisms: %ld\n", nsol);
  return 0;
}
