# Kissing configurations in dimension 11

Goal: as many unit vectors x_1..x_N in R^11 as possible with <x_i,x_j> <= 1/2 (i != j).
Baseline given at the start: N = 593 (AlphaEvolve 2025).

## Result

| N   | status | file | certificate |
|-----|--------|------|-------------|
| **604** | exact, valid | `records/N604/config_exact.txt` (= `best/best_exact.txt`) | exact arithmetic in Q(√2), all norms exactly 4, max inner product exactly 1/2 |
| 600 | exact, valid | `records/N600/config_exact.txt` | exact rational arithmetic |
| 598 | exact, valid | `records/N598/config_exact.txt` | exact arithmetic in Q(√2) |

604 equals the best value known in the literature (2026: EinsteinArena / "The Station" multi-agent
systems; our exact 604 has 22904 contacts, the same count as one of the published 604s).
No configuration with 605 points was found (see "Attempts beyond 604").

Validate:
```
python3 validate/verify_qr2.py records/N604/config_exact.txt   # exact, Q(sqrt2)
python3 validate/verify.py     records/N600/config_exact.txt   # exact, rationals
```
Both validators are independent of the search code (pure Python, fractions, no floats in the decision).

## Structure of the 604 (R^11 = R^8 ⊕ R^3, coordinates 0..7 | 8,9,10)
* D8 part (112, w = 0): ±2e_s (s < 8) and (±1)^4 on the 6 unions of two of the pairs
  {0,1},{2,3},{4,5},{6,7}.
* Three "lifted spinor" layers (384): the 8 transversal blocks of the extended Hamming code
  (E8 = D8 ∪ 128 spinors) are replaced by their triples: (±1)^3 on a transversal triple plus ±1 on
  one T-coordinate k; each layer = 8 triples pairwise meeting in ≤ 1 point (`configs/layer30_blocks.txt`).
* Four D5 slices (96): in each R^5 = pair ⊕ R^3, the vectors u = (±1,±1) on the pair combined with
  the 6 frame vectors ±√2 f_i, f = orthonormal frame with |coordinates| ≤ 1/√2.
* Core (12): cuboctahedron √2(±f_a ± f_b) in R^3.
Each pair-slice R^5 then contains a full D5 root system (40 = kissing number of R^5).

## How it was found
1. Literature search: the 594–600 constructions share a 496-vector backbone (description found).
2. Rebuilt the backbone (`scripts/layers.py`), enumerated exact candidates in its free region
   (`scripts/freecands.py`) and solved max independent set (`src/misg.c`): **598**.
3. Continuous optimization with fixed backbone (`src/kiss.c`, mode `grow`): numerical 600 → exact
   rational certificate (`scripts/exactify.py`): **600**; then numerical 604 within seconds.
4. Reading off the structure of the numerical 604 → exact construction `scripts/build604.py`: **604**.

## Attempts beyond 604 (all unsuccessful)
* Continuous basin hopping from 604+1 with fixed backbone / partially fixed / all points free,
  threshold annealing, Metropolis acceptance: stuck at penalty energy ≈ 0.07–0.08.
* Riesz-energy optimization (exponent up to ~800) of 604+1 points: best max cosine 0.51369
  (min angle 59.09°) — `candidates/near605_maxcos0.5137_INVALID.txt` (NOT valid).
* Log-sum-exp minimax (smoothing parameter up to 10^5) on 604+1, starting from generic numerical
  604 variants (different frame per pair vector): best max cosine **0.507162** (min angle 59.525°),
  `candidates/near605_maxcos0.50716_INVALID.txt` (NOT valid). A multistart over 23 further 604
  variants (`scripts/multistart605.sh`, `logs/multistart605_summary.txt`) gave 0.50724–0.5100;
  minimax basin hopping (141 hops) did not improve 0.507162. The 604+1 basins therefore sit
  consistently about 0.007 in cosine (≈0.47°) away from a valid 605.
* Exact/combinatorial: MIS over relaxed candidate sets (backbone vectors removable), fibered
  candidate sets R^8 ⊕ R^3 with integer/half-integer u-parts and a catalogue of ~60–100 exact
  directions, weighted item formulation under S-sign symmetry (`scripts/itemgraph.py`, `src/misw.c`):
  maximum found 604.
* Structural bounds inside the family: layer triples ≤ 26 (found 24), each pair-slice is a full D5
  (kissing-optimal), regions between two slices refill to exactly 48 again.
* Other templates (|T| = 2,4,5, other coordinate groupings, E8 + two R^7 slices):
  486 / 509 / 527 / 581 / 373 — all worse than 604.

## Layout
* `validate/` exact validators · `src/` C search engines (`kiss.c` continuous, `misg.c`/`misw.c`/`mis.c` MIS)
* `scripts/` constructions and candidate generators · `configs/` building blocks
* `records/N*/` validated records with README, validation logs · `best/` current best (never overwritten by a smaller N)
* `logs/` logs of successful runs · `candidates/` invalid near-misses (clearly marked)
