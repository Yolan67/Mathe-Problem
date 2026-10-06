# N = 600 kissing configuration in R^11

* `config_exact.txt` – 600 vectors with rational coordinates. The first 496 are the
  integer backbone (squared norm 4); the other 104 have norm ≈ 2 and rational
  coordinates (denominator 10^9). The certified configuration is x_i/|x_i|.
* `validation_exact.txt` – `validate/verify.py` (exact integer arithmetic): VALID.
* Max cosine 1/2 is attained only between backbone vectors (17088 exact contacts);
  every pair involving one of the 104 other points has cosine ≤ 0.4999000001.

## How it was found
1. Start: the exact 598 configuration (`records/N598`).
2. `./kiss2 grow -i records/N598/config_float.txt -f 496 -a 1 -s 21 -T 1800 -c 1`
   (backbone fixed, the 102 free points movable; new points inserted at the deepest
   hole, L-BFGS on the penalty sum max(0, cos-1/2)^2). It reached zero energy at
   N = 599 and N = 600 immediately (log: `logs/grow_from598_s21.log`).
3. Slack: `./kiss2 relax -f 496 -m 1e-4` (penalty threshold 1/2 - 1e-4 for all pairs
   involving a free point) → energy exactly 0.
4. `scripts/exactify.py ... 496 ...` rounds the free points to rationals (den 10^9).
