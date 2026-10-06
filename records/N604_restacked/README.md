# Second exact 604 in R^11: restacked, non-antipodal (not isometric to records/N604)

* `config_exact.txt` – 604 vectors of squared norm 4 with coordinates in Q(sqrt2) (`a:b` = a + b*sqrt2);
  validated by `validate/verify_qr2.py` (exact): VALID, max inner product exactly 1/2 (`validation_exact.txt`).
* Construction (`scripts/restack.py 8`): slice the exact 604 of `records/N604` along e_8.  Heights are
  0, ±1/2, ±1: equator E (402 points), upper layer U (100) + pole, lower layer L (100) + pole.  Points at
  heights >= 1/2 and <= -1/2 can never conflict, so L may be replaced by g(L) for any symmetry g of E.  One of
  the 16384 signed permutations preserving E gives a new lower layer.
* Invariants: 22840 contacts (the original has 22904), only 476 of 604 points have their antipode in the
  set (the original is antipodal), 15 distinct inner products, one non-trivial infinitesimal flex.
* Deepest holes: cos 0.6034 (original: 1/sqrt3); 605 attempts from it reached max cos 0.5106 (invalid).
