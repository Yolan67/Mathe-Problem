#!/bin/bash
# frame-variant 604 + 605th point at a random (not deepest) hole, LSE minimax
cd /home/user/Mathe-Problem
W=$1; S0=$2; S1=$3
for s in $(seq $S0 $S1); do
  python3 scripts/framevar.py $s 45 > /dev/null || continue
  ./kissR minimax -i runs/frames/fv_s$s.txt -o runs/frames/rh$s -a 1 -randhole 40 -s $s -hmax 200000 > runs/frames/rh$s.log 2>&1
  echo "seed $s $(grep inserted runs/frames/rh$s.log | cut -c1-40) $(tail -n 1 runs/frames/rh$s.log)" >> runs/frames/campaign_rh_w$W.txt
done
