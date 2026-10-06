#!/bin/bash
cd /home/user/Mathe-Problem
W=$1; S0=$2; S1=$3
for s in $(seq $S0 $S1); do
  python3 scripts/framevar.py $s 45 > /dev/null || continue
  ./kissQ minimax -i runs/frames/fv_s$s.txt -o runs/frames/fm$s -a 1 -s $s -hmax 200000 > runs/frames/fm$s.log 2>&1
  echo "seed $s $(tail -n 1 runs/frames/fm$s.log)" >> runs/frames/campaign_w$W.txt
done
