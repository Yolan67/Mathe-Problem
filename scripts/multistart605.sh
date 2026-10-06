#!/bin/bash
# worker: for seeds in range, build a generic numerical 604 (grow from the exact 600, backbone fixed),
# then minimax 605 = 604 + deepest hole.  Logs one line per seed.
cd /home/user/Mathe-Problem
W=$1; S0=$2; S1=$3; OUT=/tmp/claude-0/ms
mkdir -p $OUT
for s in $(seq $S0 $S1); do
  timeout 60 ./kiss2 grow -i records/N600/config_float.txt -f 496 -o $OUT/v -a 1 -s $s -T 45 -c 1 > $OUT/grow_$s.log 2>&1
  f=$OUT/v_N604_s$s.txt
  if [ -f $f ]; then
    ./kiss4 minimax -i $f -f 0 -o $OUT/m -a 1 -s $s -hmax 30000 -rh 400 > $OUT/mm_$s.log 2>&1
    echo "seed $s $(tail -n 1 $OUT/mm_$s.log)" >> $OUT/summary_w$W.txt
  else
    echo "seed $s no604 $(grep SUCCESS $OUT/grow_$s.log | tail -n 1 | cut -c1-20)" >> $OUT/summary_w$W.txt
  fi
done
