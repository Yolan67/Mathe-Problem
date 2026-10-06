#!/bin/bash
# R^12 calibration: random 48-system deformations of the 840 family, re-validate, then 841 minimax attempts.
cd /home/user/Mathe-Problem
W=$1; S0=$2; S1=$3
for s in $(seq $S0 $S1); do
  sig=$(python3 -c "import random; random.seed($s); print(round(random.uniform(0.1,0.45),3))")
  python3 scripts/c840family.py --deform $s $sig > /dev/null
  ./kissQ minimax -d 12 -i runs/c840/def_s$s.txt -f 744 -o runs/c840/cv$s -a 0 -s $s -hmax 100000 > runs/c840/cv$s.log 2>&1
  v=$(tail -n 1 runs/c840/cv$s.log | awk '{print $NF}' | cut -d= -f2)
  ok=$(python3 -c "print(1 if float('$v') < 0.5 + 1e-11 else 0)")
  if [ "$ok" = "1" ]; then
    ./kissQ minimax -d 12 -i runs/c840/cv${s}_mm_N840_s$s.txt -o runs/c840/cp$s -a 1 -s $s -hmax 200000 > runs/c840/cp$s.log 2>&1
    echo "seed $s sig $sig valid840 -> 841: $(tail -n 1 runs/c840/cp$s.log)" >> runs/c840/campaign_w$W.txt
  else
    echo "seed $s sig $sig invalid840 $v" >> runs/c840/campaign_w$W.txt
  fi
done
