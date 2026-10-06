#!/bin/bash
# usage: grow.sh piece seconds
cd /home/user/Mathe-Problem
p=$1; T=${2:-600}
python3 -c "
import numpy as np
V=np.load('shells/hf1.npy'); sol=[int(l) for l in open('runs/excl/$p.sol')]
np.savetxt('runs/excl/${p}_cfg.txt', V[sol]/2, fmt='%.17g'); print(len(sol))
"
./kissQ grow -i runs/excl/${p}_cfg.txt -o runs/excl/${p}_g -s 3 -T $T -c 1 > runs/excl/${p}_grow.log 2>&1
echo "$p $(grep -c SUCCESS runs/excl/${p}_grow.log) successes, last: $(grep SUCCESS runs/excl/${p}_grow.log | tail -n 1 | cut -c1-40) | $(tail -n 1 runs/excl/${p}_grow.log)" >> runs/excl/grow_summary.txt
