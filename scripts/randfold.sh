#!/bin/bash
# random folds of the 12-dim 776 -> shrink -> grow ; one line summary per seed
cd /home/user/Mathe-Problem
for s in $(seq $1 $2); do
python3 - $s << 'PY'
import sys, numpy as np
s=int(sys.argv[1]); rng=np.random.default_rng(s)
X=np.loadtxt('/tmp/claude-0/c776.txt'); T=X[:,8:]
Q,_=np.linalg.qr(rng.normal(size=(4,4)))
M3=Q[:,1:4].T; d=Q[:,0]
P=T@M3.T; rest=T@d
# fold: merge 'rest' radially into a random direction of R^3
e=rng.normal(size=3); e/=np.linalg.norm(e)
a=P@e
sg=np.sign(a+1e-9*rng.normal(size=len(a)))
Pn=P-np.outer(a,e)+np.outer(sg*np.sqrt(a**2+rest**2),e)
Y=np.column_stack([X[:,:8],Pn])
np.savetxt(f'/tmp/claude-0/rf_{s}.txt',Y,fmt='%.17g')
PY
./kissS shrink -i /tmp/claude-0/rf_$s.txt -o /tmp/claude-0/rf_$s -s $s > /tmp/claude-0/rf_$s.log 2>&1
f=$(ls /tmp/claude-0/rf_${s}_shrunk_N*_s$s.txt 2>/dev/null | head -1)
n=$(echo $f | sed -E 's/.*_N([0-9]+)_.*/\1/')
timeout 300 ./kiss2 grow -i $f -f 0 -o /tmp/claude-0/rfg_$s -a 1 -s $s -T 240 -c 1 > /tmp/claude-0/rfg_$s.log 2>&1
g=$(grep SUCCESS /tmp/claude-0/rfg_$s.log | tail -n 1 | sed -E 's/.*N=([0-9]+).*/\1/')
echo "seed $s shrunk $n grown ${g:-$n}" >> logs/randfold_summary.txt
done
