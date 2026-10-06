#!/bin/bash
cd /home/user/Mathe-Problem
for p in "$@"; do
  ./misg shells/hf1.graph -x runs/excl/${p}_x.txt -i runs/excl/${p}_init.txt -o runs/excl/${p}.sol -T 240 -s 7 -q 1 > runs/excl/${p}_mis.log 2>&1
  echo "$p $(tail -n 1 runs/excl/${p}_mis.log)" >> runs/excl/summary.txt
done
