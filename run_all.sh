#!/bin/bash

for seed in 123 23 58 39 20 30 # 1 2 3 54323 4 5 6
do
    var=12
    warmup=100

    pycmd="main.py -t smile -a glasses --pi 1 -ni 7000 -c 1 -nc 3 -nw $var --warmup $warmup -ms 32345 -ds $seed"
    echo $pycmd
    python $pycmd
done


