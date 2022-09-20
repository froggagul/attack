#!/bin/bash

# for seed in 54323 27 80 39  # 1 2 3 54323 4 5 6
for seed in 54323 27 80 39
do
    for var in 3 4 5 6 7
    do
        warmup=100
        pycmd="main.py -t smile -a glasses --pi 1 -ni 6000 -c 1 -nc 3 -nw $var --warmup $warmup -ms 32345 -ds $seed"
        echo $pycmd
        python $pycmd
    done
done

