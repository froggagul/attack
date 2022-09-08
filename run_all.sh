#!/bin/bash

for seed in 1 2 3 4 5
do
    for warmup in 300 600 900 1200
    do
        var=12
        pycmd="main.py -t smile -a glasses --pi 1 -ni 8000 -c 0 -nc 3 -nw $var --warmup $warmup -ms 12345 -ds $seed"
    	echo $pycmd
    	python $pycmd
    done
done

