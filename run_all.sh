#!/bin/bash

for warmup in 100 # 500 1000 1500 2000 2500 3000 3500 4000 4500
do
    var=17
    pycmd="main.py -t smile -a glasses --pi 1 -ni 6000 -c 0 -nc 3 -nw $var --warmup $warmup"
    echo $pycmd
    python $pycmd
done

