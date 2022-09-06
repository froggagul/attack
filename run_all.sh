#!/bin/bash

for warmup in 100 300 500 700 900 1100 1300 1500
do
    var=17
    pycmd="main.py -t smile -a glasses --pi 1 -ni 6000 -c 0 -nc 3 -nw $var --warmup $warmup"
    echo $pycmd
    python $pycmd
done

