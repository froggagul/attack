#!/bin/bash

var=12
seed=54323
warmup=5
ni=20
pycmd="main.py -t smile -a glasses --pi 1 -ni $ni -c 0 -nc 3 -nw $var --warmup $warmup -ms 12345 -ds $seed"

echo $pycmd
python $pycmd

