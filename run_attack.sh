#!/bin/bash

for f in $(ls -t -r grads/); do
    echo "File -> $f"
    regex='([a-zA-Z0-9\-]+).npz'
    [[ $f =~ $regex ]]
    pycmd="inference_attack_IFCA_old.py -f ${BASH_REMATCH[1]}"
    echo $pycmd
    python $pycmd
done

