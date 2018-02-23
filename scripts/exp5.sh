#!/bin/bash

k=5
for i in 12 6 7 8 11 5 13
do
    a="out"
    c=$a$i$k
    
    julia ./mds-scale.jl 12 $k 1 > $c &
done

