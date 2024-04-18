#!/bin/bash

rm ./new-forward.cu
if [ -z "$1" ]; then
    echo "Provide an optimization to run (ex. op_1)"
    exit 1
fi
ln -s /u/tjurczyk/ece408/Project/project/m3/$1/new-forward.cu /u/tjurczyk/ece408/Project/project/src/layer/custom/new-forward.cu
