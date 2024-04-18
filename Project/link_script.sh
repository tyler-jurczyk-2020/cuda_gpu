#!/bin/bash

if [ -e /u/tjurczyk/ece408/Project/project/src/layer/custom/new-forward.cu ]; then
    echo "Cleaning up old symlink"
    rm /u/tjurczyk/ece408/Project/project/src/layer/custom/new-forward.cu
fi

if [ -z "$1" ]; then
    echo "Provide an optimization to run (ex. base, op_1, etc)"
    exit 1
fi
ln -s /u/tjurczyk/ece408/Project/project/m3/$1/new-forward.cu /u/tjurczyk/ece408/Project/project/src/layer/custom/new-forward.cu
