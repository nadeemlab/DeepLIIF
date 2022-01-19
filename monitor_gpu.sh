#!/bin/bash

# $PYTHONUSERBASE/bin is the path if gpustat is installed in wml-a training job env
export PATH=$PATH:$PYTHONUSERBASE/bin

while true
    do 
    gpustat
    sleep 5
done