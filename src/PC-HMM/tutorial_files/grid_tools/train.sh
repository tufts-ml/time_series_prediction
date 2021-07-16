#!/bin/bash

. $PCPYROOT/grid_tools/setup_train.sh

echo "Running PC-Model training"
eval $TFPYTHONEXE -u $PCPYROOT/grid_tools/train.py $XHOST_RESULTS_DIR
