#!/bin/env bash
# Setup script to set all environment variables 

## Setup Tufts HPC for Mike
if [[ $HOME == "/cluster/home/mhughe02" ]]; then
    if [[ -z $PROJECT_REPO_DIR ]]; then
        PROJECT_REPO_DIR=/cluster/tufts/hugheslab/code/time_series_prediction/
    fi
fi


