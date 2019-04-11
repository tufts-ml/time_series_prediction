#!/bin/env bash
# Setup script to set all environment variables 

## Setup mac laptop
if [[ $HOME == "/cluster/home/bperpe01" ]]; then
    if [[ -z $PROJECT_REPO_DIR ]]; then
        PROJECT_REPO_DIR=$HOME/time_series_prediction/
    fi
fi


