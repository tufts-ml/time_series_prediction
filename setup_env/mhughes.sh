#!/bin/env bash
# Setup script to set all environment variables 

## Setup mac laptop
if [[ $HOME == "/Users/mhughes" ]]; then
    if [[ -z $PROJECT_REPO_DIR ]]; then
        PROJECT_REPO_DIR=$HOME/git/time_series_prediction/
    fi
fi


