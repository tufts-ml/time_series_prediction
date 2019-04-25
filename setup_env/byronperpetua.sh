#!/bin/env bash
# Setup script to set all environment variables 

## Setup mac laptop
if [[ $HOME == "/Users/byronperpetua" ]]; then
    if [[ -z $PROJECT_REPO_DIR ]]; then
        PROJECT_REPO_DIR=$HOME/Documents/School/Tufts/project/time_series_prediction/
    fi    
fi
