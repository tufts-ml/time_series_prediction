#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi


## number of epochs
export n_epochs=100000

## choose 2 state or 3 state simulated data
for simulation_data in 2-state 3-state
do
    export simulation_data=$simulation_data
    
## define learning rates for optimizer
## for lr in 1000 100 10 1 0.1 0.01 0.001 0.0001 0.00001
for lr in 10 9 7 5 3 1 0.9 0.7 0.5 0.3 0.1 0.01 0.001  
do
    export lr=$lr
    
## Architecture size (num hidden units)
for arch in 032 128
do
    export hidden_layer_sizes=$arch
    export filename_prefix="$simulation_data-rnn-simulated-data-max-epochs=$n_epochs-arch=$arch-lr=$lr"

    ## Use this line to see where you are in the loop
    echo "simulation_data=$simulation_data max-epochs=$n_epochs  hidden_layer_sizes=$hidden_layer_sizes lr=$lr"

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < do_rnn_validation.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_rnn_validation.slurm
    fi

done
done
done