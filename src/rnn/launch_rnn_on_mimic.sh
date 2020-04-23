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

## define learning rates for optimizer
# for lr in 1000 100 10 1 0.1 0.01 0.001 0.0001 0.00001
# for lr in 100 50 10 9 7 5 3 1 0.9 0.7 0.5 0.3 0.1  
for lr in 100 80 60 50 10 9 7 5 3 1 0.9 0.7 0.5 0.3 0.1 
do
    export lr=$lr
    
## Architecture size (num hidden units)
for arch in 32 128
do
    export hidden_layer_sizes=$arch
    export filename_prefix="rnn-mimic-mortality-prediction-max-epochs=$n_epochs-arch=$arch-lr=$lr"

    ## Use this line to see where you are in the loop
    echo "rnn-mimic max-epochs=$n_epochs  hidden_layer_sizes=$hidden_layer_sizes lr=$lr"

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < do_mortality_prediction_on_mimic_with_rnn.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_mortality_prediction_on_mimic_with_rnn.slurm
    fi

done
done