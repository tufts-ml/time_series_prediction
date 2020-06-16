#!/usr/bin/env bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 0-24:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 8000  # Memory (in MB) per cpu
#SBATCH -o log_%j_rnn_validation.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j_rnn_validation.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Path to directory with github code
SOURCE_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/src"

# Path to store the performance results
REPORT_DIR="/cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/rnn/simulated_data_results"

# Load the right conda environment
source activate bdl2019f_readonly

# set some environment variables if not defined
if [[ -z $n_epochs ]]; then
    n_epochs=350;
fi
if [[ -z $hidden_layer_sizes ]]; then
    hidden_layer_sizes=32
fi
if [[ -z $lr ]]; then
    lr=10
fi
if [[ -z $simulation_data ]]; then
    simulation_data='2-state'
fi
if [[ -z $filename_prefix ]]; then
    filename_prefix="$simulation_data-rnn-simulated-data-max-epochs=$n_epochs-lr=$lr-arch=$hidden_layer_sizes"
fi
if [[ -z $batch_size ]]; then
    batch_size=160
fi

if [[ $simulation_data == '2-state' ]]; then
    ## Create simulated data with 2 states
    echo "Creating simulated data with 2 states"
    simulated_data_dir='simulated_data/2-state'    
    python $SOURCE_PATH/rnn/create_synthetic_dataset_for_rnn_validation.py \
        --T 100 \
        --D 1 \
        --N 600 \
        --save_dir $simulated_data_dir
    
elif [[ $simulation_data == '3-state' ]]; then
    # Create simulated data with 3 states
    echo "Creating simulated data with 3 states"
    simulated_data_dir='simulated_data/3-state'
    python $SOURCE_PATH/rnn/create_harder_synthetic_dataset_for_rnn_validation.py \
        --T 100 \
        --D 1 \
        --N 600 \
        --save_dir $simulated_data_dir
fi

echo "running RNN with params max_epochs=$n_epochs lr=$lr arch=$hidden_layer_sizes on simulated data with $simulation_data$"
# Pass along all ENV variables as arguments to my Python script
python $SOURCE_PATH/rnn/main_mimic.py \
    --is_data_simulated True \
    --epochs $n_epochs \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_units $hidden_layer_sizes \
    --report_dir $REPORT_DIR \
    --simulated_data_dir $SOURCE_PATH/rnn/$simulated_data_dir \
    --output_filename_prefix $filename_prefix \

source deactivate