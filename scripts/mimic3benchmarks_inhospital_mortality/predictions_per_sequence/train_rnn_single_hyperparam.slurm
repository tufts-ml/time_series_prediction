#!/usr/bin/env bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 0-48:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 30000  # Memory (in MB) per cpu
#SBATCH -o /cluster/tufts/hugheslab/prath01/model_logs/log_%j_rnn_mimic.out       
# Write stdout to file named log_JOBIDNUM.out in logs dir
#SBATCH -e /cluster/tufts/hugheslab/prath01/model_logs/log_%j_rnn_mimic.err       
# Write stderr to file named log_JOBIDNUM.err in logs dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Path to directory with github code
SOURCE_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/src"

# Path for RNN train test data
TRAIN_TEST_SPLIT_PATH="/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/datasets/mimic3_inhospital_mortality/v20201207/split-by=subject_id/features_per_timestep"

# Path to save loss plots and training results
RESULTS_DIR="/tmp/results/mimic3_inhospital_mortality/v20201207/split-by\=subject_id/features_per_timestep/rnn/"

# Path to project repo
PROJECT_REPO_DIR="/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/src"

# Load the right conda environment
source activate bdl2019f_readonly


echo "running RNN with $fn_prefix"
# Pass along all ENV variables as arguments to my Python script

python -u $PROJECT_REPO_DIR/rnn/main_mimic.py \
    --outcome_col_name 'mort_hosp' \
    --output_dir $RESULTS_DIR \
    --train_csv_files $TRAIN_TEST_SPLIT_PATH/x_train.csv,$TRAIN_TEST_SPLIT_PATH/y_train.csv \
    --test_csv_files $TRAIN_TEST_SPLIT_PATH/x_test.csv,$TRAIN_TEST_SPLIT_PATH/y_test.csv \
    --data_dict_files $TRAIN_TEST_SPLIT_PATH/x_dict.json,$TRAIN_TEST_SPLIT_PATH/y_dict.json \
    --validation_size 0.15 \
    --hidden_layers $hidden_layers \
    --hidden_units $hidden_units \
    --lr $lr \
    --batch_size $batch_size \
    --dropout $dropout \
    --weight_decay $weight_decay \
    --output_filename_prefix $fn_prefix \

source deactivate