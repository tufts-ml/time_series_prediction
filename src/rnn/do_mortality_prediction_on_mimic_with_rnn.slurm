#!/usr/bin/env bash
#SBATCH -n 10                # Number of cores
#SBATCH -t 0-48:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 8000  # Memory (in MB) per cpu
#SBATCH -o /cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/rnn/model_logs/log_%j_rnn_mimic.out       # Write stdout to file named log_JOBIDNUM.out in logs dir
#SBATCH -e /cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/rnn/model_logs/log_%j_rnn_mimic.err       # Write stderr to file named log_JOBIDNUM.err in logs dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Path to directory with github code
SOURCE_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/src"
TEMP_DATA_PATH="/cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/rnn"
TS_METADATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/metadata_per_seq.csv"
TS_DATA_DICT_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/docs/mimic-iii-v1.4/20190406/mimic_dict.json"
REPORT_DIR="/cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/rnn/performance_results"

# Load the right conda environment
source activate bdl2019f_readonly

# set some environment variables if not defined
if [[ -z $n_epochs ]]; then
    n_epochs=100000;
fi
if [[ -z $hidden_layer_sizes ]]; then
    hidden_layer_sizes=32
fi
if [[ -z $lr ]]; then
    lr=10
fi
if [[ -z $filename_prefix ]]; then
    filename_prefix="$-rnn-mimic-data-max-epochs=$n_epochs-lr=$lr-arch=$hidden_layer_sizes"
fi
if [[ -z $batch_size ]]; then
    batch_size=256
fi

# echo "pre-processing data..."
# run full_pipeline.sh to fill missing values, split train test etc.
# $SOURCE_PATH/rnn/full_pipeline.sh

echo "running RNN with params max_epochs=$n_epochs lr=$lr arch=$hidden_layer_sizes on mimic data$"
# Pass along all ENV variables as arguments to my Python script
python $SOURCE_PATH/rnn/main_mimic.py \
    --train_vitals_csv $TEMP_DATA_PATH/ts_test_train/train.csv \
    --test_vitals_csv $TEMP_DATA_PATH/ts_test_train/test.csv \
    --metadata_csv $TS_METADATA_PATH \
    --data_dict $TS_DATA_DICT_PATH \
    --batch_size 3000 \
    --epochs $n_epochs \
    --lr $lr \
    --hidden_units $hidden_layer_sizes \
    --report_dir $REPORT_DIR \
    --output_filename_prefix $filename_prefix \

source deactivate