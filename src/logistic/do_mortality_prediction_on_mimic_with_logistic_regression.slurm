#!/usr/bin/env bash
#SBATCH -n 30                # Number of cores
#SBATCH -t 0-24:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 20000  # Memory (in MB) per cpu
#SBATCH -o log_%j_logistic_reg_mimic.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j_logistic_reg_mimic.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Path to directory with github code
SOURCE_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/src"

# Paths to raw dataset
TS_DATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/vitals_data_per_tstamp.csv"
TS_METADATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/metadata_per_seq.csv"
TS_DATA_DICT_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/docs/mimic-iii-v1.4/20190406/mimic_dict.json"

# Path to directory in which modified dataset files will be stored
TEMP_DATA_PATH="/cluster/tufts/hugheslab/prath01/projects/mimic3_benchmarks/code_results/no_fill/"

# Path to directory in which html classifier performance report should be saved
REPORT_DIR="$SOURCE_PATH/logistic/html/no_fill/"

# Load the right conda environment
source activate bdl2019f_readonly

# echo "Predicting mortality on MIMIC with logistic regresssion"
# Pass along all ENV variables as arguments to my Python script
# $SOURCE_PATH/logistic/full_pipeline.sh


echo "Run classifier" 
python $SOURCE_PATH/logistic/main_mimic.py \
    --train_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/train.csv \
    --test_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/test.csv \
    --metadata_csv $TS_METADATA_PATH \
    --data_dict $TEMP_DATA_PATH/temp_dd.json \
    --report_dir $REPORT_DIR

source deactivate
