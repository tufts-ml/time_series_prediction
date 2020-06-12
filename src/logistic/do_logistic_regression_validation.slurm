#!/usr/bin/env bash
#SBATCH -n 30                # Number of cores
#SBATCH -t 0-24:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 20000  # Memory (in MB) per cpu
#SBATCH -o log_%j_logistic_reg_validation.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j_logistic_reg_validation.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Path to directory with github code
SOURCE_PATH="/cluster/home/prath01/projects/mimic3_benchmarks/Code/time_series_prediction/src"

# Load the right conda environment
source activate bdl2019f_readonly

# set some environment variables if not defined
if [[ -z $filename_prefix ]]; then
    filename_prefix='logistic_regression'
fi

# Create 2 state simulated data
# echo "Creating 2 state simulated data"
# python $SOURCE_PATH/logistic/create_synthetic_dataset_for_lr_validation.py \
#    --T 100 \
#    --D 1 \
#    --N 600 \
    
# Create 3 state simulated data
echo "Creating 3 state simulated data"
python $SOURCE_PATH/logistic/create_harder_synthetic_dataset_for_lr_validation.py \
    --T 100 \
    --D 1 \
    --N 600 \

echo "Running classifier"
# Pass along all ENV variables as arguments to my Python script
python $SOURCE_PATH/logistic/main_mimic.py \
    --is_data_simulated True \
    --report_dir $SOURCE_PATH/logistic/html \
    --output_filename_prefix $filename_prefix \

source deactivate
