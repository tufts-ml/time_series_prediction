#!/bin/bash

# Path to directory with github code
SOURCE_PATH="/cluster/home/onewla01/hughes/time_series_prediction/src"

# Paths to raw dataset
TS_DATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/vitals_data_per_tstamp.csv"
TS_METADATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/metadata_per_seq.csv"
TS_DATA_DICT_PATH="/cluster/home/onewla01/hughes/time_series_prediction/docs/mimic-iii-v1.4/20190406/mimic_dict.json"

# Path to directory in which modified dataset files will be stored
TEMP_DATA_PATH="/cluster/tufts/hugheslab/onewla01/mimic"

# Path to directory in which html classifier performance report should be saved
REPORT_DIR="$SOURCE_PATH/random_forest/html"

# Check directory and file exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Could not find directory SOURCE_PATH: $SOURCE_PATH"
    exit 1
fi
if [ ! -f "$TS_METADATA_PATH" ]; then
    echo "Could not find file TS_METADATA_PATH: $TS_METADATA_PATH"
    exit 1
fi

# Format data unless user adds command line arg "classifier"
if [ "$1" != "classifier" ]; then

    # Check files and directories exist
    if [ ! -f "$TS_DATA_PATH" ]; then
        echo "Could not find file TS_DATA_PATH: $TS_DATA_PATH"
        exit 1
    fi
    if [ ! -f "$TS_DATA_DICT_PATH" ]; then
        echo "Could not find file TS_DATA_DICT_PATH: $TS_DATA_DICT_PATH"
        exit 1
    fi
    if [ ! -d "$TEMP_DATA_PATH" ]; then
        echo "Could not find directory TEMP_DATA_PATH: $TEMP_DATA_PATH"
        exit 1
    fi

    # Format data
    echo "Align to grid"
    python $SOURCE_PATH/align_to_grid.py \
        --input_ts_csv_path $TS_DATA_PATH \
        --data_dict $TS_DATA_DICT_PATH \
        --step_size 1 \
        --output $TEMP_DATA_PATH/temp.csv

    echo "Fill missing values"
    python $SOURCE_PATH/fill_missing_values.py \
        --data $TEMP_DATA_PATH/temp.csv \
        --data_dict $TS_DATA_DICT_PATH \
        --multiple_strategies True \
        --strategy carry_forward \
        --second_strategy pop_mean \
        --output $TEMP_DATA_PATH/temp.csv \
        --third_strategy nulls

    echo "Normalize Features"
    python $SOURCE_PATH/normalize_features.py \
        --input $TEMP_DATA_PATH/temp.csv \
        --data_dict $TS_DATA_DICT_PATH \
        --output $TEMP_DATA_PATH/temp.csv 

    echo "Feature transformations collapse"
    python $SOURCE_PATH/feature_transformation.py \
        --input $TEMP_DATA_PATH/temp.csv \
        --data_dict $TS_DATA_DICT_PATH \
        --output $TEMP_DATA_PATH/temp.csv \
        --data_dict_output $TEMP_DATA_PATH/temp_dd.json \
        --collapse 

    echo "Split dataset"
    python $SOURCE_PATH/split_dataset.py \
        --input $TEMP_DATA_PATH/temp.csv \
        --data_dict $TEMP_DATA_PATH/temp_dd.json \
        --test_size 0.1 \
        --output_dir $TEMP_DATA_PATH/collapsed_test_train 
fi 

# Check files and directories exist
if [ ! -f "$TEMP_DATA_PATH/collapsed_test_train/train.csv" ]; then
    echo "Could not find directory train.csv in $TEMP_DATA_PATH/collapsed_test_train/"
    exit 1
fi
if [ ! -f "$TEMP_DATA_PATH/collapsed_test_train/test.csv" ]; then
    echo "Could not find directory test.csv in $TEMP_DATA_PATH/collapsed_test_train/"
    exit 1
fi
if [ ! -d "$REPORT_DIR" ]; then
    echo "Could not find directory REPORT_DIR: $REPORT_DIR"
    exit 1
fi

# Run classifier
echo "Run classifier" 
python $SOURCE_PATH/random_forest/main_mimic.py \
    --train_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/train.csv \
    --test_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/test.csv \
    --metadata_csv $TS_METADATA_PATH \
    --data_dict $TEMP_DATA_PATH/temp_dd.json \
    --report_dir $REPORT_DIR

