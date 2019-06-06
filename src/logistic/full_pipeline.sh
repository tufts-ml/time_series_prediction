#!/bin/bash

SOURCE_PATH="/Users/oliver/hughes/time_series_prediction/src"
TS_DATA_PATH="/Users/oliver/hughes/time_series_prediction/datasets/mimic/v20190406_500_seqs/vitals_data_per_tstamp.csv"
TS_METADATA_PATH="/Users/oliver/hughes/time_series_prediction/datasets/mimic/v20190406_500_seqs/metadata_per_seq.csv"
TS_DATA_DICT_PATH="/Users/oliver/hughes/time_series_prediction/docs/mimic-iii-v1.4/20190406/mimic_dict.json"
TEMP_DATA_PATH="/Users/oliver/hughes/time_series_prediction/src/data/mimic_test"
REPORT_DIR="/Users/oliver/hughes/time_series_prediction/src/logistic/html"

echo "Align to grid"
python3 $SOURCE_PATH/align_to_grid.py \
    --input_ts_csv_path $TS_DATA_PATH \
    --data_dict $TS_DATA_DICT_PATH \
    --step_size 1 \
    --output $TEMP_DATA_PATH/temp.csv

echo "Fill missing values"
python3 $SOURCE_PATH/fill_missing_values.py \
    --data $TEMP_DATA_PATH/temp.csv \
    --data_dict $TS_DATA_DICT_PATH \
    --multiple_strategies True \
    --strategy carry_forward \
    --second_strategy pop_mean \
    --output $TEMP_DATA_PATH/temp.csv \
    --third_strategy nulls

echo "Normalize Features"
python3 $SOURCE_PATH/normalize_features.py \
    --input $TEMP_DATA_PATH/temp.csv \
    --data_dict $TS_DATA_DICT_PATH \
    --output $TEMP_DATA_PATH/temp.csv 

echo "Feature transformations collapse"
python3 $SOURCE_PATH/feature_transformation.py \
    --input $TEMP_DATA_PATH/temp.csv \
    --data_dict $TS_DATA_DICT_PATH \
    --output $TEMP_DATA_PATH/temp.csv \
    --data_dict_output $TEMP_DATA_PATH/temp_dd.json \
    --collapse 

echo "Split dataset"
python3 $SOURCE_PATH/split_dataset.py \
    --input $TEMP_DATA_PATH/temp.csv \
    --data_dict $TEMP_DATA_PATH/temp_dd.json \
    --test_size 0.1 \
    --output_dir $TEMP_DATA_PATH/collapsed_test_train/ 

echo "Run classifier" 
python3 $SOURCE_PATH/logistic/main_mimic.py \
    --train_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/train.csv \
    --test_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/test.csv \
    --metadata_csv $TS_METADATA_PATH \
    --data_dict $TEMP_DATA_PATH/temp_dd.json \
    --report_dir $REPORT_DIR

