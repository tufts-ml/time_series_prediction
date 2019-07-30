#!/bin/bash

SOURCE_PATH="/cluster/home/onewla01/hughes/time_series_prediction/src"
TS_DATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/vitals_data_per_tstamp.csv"
TS_METADATA_PATH="/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/tidy/mimic3benchmarks_inhospital_mortality/20190406/metadata_per_seq.csv"
TS_DATA_DICT_PATH="/cluster/home/onewla01/hughes/time_series_prediction/docs/mimic-iii-v1.4/20190406/mimic_dict.json"
TEMP_DATA_PATH="/cluster/tufts/hugheslab/onewla01/mimic"
REPORT_DIR="/cluster/tufts/hugheslab/onewla01/mimic/html"

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

echo "Run classifier" 
python $SOURCE_PATH/logistic/main_mimic.py \
    --train_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/train.csv \
    --test_vitals_csv $TEMP_DATA_PATH/collapsed_test_train/test.csv \
    --metadata_csv $TS_METADATA_PATH \
    --data_dict $TEMP_DATA_PATH/temp_dd.json \
    --report_dir $REPORT_DIR

