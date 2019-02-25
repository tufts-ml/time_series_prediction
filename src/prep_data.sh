echo "Aligning to grid"
python align_to_grid.py \
    --input ../datasets/eeg/eeg_ts.csv \
    --data_dict ../docs/eeg_spec.json \
    --step_size 1 \
    --output _aligned.csv
echo "Filling missing values"
python fill_missing_values.py \
    --data _aligned.csv \
    --static ../datasets/eeg/eeg_static.csv \
    --data_dict ../docs/eeg_spec.json \
    --static_dict ../docs/eeg_spec.json \
    --strategy carry_forward
echo "Normalizing features"
python normalize_features.py \
    --input ts_filled.csv \
    --data_dict ../docs/eeg_spec.json \
    --output _normalized.csv
# Note: slow
echo "Transforming features"
python feature_transformation.py \
    --data _normalized.csv \
    --data_dict ../docs/eeg_spec.json \
    --collapse 
echo "Splitting dataset"
python split_dataset.py \
    --input ts_transformed.csv \
    --data_dict transformed.json \
    --test_size 0.1 \
    --output_dir .
echo "Evaluating classifier"
python eval_classifier.py logistic \
    --ts_dir . \
    --data_dict transformed.json \
    --static_file ../datasets/eeg/eeg_static.csv \
    --validation_size 0.1 \
    --class_weight balanced \
    --grid_C 0.01 0.1 1 10 100