python align_to_grid.py ../datasets/eeg/eeg_signal_data.csv 1
# python fill_missing_values.py --data aligned.csv --strategy carry_forward]
mv aligned.csv ts_filled.csv
python normalize_features.py ts_filled.csv ../docs/eeg_spec.json
# python transform_features.py normalized.csv args...]
mv normalized.csv transformed.csv
python split_dataset.py transformed.csv 0.1 0.1