python align_to_grid.py ../datasets/eeg/eeg_ts.csv ../docs/eeg_spec.json 1 _aligned.csv
# python fill_missing_values.py --data aligned.csv --strategy carry_forward]
  mv _aligned.csv _ts_filled.csv
python normalize_features.py _ts_filled.csv ../docs/eeg_spec.json _normalized.csv
# python transform_features.py _normalized.csv args...]
  mv _normalized.csv _transformed.csv
python split_dataset.py _transformed.csv ../docs/eeg_spec.json 0.1 0.1
rm _*.csv