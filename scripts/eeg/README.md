EEG dataset

# Prereqs

It's assumed you have the `tspred_env` conda environment installed locally.

For detailed instructions, see `../setup_env/README.md`.

# Workflow

You can use the given 'Makefile' to complete every step of the required pipeline. The associated commands are:

```
help                         Show help messages for each command
download_raw_dataset         Download dataset from UCI repository
build_std_dataset_from_raw   Build standardized flat file time-series dataset
align_to_grid                Build time-series aligned to regular intervals
normalize_features           Build time series with normalized feature cols
collapse_ts                  Collapse time-series into fixed-size feature vector
split_into_train_and_test    Split into train and test
```
