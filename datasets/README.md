# Dataset file structure

Each dataset should be contained in a dedicated folder, either here
or somewhere else on your local filesystem more suitable.

Let's call this folder the `DATASET_ROOT`

Typically, the folder will have:

* raw/
    * Contains raw data (e.g. from original source)
    * May be in any format

* vYYYYMMDD/
    * Contains *standardized* dataset in the tidy sequential CSV format
    * One or more *.csv files
    * One or more *.json files

# Refined dataset file structure

Any further transformation should be contained in a separate folder.

Usually, we suggest the following folder structure:

* vYYYYMMDD/split-size={size}-seed={seed}/collapsed_features_per_sequence/
    * x_dict.json
    * x_{split}.csv : subject_id,sequence_id
    * y_dict.json
    * y_{split}.csv : subject_id,sequence_id

* vYYYYMMDD/split-size={size}-seed={seed}/regular_ts_features_per_hour/
    * x_dict.json
    * x_{split}.csv : subject_id,sequence_id,hour
    * y_dict.json
    * y_{split}.csv : subject_id,sequence_id

* vYYYYMMDD/split-size={size}-seed={seed}/irregular_ts_features_per_timestamp/
    * x_dict.json
    * x_{split}.csv : subject_id,sequence_id,timestamp
    * y_dict.json
    * y_{split}.csv : subject_id,sequence_id

Obviously, the specific keys used will change between datasets.
The dictionaries will keep you up to date.

