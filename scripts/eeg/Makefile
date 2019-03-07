SHELL:=/bin/bash

# Verify required programs (like 'conda' are on the current path)
# Make will terminate immediately if one of these is not available
REQD_EXECUTABLES = ls grep conda
HAS_REQD_EXE := $(foreach exec,$(REQD_EXECUTABLES),\
        $(if $(shell which $(exec)),$(exec),$(error "Required program '$(exec)' not in PATH. Please install or fix path issues.")))

# Default environment variables
# Can override with your setup_env/$USER.sh, or with local env variables
PYTHON_VERSION?=3.6
PROJECT_ENV_NAME?=tspred_env
PROJECT_REPO_DIR?=$(abspath ../../)
EEG_VERSION?=v20190228
EEG_DATASET_TOP_PATH?=$(PROJECT_REPO_DIR)/datasets/eeg/
EEG_DATASET_PATH?=$(EEG_DATASET_TOP_PATH)/$(EEG_VERSION)

CONDA_ENV_PATH:=$(shell conda env list --json | grep ${PROJECT_ENV_NAME} | head -n1 | cut -d'"' -f2)
# Quit early if environment not found
ifneq ($(filter download_raw_dataset build_std_dataset_from_raw,$(MAKECMDGOALS)),$())
ifndef CONDA_ENV_PATH 
$(error CONDA_ENV_PATH not defined)
endif
endif

.PHONY: help
help:                                                 				## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/,/' | column -s, -t

# =====
.PHONY: download_raw_dataset
download_raw_dataset: $(EEG_DATASET_TOP_PATH)/raw/data.csv 				## Download dataset from UCI repository

$(EEG_DATASET_TOP_PATH)/raw/data.csv:
	@{ \
	echo "> DOWNLOADING RAW EEG DATA..."; \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p $(EEG_DATASET_TOP_PATH)/raw/; \
	wget http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv --directory-prefix $(EEG_DATASET_TOP_PATH)/raw/; \
	echo "DONE. Saved to $(EEG_DATASET_TOP_PATH)/raw/data.csv"; \
	}

# =====
.PHONY: build_std_dataset_from_raw 
build_std_dataset_from_raw: $(EEG_DATASET_PATH)/eeg_sensors_ts.csv 		## Build standardized flat file time-series dataset

$(EEG_DATASET_PATH)/eeg_sensors_ts.csv: $(EEG_DATASET_TOP_PATH)/raw/data.csv src/build_std_eeg_dataset_from_raw.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p $(EEG_DATASET_PATH); \
	python -u ./src/build_std_eeg_dataset_from_raw.py \
		--input_csv_path $(EEG_DATASET_TOP_PATH)/raw/data.csv \
		--output_all_csv_path $(EEG_DATASET_PATH)/eeg_all_data.csv \
		--output_sensors_ts_csv_path $(EEG_DATASET_PATH)/eeg_sensors_ts.csv \
		--output_labels_static_csv_path $(EEG_DATASET_PATH)/eeg_labels_static.csv \
		; \
	}

# =====
.PHONY: align_to_grid 
align_to_grid: $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned.csv 		## Build time-series aligned to regular intervals

$(EEG_DATASET_PATH)/eeg_sensors_ts_aligned.csv: 
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/align_to_grid.py \
	    --input_ts_csv_path $(EEG_DATASET_PATH)/eeg_sensors_ts.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/eeg_spec.json \
	    --step_size 1 \
	    --output $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned.csv \
	; \
	}

# =====
.PHONY: normalize_features 
normalize_features: $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned_normalized.csv 		## Build time series with normalized feature cols

$(EEG_DATASET_PATH)/eeg_sensors_ts_aligned_normalized.csv: $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned.csv $(PROJECT_REPO_DIR)/src/normalize_features.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/normalize_features.py \
	    --input $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/eeg_spec.json \
	    --output $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned_normalized.csv \
	; \
	}

# =====
.PHONY: collapse_ts 
collapse_ts: $(EEG_DATASET_PATH)/eeg_sensors_ts_normalized_collapsed.csv 		## Collapse time-series into fixed-size feature vector

$(EEG_DATASET_PATH)/eeg_sensors_ts_normalized_collapsed.csv: $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned_normalized.csv $(PROJECT_REPO_DIR)/src/feature_transformation.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/feature_transformation.py \
	    --input $(EEG_DATASET_PATH)/eeg_sensors_ts_aligned_normalized.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/eeg_spec.json \
	    --output $(EEG_DATASET_PATH)/eeg_sensors_ts_normalized_collapsed.csv \
		--data_dict_output $(EEG_DATASET_PATH)/collapsed.json \
	    --collapse \
	; \
	}


# =====
.PHONY: split_into_train_and_test 
split_into_train_and_test: $(EEG_DATASET_PATH)/train.csv 		## Split into train and test

$(EEG_DATASET_PATH)/train.csv: $(EEG_DATASET_PATH)/eeg_sensors_ts_normalized_collapsed.csv $(PROJECT_REPO_DIR)/src/split_dataset.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/split_dataset.py \
		--input $(EEG_DATASET_PATH)/eeg_sensors_ts_normalized_collapsed.csv \
		--data_dict $(EEG_DATASET_PATH)/collapsed.json \
		--test_size 0.1 \
		--output_dir $(EEG_DATASET_PATH)/ \
	; \
	}
