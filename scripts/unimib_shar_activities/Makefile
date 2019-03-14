SHELL:=/bin/bash

# Verify required programs (like 'conda' are on the current path)
# Make will terminate immediately if one of these is not available
REQD_EXECUTABLES = ls grep conda unzip
HAS_REQD_EXE := $(foreach exec,$(REQD_EXECUTABLES),\
        $(if $(shell which $(exec)),$(exec),$(error "Required program '$(exec)' not in PATH. Please install or fix path issues.")))

# Default environment variables
# Can override with your setup_env/$USER.sh, or with local env variables
PYTHON_VERSION?=3.6
PROJECT_ENV_NAME?=tspred_env
PROJECT_REPO_DIR?=$(abspath ../../)
DATA_VERSION?=v20190307
DATASET_TOP_PATH?=$(PROJECT_REPO_DIR)/datasets/unimib_shar_activities/
DATASET_PATH?=$(DATASET_TOP_PATH)/$(DATA_VERSION)
DATA_VERSION_SUBPATH?=unimib_shar_activities/$(DATA_VERSION)

CONDA_ENV_PATH:=$(shell conda env list --json | grep ${PROJECT_ENV_NAME} | head -n1 | cut -d'"' -f2)
# Quit early if environment not found
ifneq ($(filter download_raw_dataset build_std_dataset_from_raw,$(MAKECMDGOALS)),$())
ifndef CONDA_ENV_PATH 
$(error CONDA_ENV_PATH not defined)
endif
endif

.PHONY: all

all: $(DATASET_PATH)/sensor_data_per_seq__collapsed.csv

.PHONY: help
help:                                                 				## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/,/' | column -s, -t

# =====
.PHONY: download_raw_dataset
download_raw_dataset: $(DATASET_TOP_PATH)/raw/UniMiB-SHAR.zip		## Download dataset from source repository

$(DATASET_TOP_PATH)/raw/UniMiB-SHAR.zip:
	@{ \
	mkdir -p $(DATASET_TOP_PATH)/raw/; \
	echo "> MANUAL ACTION NEEDED!"; \
	echo "> Manually download ZIP file from this URL:";\
	echo "https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip";\
	echo "> Save the ZIP file here:";\
	echo "$(DATASET_TOP_PATH)/raw/UniMiB-SHAR.zip"; \
	echo "> Then do:";\
	echo "make unzip_raw_dataset";\	
	}

# =====
.PHONY: unzip_raw_dataset
unzip_raw_dataset: $(DATASET_TOP_PATH)/raw/data/full_data.mat		## Unzip raw dataset into .mat files

# Unzip the files, avoiding top level 'UniMiB-SHAR/' directory \

$(DATASET_TOP_PATH)/raw/data/full_data.mat: $(DATASET_TOP_PATH)/raw/UniMiB-SHAR.zip
	@{ \
	echo "> Unzipping the files..."; \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	cd $(DATASET_TOP_PATH)/raw/ \
		&& unzip UniMiB-SHAR.zip \
		&& mv UniMiB-SHAR/* ./ \
		&& rmdir UniMiB-SHAR/ \
		&& echo "> DONE. MAT files exist $(DATASET_TOP_PATH)/raw/data/full_data.mat"; \
	}

# =====
.PHONY: build_std_dataset_from_raw 
build_std_dataset_from_raw: $(DATASET_PATH)/sensor_data_per_tstep.csv 	## Build standardized flat file time-series dataset

$(DATASET_PATH)/sensor_data_per_tstep.csv: $(DATASET_TOP_PATH)/raw/data/adl_data.mat src/make_csv_dataset_from_raw.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p $(DATASET_PATH); \
	python -u ./src/make_csv_dataset_from_raw.py \
		--dataset_path $(DATASET_TOP_PATH)/raw/data/ \
		--output_sensors_ts_csv_path $(DATASET_PATH)/sensor_data_per_tstep.csv \
		--output_metadata_for_each_ts_csv_path $(DATASET_PATH)/metadata_per_seq.csv \
		--output_metadata_for_each_subj_csv_path $(DATASET_PATH)/metadata_per_subj.csv \
		; \
	}

# =====
.PHONY: align_to_grid 
align_to_grid: $(DATASET_PATH)/sensor_data_per_tstep__aligned.csv 		## Build time-series aligned to regular intervals

$(DATASET_PATH)/sensor_data_per_tstep__aligned.csv: 
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/align_to_grid.py \
	    --input_ts_csv_path $(DATASET_PATH)/sensor_data_per_tstep.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/$(DATA_VERSION_SUBPATH)/activity_dict.json \
	    --step_size 1 \
	    --output $(DATASET_PATH)/sensor_data_per_tstep__aligned.csv \
	; \
	}

# =====
.PHONY: normalize_features 
normalize_features: $(DATASET_PATH)/sensor_data_per_tstep__normalized.csv 		## Build time series with normalized feature cols

$(DATASET_PATH)/sensor_data_per_tstep__normalized.csv: $(DATASET_PATH)/sensor_data_per_tstep__aligned.csv $(PROJECT_REPO_DIR)/src/normalize_features.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/normalize_features.py \
	    --input $(DATASET_PATH)/sensor_data_per_tstep__aligned.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/$(DATA_VERSION_SUBPATH)/activity_dict.json \
	    --output $(DATASET_PATH)/sensor_data_per_tstep__normalized.csv \
	; \
	}

# =====
.PHONY: collapse_ts 
collapse_ts: $(DATASET_PATH)/sensor_data_per_seq__collapsed.csv 		## Collapse time-series into fixed-size feature vector

$(DATASET_PATH)/sensor_data_per_seq__collapsed.csv: $(DATASET_PATH)/sensor_data_per_tstep__normalized.csv $(PROJECT_REPO_DIR)/src/feature_transformation.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/feature_transformation.py \
	    --input $(DATASET_PATH)/sensor_data_per_tstep__normalized.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/$(DATA_VERSION_SUBPATH)/activity_dict.json \
	    --output $(DATASET_PATH)/sensor_data_per_seq__collapsed.csv \
		--data_dict_output $(DATASET_PATH)/activity_dict__collapsed.json \
	    --collapse \
	; \
	}


# =====
.PHONY: split_into_train_and_test 
split_into_train_and_test: $(DATASET_PATH)/train.csv 		## Split into train and test

$(DATASET_PATH)/train.csv: $(DATASET_PATH)/sensor_data_per_seq__collapsed.csv $(PROJECT_REPO_DIR)/src/split_dataset.py
	@{ \
	python -u $(PROJECT_REPO_DIR)/src/split_dataset.py \
		--input $(DATASET_PATH)/sensor_data_per_seq__collapsed.csv \
		--data_dict $(DATASET_PATH)/activity_dict__collapsed.json \
		--test_size 0.1 \
		--output_dir $(DATASET_PATH)/ \
	; \
	}
