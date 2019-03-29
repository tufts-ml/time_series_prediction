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
N_SEQS?=50
DATA_VERSION?=v20190314
DATASET_TOP_PATH?=/cluster/tufts/hugheslab/datasets/mimic-iii-v1.4/v20181213/
DATASET_SRC_PATH?=$(DATASET_TOP_PATH)/mimic3benchmarks_csv_by_subj/in_hospital_mortality/
DATASET_OUT_PATH?=$(DATASET_TOP_PATH)/tidy/mimic3benchmarks_inhospital_mortality/$(DATA_VERSION)

CONDA_ENV_PATH:=$(shell conda env list --json | grep ${PROJECT_ENV_NAME} | head -n1 | cut -d'"' -f2)
# Quit early if environment not found
ifneq ($(filter build_std_dataset_from_raw,$(MAKECMDGOALS)),$())
ifndef CONDA_ENV_PATH 
$(error CONDA_ENV_PATH not defined. Follow install in ${PROJECT_REPO_DIR}/setup_env/Makefile )
endif
endif

.PHONY: help
help:                                                 				## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/,/' | column -s, -t

# =====
.PHONY: build_std_dataset_from_raw 
build_std_dataset_from_raw: $(DATASET_OUT_PATH)/vitals_data_per_tstamp.csv 	## Build standardized flat file time-series dataset

$(DATASET_OUT_PATH)/vitals_data_per_tstamp.csv: $(DATASET_SRC_PATH)/train/*.csv src/make_csv_dataset_from_raw.py
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p $(DATASET_OUT_PATH); \
	echo "DATASET_TOP_PATH:\n$(DATASET_TOP_PATH)"; \
	python -u ./src/make_csv_dataset_from_raw.py \
		--dataset_path $(DATASET_SRC_PATH)/ \
		--data_per_subject_path $(DATASET_TOP_PATH)/mimic3benchmarks_csv_by_subj/ \
		--n_sequences_to_read_per_split $(N_SEQS) \
		--output_vitals_ts_csv_path $(DATASET_OUT_PATH)/vitals_data_per_tstamp.csv \
		--output_metadata_per_seq_csv_path $(DATASET_OUT_PATH)/metadata_per_seq.csv \
		--output_metadata_per_subj_csv_path $(DATASET_OUT_PATH)/metadata_per_subj.csv \
		; \
	}

# =====
.PHONY: align_to_grid 
align_to_grid: $(DATASET_OUT_PATH)/vitals_data_per_tstamp__aligned.csv 		## Build time-series aligned to regular intervals

$(DATASET_OUT_PATH)/vitals_data_per_tstamp__aligned.csv: 
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	python -u $(PROJECT_REPO_DIR)/src/align_to_grid.py \
	    --input_ts_csv_path $(DATASET_OUT_PATH)/vitals_data_per_tstamp.csv \
	    --data_dict $(PROJECT_REPO_DIR)/docs/$(DATA_VERSION)/mimic_dict.json \
	    --step_size 1 \
	    --output $(DATASET_PATH)/vitals_data_per_tstamp__aligned.csv \
	; \
	}