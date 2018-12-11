SHELL:=/bin/bash

EEG_VERSION=v20181211

# Default settings
# Can override with your setup_env/$USER.sh, or with local env variables
PROJECT_ENV_NAME?=tspred_env
PROJECT_REPO_DIR?=../../

CONDA_EXECUTABLE:=$(shell which conda)
CONDA_ENV_EXISTS:=$(shell conda env list | grep -c ${PROJECT_ENV_NAME})

.PHONY: help
help:                                                 								## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/,/' | column -s, -t

#=== Commands

setup_env_vars: $(PROJECT_REPO_DIR)/setup_env/$(USER).sh 							## Define user-specific env vars 
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh

.PHONY: create_env
create_env: setup_env_vars has_conda_exe									        ## Create new env for this project
	{ \
	echo ">>> "${PROJECT_REPO_DIR}; \
	conda create --name ${PROJECT_ENV_NAME} python=2.7; \
	}

.PHONY: install_pkgs_to_env
install_pkgs_to_env: has_conda_env $(PROJECT_REPO_DIR)/requirements.txt				## Install req'd packages in new env 
	{ \
	source activate ${PROJECT_ENV_NAME}; \
	conda install --yes --file $(PROJECT_REPO_DIR)/requirements.txt; \
	which wget || conda install wget --yes; \
	}

.PHONY: download_dataset
download_dataset: raw/data.csv 										## Download dataset from UCI repository


raw/data.csv: setup_env_vars has_conda_env
	{ \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p raw/; \
	wget https://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv --directory-prefix raw/; \
	}

.PHONY: build_std_from_raw 
build_std_from_raw: setup_env_vars has_conda_env download_dataset					## Build standardized flat files
	{ \
	source activate ${PROJECT_ENV_NAME}; \
	mkdir -p ${EEG_VERSION}; \
	python -u src/build_std_eeg_dataset_from_raw.py \
		--output_csv_path ${EEG_VERSION}/eeg_signal_data.csv \
	; \
	}



#=== Environment Variable Checks

.PHONY: has_conda_env
has_conda_env: has_conda_exe
ifndef CONDA_ENV_EXISTS
	$(error "Error: Conda env '${PROJECT_ENV_NAME}' does not exist. Do this: 'make create_env'")
endif

.PHONY: has_conda_exe
has_conda_exe: setup_env_vars
ifndef CONDA_EXECUTABLE
	$(error "Error: 'conda' is undefined. Please install/add to current path via instructions here:\nhttps://conda.io/docs/user-guide/install/index.html")
endif
