SHELL:=/bin/bash

# Verify required programs (like 'conda' are on the current path)
# Make will terminate immediately if one of these is not available
REQD_EXECUTABLES = ls grep conda
K := $(foreach exec,$(REQD_EXECUTABLES),\
        $(if $(shell which $(exec)),$(exec),$(error "Required program '$(exec)' not in PATH. Please install or fix path issues.")))

# Default environment variables
# Can override with your setup_env/$USER.sh, or with local env variables
PYTHON_VERSION?=3.6
PROJECT_ENV_NAME?=tspred_env
PROJECT_REPO_DIR?=$(abspath ../)

CONDA_ENV_PATH:=$(shell conda env list --json | grep ${PROJECT_ENV_NAME} | head -n1 | cut -d'"' -f2)

.SILENT: create_env

.PHONY: help
help:                                                 								## Show help messages for each command
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/:.*##/,/' | column -s, -t

#=== Commands

.PHONY: create_env
create_env: ${CONDA_ENV_PATH}/bin/python											## Create new env for this project named `tspred_env`

${CONDA_ENV_PATH}/bin/python:					        
	@{ \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	echo "> PROJECT_REPO_DIR: "${PROJECT_REPO_DIR}; \
	echo "> CONDA_ENV_PATH: "${CONDA_ENV_PATH}; \
	conda create --name ${PROJECT_ENV_NAME} python=${PYTHON_VERSION}; \
	}


.PHONY: install_pkgs_to_env
install_pkgs_to_env: create_env $(PROJECT_REPO_DIR)/requirements.txt			## Install req'd packages in new env 
	@{ \
	echo "> PROJECT_REPO_DIR: "${PROJECT_REPO_DIR}; \
	echo "> CONDA_ENV_PATH: "${CONDA_ENV_PATH}; \
	echo "> Installing/updating all packages in requirements.txt, which are:"; \
	cat $(PROJECT_REPO_DIR)/requirements.txt; \
	source $(PROJECT_REPO_DIR)/setup_env/$(USER).sh; \
	source activate ${PROJECT_ENV_NAME}; \
	conda install --yes --file $(PROJECT_REPO_DIR)/requirements.txt; \
	which wget || conda install wget --yes; \
	}

.PHONY: all
all: create_env install_pkgs_to_env
