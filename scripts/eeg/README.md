EEG dataset

# Prereqs

It's assumed you have a local install of Conda and Snakemake

# Workflow

You can use the given 'Snakefile' to complete every step of the required pipeline.

```
$ snakemake --use-conda --cores 1 all
```

The possible commands are:

```
download_raw_dataset         Download dataset from UCI repository
build_std_dataset_from_raw   Build standardized flat file time-series dataset
build_spec_json_from_csv     Build standardized JSON data dictonary
```
