MIMIC-III Benchmark: In-Hospital Mortality dataset

# Prereqs

It's assumed you have the `tspred_env` conda environment installed locally.

For detailed instructions, see `setup_env/README.md`.

# Workflow

You can use the given 'Makefile' to complete every step of the required pipeline. The associated commands are:

```
help                         Show help messages for each command
build_std_dataset_from_raw   Build standardized flat file time-series dataset
                             * set N_SEQS=n to extract only n sequences, N_SEQS=-1 to extract all
                             * set env var DATA_VERSION=2019MMDD to define the prefix where results are saved
```
