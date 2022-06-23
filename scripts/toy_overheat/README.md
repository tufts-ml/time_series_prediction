Toy Overheat dataset

# Background

The `toy_overheat` dataset is a simple univariate time series dataset with fictional "temperature" measurements for a device over a multi-day operating period (50-200 hours).

Many such devices are observed (many sequences). For each one, we measure one temperature measurement per hour.  The period of observation changes across sequences (some are longer, some are shorter).

6 example sequences are shown here (left column: y=0 means no overheating, right: y=1 means overheating)

![image](https://user-images.githubusercontent.com/2365444/175380946-27c518b6-630b-436d-9dfa-7a381c97bb1d.png)

The classification goal is to look at temperature over time and answer the binary question "did the device" overheat?
In this context, overheating means a spike in temperature that exceeded ~5 degrees C.

Code to generate multiple sequences and their labels is here:

https://github.com/tufts-ml/time_series_prediction/blob/master/scripts/toy_overheat/standardize_dataset/make_dataset.py

# Prereqs

It's assumed you have the `tspred_env` conda environment installed locally, and that this environment is active

```console
$ conda activate tspred_env
```


# Workflow

You can use the provided "*.smk" files or "Snakefile" files to complete every step of the required pipeline.


### Prepare standardized dataset

Generate a dataset of 600 sequences (1/3 positive, 2/3 negative)

```console
$ cd standardize_dataset/
$ snakemake --cores 1 all
```

**Expected output**

CSV data files and JSON data-dictionary files on disk.

* datasets/toy_overheat/v20200515/features_per_tstep.csv
* datasets/toy_overheat/v20200515/Spec_FeaturesPerTimestep.json

* datasets/toy_overheat/v20200515/outcomes_per_seq.csv
* datasets/toy_overheat/v20200515/Spec_OutcomesPerSequence.csv


### Prepare collapsed features representation


```console
$ cd predictions_collapsed/
$ snakemake --cores 4 -s make_collapsed_dataset_and_split_train_test.smk all

**Expected output**

New CSV files written to disk, which have one feature vector per sequence

* datasets/toy_overheat/v20200515/features_per_seq.csv

Features are named like:

* temperature_mean_0_to_100
* temperature_mean_0_to_50
* ...
* temperature_max_0_to_100
* temperature_max_0_to_50
* ...

Each one named "{name}_{func}_{A}_to_{B}" can be understood to mean "take the time series for variable {name} over the interval from {A} to {B}, and compute the {func} function that summarizes that time series into a single number"

### Train and evaluate LR and RF models

``` console
$ # To be done in predictions_collapsed/ folder`
$ snakemake --cores 4 -s train_logistic_regression.smk all
$ snakemake --cores 4 -s train_random_forest.smk all
```

**Expected output**

HTML reports on binary classifier performance

Point your browser to:

* </tmp/results/toy_overheat/v20200515/split-by=sequence_id/collapsed_features_per_sequence/logistic_regression/report.html>
* </tmp/results/toy_overheat/v20200515/split-by=sequence_id/collapsed_features_per_sequence/random_forest/report.html>

