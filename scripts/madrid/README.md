EEG dataset

# Prereqs

### Local Snakemake Installation
It's assumed you have a local install of Conda and Snakemake. It is also assumed that you have access to the Madrid Cluster.

### Pre-processed EHR Data
For instructions on pre-processing the raw data on the madrid cluster to get the demographics, vitals, labs and outcomes of the filtered cohort please refer to : [madrid-data-prep](https://github.com/tufts-ml/madrid-data-prep/tree/fix_preproc)


# Workflow

### SSH into Madrid VM
```
$ ssh to madrid vm
```

## Report Summary Statistics of Madrid EHR data
```
$ cd summary_statistics
$ snakemake --cores 1 --snakefile report_summary_statisitics.smk compute_summary_statistics
```

## Prediction with Collapsed Features using Shallow Models

### Make collapsed representation of EHR data for multiple patient-stay-slices for prediction with shallow models (LR, RF, MEWS)
```
$ cd predictions_collapsed

----------------------------------------------------------------------------------------------------------------------------------------
COLLAPSING FEATURES IN EACH TSLICE
----------------------------------------------------------------------------------------------------------------------------------------

Filter admissions by patient-stay-slice (For eg, first 4 hrs, last 4hrs, first 30%)
---------------------------------------------------------------------------
Note : For eg. If we predict using first 4 hours of data, we ensure that every patient in the cohort has atleast 4 hours of data.

$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk filter_admissions_by_tslice_many_tslices

Collapsing features and saving to slice specific folders
----------------------------------------------------------------
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk collapse_features_many_tslices

Computing slice specific MEWS scores
--------------------------------------------
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk compute_mews_score_many_tslices

---------------------------------------------------------------------------------------------------------------------------------------
MERGE ALL TSLICES AND SPLITTING INTO TRAIN AND TEST
---------------------------------------------------------------------------------------------------------------------------------------

Merge all the collapsed features across tslices into a single features table
----------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk merge_collapsed_features_all_tslices

Split the features table into train - test. A single classifier will be trained on this training fold
-------------------------------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk split_into_train_and_test

Do every step above in squence
-------------------------------------
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk all
```

### Perform prediction of clinical deterioration with shallow models (LR, RF, MEWS) on multiple patient-stay-slices
```
$ cd predictions_collapsed

------------------------------------------------------------------------------------------------------------------------------------------
LOGISTIC REGRESSION
----------------------------------------------------------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile logistic_regression.smk

------------------------------------------------------------------------------------------------------------------------------------------
RANDOM FOREST
----------------------------------------------------------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile random_forest.smk

------------------------------------------------------------------------------------------------------------------------------------------
MEWS
----------------------------------------------------------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile eval_mews_score.smk

```

### Visualize prediction performance
```
$ cd predictions_collapsed

Plotting performance metrics such as AUROC, average precision, balanced accuracy on patient-stay-slice cohorts  
-------------------------------------------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile evaluate_classifier_pertslice_performance.smk evaluate_performance


Plotting probability of deterioration over time for individual patient-stays
----------------------------------------------------------------------------
$ snakemake --cores 1 --snakefile evaluate_proba_deterioration_over_time.smk evaluate_proba_deterioration

```

## Prediction with Full Sequences Using Time Series Models

### Prepare features containing full sequences and a single outcome label
```
cd predictions_per_sequence

$ snakemake --cores all --snakefile make_features_and_outcomes_and_split_train_test.smk
```

### Predict with RNN
```
$ snakemake --cores 1 --snakefile rnn.smk
```