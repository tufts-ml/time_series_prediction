Madrid dataset

# Prereqs

### Local Snakemake Installation
It's assumed you have a local install of Conda and Snakemake. It is also assumed that you have access to the Madrid Cluster.

### Pre-processed EHR Data
For instructions on pre-processing the raw data on the madrid cluster to get the demographics, vitals, labs and outcomes of the filtered cohort please refer to : [madrid-data-prep](https://github.com/tufts-ml/madrid-data-prep/tree/fix_preproc)


# Workflow

## A) SSH into Madrid VM
---------------------------
```
$ ssh to madrid vm
```

## B) Report Summary Statistics of Madrid EHR data
------------------------------------------------------------------------
```
$ cd summary_statistics
$ snakemake --cores 1 --snakefile report_summary_statisitics.smk compute_summary_statistics
```

## C) Prediction of clinical deterioration with collapsed features using shallow models
---------------------------------------------------------------------------------------------------------------------

### As a part of the prediction of clinical deterioration workflow, we make collapsed representation of EHR data for multiple patient-stay-slices for prediction with shallow models (LR, RF, MEWS)

### 1) Filter admissions by patient-stay-slice (For eg, first 4 hrs, last 4hrs, first 30%)
---------------------------------------------------------------------------
Note : For eg. If we predict using first 4 hours of data, we ensure that every patient in the cohort has atleast 4 hours of data.
```
$ cd predictions_collapsed
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk filter_admissions_by_tslice_many_tslices
```
### 2) Collapsing features and saving to slice specific folders
----------------------------------------------------------------
```
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk collapse_features_many_tslices
```
### 3) Computing slice specific MEWS scores
--------------------------------------------
```
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk compute_mews_score_many_tslices
```

### 4) Merge all the collapsed features across tslices into a single features table
----------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk merge_collapsed_features_all_tslices
```

### 5) Split the features table into train - test. A single classifier will be trained on this training fold
-------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk split_into_train_and_test
```

### 6) Do every step above in squence
-------------------------------------------
```
$ snakemake --cores all --snakefile make_collapsed_dataset_per_tslice_and_split_train_test.smk all
```

### 7) Perform prediction of clinical deterioration with shallow models (LR, RF, MEWS) on multiple patient-stay-slices
---------------------------------------------------------------------------------------------------------------------------------
```
$ cd predictions_collapsed
```

### 8) Predict with Logistic Regression
----------------------------------------
```
$ snakemake --cores 1 --snakefile logistic_regression.smk
```
### 9) Predict with Random Forest
----------------------------------------
```
$ snakemake --cores 1 --snakefile random_forest.smk
```

### 10) Predict with MEWS
-------------------------------
```
$ snakemake --cores 1 --snakefile eval_mews_score.smk
```

### 11) Visualize prediction performance
---------------------------------------------
```
$ cd predictions_collapsed
```
#### 11a) Plotting performance metrics such as AUROC, average precision, balanced accuracy on patient-stay-slice cohorts  
-------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile evaluate_classifier_pertslice_performance.smk evaluate_performance
```

#### 11b) Plotting probability of deterioration over time for individual patient-stays
--------------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile evaluate_proba_deterioration_over_time.smk evaluate_proba_deterioration
```

## D) Prediction of clinical deterioration with full sequences using RNN's

### 1) Filter and save multiple patient-stay-slices (0-8h, 0-16h, first 30%, last 5 hours) for evaluation
----------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores all --snakefile make_features_and_outcomes_and_split_train_test.smk filter_admissions_by_tslice_many_tslices
```

### 2) Make a single set of features containing from all patient-stays and outcomes for each patient stay
----------------------------------------------------------------------------------------------------------
```
$  snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk make_features_and_outcomes
```
### 3) Split into train and test sets containing sequences
--------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk split_into_train_and_test
```
### 4) Impute missing values for train and test sets seaparately by carry forward and population mean
----------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_features_and_outcomes_and_split_train_test.smk impute_missing_values
```
### 5) Predict with RNN with hyperparams specified in rnn.json
----------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores all --snakefile rnn.smk train_and_evaluate_classifier_many_hyperparams
```