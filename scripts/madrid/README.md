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

## C) Creating Collapsing Representations for Training Shallow Classifiers
---------------------------------------------------------------------------------------------------------------------

### 1) Collapsing features in multiple patient stay slices, with their corresponding outputs, dynamically
---------------------------------------------------------------------------
```
$ cd predictions_collapsed
$ snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_and_split_train_test.smk make_collapsed_features_for_dynamic_output_prediction
```

### 2) Merge all the dynamic collapsed features and static features into a single features table
----------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_and_split_train_test.smk merge_dynamic_collapsed_features
```

### 3) Split the features table into train\valid\test based on the year of admission. Trian on first 3 years of admission. Vlaidate and test on the 4th and 5th years of admission
-------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores 1 --snakefile make_collapsed_dataset_dynamic_input_output_and_split_train_test.smk split_into_train_and_test
```

## D) Training Classifiers Perform prediction of clinical deterioration with shallow models (LR, RF, MLP, SVM, lightGBM) 
---------------------------------------------------------------------------------------------------------------------
```
$ cd predictions_collapsed
```

### Predict with Logistic Regression
----------------------------------------
```
$ snakemake --cores 1 --snakefile skorch_logistic_regression_dynamic.smk
```

### Predict with Random Forest
----------------------------------------
```
$ snakemake --cores 1 --snakefile random_forest_dynamic.smk
```

### Predict with MLP
----------------------------------------
```
$ snakemake --cores 1 --snakefile skorch_mlp_dynamic.smk
```

### Predict with Linear SVM
----------------------------------------
```
$ snakemake --cores 1 --snakefile SVC_dynamic.smk
```
### Predict with lightGBM
----------------------------------------
```
$ snakemake --cores 1 --snakefile lightGBM_dynamic.smk
```

## E) Evaluate and Visualize Performance
---------------------------------------------

### Choose hyperparameter, plot precision-recall curves, alarm m distributions etc.
----------------------------------------
```
$ cd predictions_collapsed
$ snakemake --cores 1 --snakefile evaluate_dynamic_prediction_performance.smk
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
### 5) Train with RNN with hyperparams specified in rnn.json
----------------------------------------------------------------------------------------------------------------
```
$ snakemake --cores all --snakefile rnn.smk train_and_evaluate_classifier_many_hyperparams
```