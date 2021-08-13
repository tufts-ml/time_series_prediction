ReadMe.txt
Running experiments on USC-HAD dataset
Created by Michael Riley
08/02/2021

Goal: The goal of this ReadMe is to provide instructions and helpful tips for someone looking to re-create the experiments done on the USC Human Activities Dataset. Ideally someone can follow this document, and successfully go from downloading the raw dataset, all the way through getting the html report results. 

------------------------------------------------------------------------------------------
Please refer to the paper at the link provided for more information on the dataset:

http://sipi.usc.edu/had/mi_ubicomp_sagaware12.pdf

There have been three cases of problems done using this toolbox on this dataset thus far. They are:
1) Binary problem: Less intense activities mapped to the outcome 0, more intense activities mapped to 1.
2) Three classes: Activities including walking (Activities 1-5) mapped to the outcome 1, activities requiring more intese movements (activities 6-7) mapped to 2, and activities requiring less intense movements (activities 8-12) mapped to 3.
3) All 12 classes: Classifier ran attempting to successfully indicate all 12 activities provided by the dataset.

For more information on the spec of the dataset, please refer to the Gsheet below:

https://docs.google.com/spreadsheets/d/1UaNvf3TOeeN7UK8F65T8IjLHj6JPGs24XggJ2T8_nTg/edit#gid=0

------------------------------------------------------------------------------------------
Step 1: Download the raw data
Step 2: Download the spec for proper JSON files
Step 3: Preprocess the data using the script provided
Step 4: Collapse features
Step 5: Split the data into training and test sets for experiments
Step 6: Run training & evaluation for logistic regression and/or random forest models
------------------------------------------------------------------------------------------

------------------------------
Step 1: Download the raw data
------------------------------

The raw data can be downloaded at the link below:

http://sipi.usc.edu/had/

-----------------------------------------------
Step 2: Download the spec for proper JSON files
-----------------------------------------------

This is link to the spec we have provided for the USC-HAD dataset (same as Gsheet already linked above):

https://docs.google.com/spreadsheets/d/1UaNvf3TOeeN7UK8F65T8IjLHj6JPGs24XggJ2T8_nTg/edit#gid=0

Downloading the specs:

Assuming you cloned the current repository, you should have the file 'spec_config.json' in the folder scripts > usc_had > standardize_dataset. 

Now, you can run the rules 'download_spec_from_gsheet_as_csv' and 'build_spec_json_from_csv' in that order using the shell commands within 'Snakefile' to get json specs for each of your csv files.

PLEASE NOTE: You will need to move these spec files into your dataset folder before running experiments.

-----------------------------------------------------
Step 3: Preprocess the data using the script provided
-----------------------------------------------------

Observe the file: datasets > usc_had > make_csvs_from_raw_data.py

NOTICE: At the top of the file, the variable RAW_DATA_PATH must be changed to the directory that your raw, downloaded USC-HAD data is stored.

Once this is changed, simply run the script. This will produce 3 files necessary for the experiments:
1) features_per_subject.csv
2) features_per_timestep.csv
3) outcomes_per_sequence.csv

Before running the experiments, you'll want to create a folder with the name 'vYYYYMMDD' to follow the workflow format, and store these 3 files within that folder. This way, you can keep track of different split versions of the dataset, and analyze the results from each. Be sure that your config.json file within the 'scripts' folder has the correct 'DATASET_VERSION' value.

-------------------------
Step 4: Collapse features
-------------------------

Refer to: usc_had > predictions_collapsed > make_collapsed_dataset_and_split_train_test.smk

Here you will pass in your features_per_timestep.csv file, along with its spec in order to get a collapsed features_per_sequence.csv file. You can run the shell commands within the rule 'collapse_features'. Be sure to pass in the path of the 'feature_transformation.py' file within the 'src' folder. 

Or if you'd like, you can combine steps 4 and 5 by running the entire snakefile at once. Please skip down to Step 5 if you will be running these two steps in conjuncture using the snakefile.

------------------------------------------------------------------
Step 5: Split the data into training and test sets for experiments
------------------------------------------------------------------

Refer to: usc_had > predictions_collapsed > make_collapsed_dataset_and_split_train_test.smk

Here you'll split your collapsed features_per_sequence.csv and outcomes_per_sequence.csv files into training and test sets. You can follow the shell commands under the rule 'split_into_train_and_test'. Be sure to pass in the path of the 'split_dataset.py' file within the 'src' folder. 

PLEASE NOTE: Running entire snakefile to complete steps 4 & 5:
- In the 'config.json' file, you should ensure the OUTPUT_COL_NAME is the name of the column you will be predicting: 
	- act_binary for the binary problem
	- act_three_classes for three-class case
	- act_no for all 12 class case

PLEASE NOTE: 
- Pass in the same random state for splitting your collapsed dataset and outcomes (when running shell commands; this can be anything you want).
- test_size should be a decimal representing what percentage of the dataset you want to be the test set. Ex) 0.1 = 10% of the dataset.
- group_cols (w/in make_collapsed_dataset_and_split_train_test.smk) should be the columns you wish to keep constant across train and test sets. For example, if your data consists of multiple subjects, you'd want the subjects in the testing set to be unique from those in the training set. This way, you can simulate giving your model completely unseen data.

-------------------------------------------------------------------------------------
Step 6: Run training & evaluation for logistic regression and/or random forest models
-------------------------------------------------------------------------------------

Refer to: usc_had > predictions_collapsed > train_logistic_regression.smk / train_random_forest.smk

PLEASE NOTE: For the binary problem, we recommend using this scoring formula: roc_auc_score+0.001*cross_entropy_base2_score. For multiple-class problems, we recommend using: cross_entropy_base2_score. Within either of these files, you can switch between these by editting the '--scoring' attribute.

PLEASE NOTE: When running shell commands to get results, you may get a ModuleNotFoundError for 'yattag' in 'eval_classifier.py'. You can simply run 'pip install yattag' in order to get around this. If you are running the all command for this snakemake file, you should not run into any issues. 

Upon running one of the train and evaluate files, an html report will be generated in your /tmp file. These are your final results.



