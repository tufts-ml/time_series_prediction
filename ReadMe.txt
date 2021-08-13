ReadMe.txt
Running experiments on new dataset
Created by Michael Riley
07/08/2021

Goal: The goal of this ReadMe is to provide instructions and helpful tips for someone looking to conduct experiments similar to those already in this toolbox, on a different dataset. Ideally, this helps from the step of downloading a new, raw dataset, all the way through until collecting and analyzing results of the experiments.

------------------------------------------------------------------------------------------
Step 1: Find a dataset and download the raw data
Step 2: Create a spec for the data, and download the json files from this spec
Step 3: Preprocess the data, organizing it into the format that you desire
Step 4: Collapse your datasets features
Step 5: Split the data into training and test sets for experiments
Step 6: Run the experiments, and analyze results
------------------------------------------------------------------------------------------

-------------------------------------------------
Step 1: Find a dataset and download the raw data
-------------------------------------------------

This toolbox is based on the MIMIC Extract patient dataset. This is a time-series dataset of patient interventions with length-of-stay and mortality outcomes. For the purpose of testing models, we recommend finding public datasets using human activities to predict some outcomes, whether they be binary or not. A good dataset should have multiple subjects, each with many sequences of data mapped to a particular outcome. This way your model can be trained to classify sequences, and tested on heldout sets of users it has not yet seen. This will be good for simulating your model generalizing to completely new, unseen data.

------------------------------------------------------------------------------
Step 2: Create a spec for the data, and download the json files from this spec
------------------------------------------------------------------------------

You'll need the dataset to be converted to a format that is interpretable by the current workflow. Most likely, the dataset will consist of multiple subjects, with multiple sequences of data for each of them. We recommend organizing 3 files in CSV format:

1) Features Per Subject - Organize with one row per subject id, providing features for each subject.

2) Features Per Timestep - One row per timestep, subject id, and sequence id depending on the structure of the data. Provide features/measurements at every timestep.

3) Outcomes Per Sequence - One row per sequence id, and subject id / trial number depending on the structure of the data. Provide the outcome of each sequence.

We recommend creating a google sheet (Gsheet) to organize the desired structure of your dataset. For reference, see the Gsheet from the UnimibShar activities dataset: 

https://docs.google.com/spreadsheets/d/1O_JW82xxjhFf6UVOIJJj7HyXs-acSD-VARVMoLRl5c8/edit?pli=1#gid=1839800383

PLEASE NOTE: You must go into your Gsheet's settings and make it available to all users with the url in order to successfully use the workflow to download your specs.

Downloading your JSON specs:

Refer to the folder: scripts > unimib_shar_activities > standardize_dataset

First, create a 'spec_config.json' file for your dataset. Observe the structure of the unimib_shar_activities spec_config.json. The only items you should need to change from this file is the "raw_data_url","spec_gsheet_key", and "spec_gid_list". The spec_gsheet_key will be the portion of your Gsheets url that corresponds to the spec_gsheet_key part of the url pattern. Also, notice in the url for your Gsheet, when you switch between the tabs of the sheet, the url 'gid' number will change. Organize your "spec_gid_list" be the gid numbers of each tab, in the order of your "spec_sheet_name_list". 

Now, you can run the rules 'download_spec_from_gsheet_as_csv' and 'build_spec_json_from_csv' in that order using the shell commands within 'Snakefile' to get json specs for each of your csv files.

PLEASE NOTE: You will need to change the variable "DATASET_SCRIPTS_ROOT" in order to point to your new dataset. If you are going off of the example Snakefile within the UnimibShar Activities dataset, you should change the string 'unimib_shar_activities' to the name of your dataset's folder within the 'scripts' folder.

If you have a custom script to preprocess your data into these three files, you can store it within your standardize_dataset folder, and alter the input of the rule 'build_csv_dataset' within the 'Snakefile'. Then, you could run this file in order to perform steps 2 and 3 simultaneously.

PLEASE NOTE: You will need to move these spec files into your dataset folder before running experiments.

--------------------------------------------------------------------------
Step 3: Preprocess the data, organizing it into the format that you desire
--------------------------------------------------------------------------
We recommend generating a script to parse the raw data and output these three files. The general structure of a script like this can be found in scripts > unimib_shar_activities > standardize_dataset > make_csv_dataset_from_raw.py.

For easier reproducability, you can point your altered 'Snakefile' to your custom script for preprocessing the data as mentioned above. Otherwise, you can run your script individually.

---------------------------------------
Step 4: Collapse your datasets features
---------------------------------------
Refer to: unimib_shar_activities > predictions_collapsed > make_collapsed_dataset_and_split_train_test.smk

Here you will pass in your features_per_timestep.csv file, along with its spec in order to get a collapsed features_per_sequence.csv file. You can run the shell commands within the rule 'collapse_features'. Be sure to pass in the path of the 'feature_transformation.py' file within the 'src' folder as the input script. 

Or if you'd like, you can combine steps 4 and 5 by running the entire snakefile at once. Please skip down to Step 5 if you will be running these two steps in conjuncture using the snakefile.

------------------------------------------------------------------
Step 5: Split the data into training and test sets for experiments
------------------------------------------------------------------
Refer to: unimib_shar_activities > predictions_collapsed > make_collapsed_dataset_and_split_train_test.smk

Here you'll split your collapsed features_per_sequence.csv and outcomes_per_sequence.csv files into training and test sets. You can follow the shell commands under the rule 'split_into_train_and_test'. Be sure to pass in the path of the 'split_dataset.py' file within the 'src' folder as the input script. 

PLEASE NOTE: When running entire snakefile to complete steps 4 & 5:
- You will need a file 'config.json', of the structure like scripts > unimib_shar_activities > config.json. Within this file, you should ensure the OUTPUT_COL_NAME is the name of the column you will be predicting. 
- You will need a file 'config_loader.py' in your predictions_collapsed folder, of the structure like scripts > unimib_shar_activities > predictions_collapsed > config_loader.py. You should only need to change the name of the dataset within the DATASET_SCRIPTS_ROOT variable.

PLEASE NOTE: When running via shell commands: 
- Pass in the same random state for splitting your collapsed dataset, and outcomes file.
- test_size should be a decimal representing what percentage of the dataset you want to be the test set. Ex) 0.1 = 10% of the dataset.
- group_cols (w/in make_collapsed_dataset_and_split_train_test.smk) should be the columns you wish to keep constant across train and test sets. For example, if your data consists of multiple subjects, you'd want the subjects in the testing set to be unique from those in the training set. This way, you can simulate giving your model completely unseen data.

------------------------------------------------
Step 6: Run the experiments, and analyze results
------------------------------------------------

Refer to: unimib_shar_activities > predictions_collapsed > train_logistic_regression.smk / train_random_forest.smk

PLEASE NOTE: For binary problems, we recommend using this scoring formula: roc_auc_score+0.001*cross_entropy_base2_score. For multiple-class problems, we recommend using: cross_entropy_base2_score. Within either of these files, you can switch between these by editting the '--scoring' attribute.

PLEASE NOTE: When running shell commands to get results, you may get a ModuleNotFoundError for 'yattag' in 'eval_classifier.py'. You can simply run 'pip install yattag' in order to get around this. If you are running the all command for this snakemake file, you should not run into any issues.

Upon running one of the train and evaluate files, an html report will be generated in your /tmp file. These are your final results. 






