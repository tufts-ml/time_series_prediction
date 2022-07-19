  
'''
Get the performance plots of a single trained classifier for clinical deterioration at multiple patient-stay-slices
Usage : Evaluate the performance of different shallow classifiers (LR, RF) on multiple patient-stay-slices 
----------------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_skorch_mlp_pertslice_performance.smk evaluate_performance
'''

from config_loader import (
    D_CONFIG,
    DATASET_STD_PATH, DATASET_SPLIT_PATH,
    PROJECT_REPO_DIR, PROJECT_CONDA_ENV_YAML, 
    DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
    RESULTS_FEAT_PER_TSLICE_PATH, 
    DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH,
    RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH)

evaluate_tslice_hours_list=D_CONFIG['EVALUATE_TIMESLICE_LIST']
random_seed_list=D_CONFIG['CLF_RANDOM_SEED_LIST']
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH, 'classifier_train_test_split_dir')
RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH="/cluster/tufts/hugheslab/prath01/results/mimic3/skorch_mlp"


rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_skorch_mlp_pertslice_performance.py")

    params:
        clf_models_dir=RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        evaluation_tslices=evaluate_tslice_hours_list,
        collapsed_tslice_folder=DATASET_SPLIT_COLLAPSED_FEAT_PER_TSLICE_PATH,
        tslice_folder=DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
        preproc_data_dir=DATASET_SPLIT_PATH,
        random_seed_list=random_seed_list,
        output_dir=os.path.join(RESULTS_COLLAPSED_FEAT_PER_SEQUENCE_PATH, 'classifier_per_tslice_performance')
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --tslice_folder {params.tslice_folder}\
        --collapsed_tslice_folder {params.collapsed_tslice_folder}\
        --evaluation_tslices "{params.evaluation_tslices}"\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --random_seed_list "{params.random_seed_list}"\
        --output_dir {params.output_dir}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])