  
'''
Compare BCE , BCE+Threshold search, Surrogate Loss for LR
----------------------------------------------------------------------------------------------------------------------------
>> snakemake --cores 1 --snakefile evaluate_skorch_lr_surrogate_loss.smk evaluate_performance
'''

SEMI_SYNTHETIC_DATA_PATH = "/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/datasets/semi_synthetic_precision_recall/"

RESULTS_SEMI_SYNTHETIC_PATH="/cluster/tufts/hugheslab/prath01/results/semi-synthetic/skorch_logistic_regression"


rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_skorch_lr_surrogate_loss.py")

    params:
        clf_models_dir=RESULTS_SEMI_SYNTHETIC_PATH,
        clf_train_test_split_dir=SEMI_SYNTHETIC_DATA_PATH,
        
    shell:
        '''
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --outcome_column_name "synthetic_outcome"\
        '''