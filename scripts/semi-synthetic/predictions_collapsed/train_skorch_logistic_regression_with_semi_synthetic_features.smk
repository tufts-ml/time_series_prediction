'''
Train logistic regression on collapsed features for seni-synthetic features (MIMIC Collapsed + synthetic)

Usage
-----
snakemake --cores 1 --snakefile train_skorch_logistic_regression_with_semi_synthetic_features.smk train_and_evaluate_classifier_many_hyperparams

snakemake --snakefile train_skorch_logistic_regression_with_semi_synthetic_features.smk --profile ../../../profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

'''

configfile : "skorch_logistic_regression_semi_synthetic_bce.json"

PROJECT_REPO_DIR = "/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/"
SEMI_SYNTHETIC_DATA_PATH = "/cluster/tufts/hugheslab/prath01/projects/time_series_prediction/datasets/semi_synthetic_precision_recall/"

RESULTS_SEMI_SYNTHETIC_PATH="/cluster/tufts/hugheslab/prath01/results/semi-synthetic/skorch_logistic_regression"

print("Training logistic regression")
print("--------------------------")
print("Results and trained model will go to:")
print(RESULTS_SEMI_SYNTHETIC_PATH)

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_SEMI_SYNTHETIC_PATH, "skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-warm_start={warm_start}-incremental_min_precision={incremental_min_precision}-initialization_gain={initialization_gain}.json").format(lr=lr, weight_decay=weight_decay, batch_size=batch_size, scoring=scoring, seed=seed, lamb=lamb, warm_start=warm_start, incremental_min_precision=incremental_min_precision, initialization_gain=initialization_gain) for lr in config['lr'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for scoring in config['scoring'] for seed in config['seed'] for lamb in config['lamb'] for warm_start in config['warm_start'] for incremental_min_precision in config['incremental_min_precision'] for initialization_gain in config['initialization_gain']]


rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchLogisticRegression', 'main_synthetic.py'),
        x_train_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'x_train.csv'),
        y_train_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'y_train.csv'),
        x_valid_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'x_valid.csv'),
        y_valid_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'y_valid.csv'),
        x_test_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'x_test.csv'),
        y_test_csv=os.path.join(SEMI_SYNTHETIC_DATA_PATH, 'y_test.csv'),
        

    params:
        output_dir=RESULTS_SEMI_SYNTHETIC_PATH,
        fn_prefix="skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-warm_start={warm_start}-incremental_min_precision={incremental_min_precision}-initialization_gain={initialization_gain}"

    output:
        os.path.join(RESULTS_SEMI_SYNTHETIC_PATH, "skorch_logistic_regression_lr={lr}-weight_decay={weight_decay}-batch_size={batch_size}-scoring={scoring}-seed={seed}-lamb={lamb}-warm_start={warm_start}-incremental_min_precision={incremental_min_precision}-initialization_gain={initialization_gain}.json")


    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name "synthetic_outcome" \
            --output_dir {params.output_dir} \
            --output_filename_prefix {params.fn_prefix} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --scoring {wildcards.scoring} \
            --lr {wildcards.lr} \
            --weight_decay {wildcards.weight_decay} \
            --batch_size {wildcards.batch_size} \
            --seed {wildcards.seed} \
            --lamb {wildcards.lamb} \
            --warm_start {wildcards.warm_start} \
            --incremental_min_precision {wildcards.incremental_min_precision}\
            --initialization_gain {wildcards.initialization_gain}\
        '''