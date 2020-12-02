'''
Train PC-HMM ON Toy Overheat Binary Classification

Usage
-----
snakemake --cores all --snakefile train_pchmm.smk train_and_evaluate_classifier_many_hyperparams

'''

# Default environment variables
# Can override with local env variables
configfile:"pchmm.json"

TOY_OVERHEAT_VERSION = os.environ.get('TOY_OVERHEAT_VERSION', 'v20200515')
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

RESULTS_FEAT_PER_TSTEP_PATH = "/tmp/results/toy_overheat/pchmm/"

DATASET_STD_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION)
DATASET_SPLIT_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION, 'train_test_split_dir')
print("Results will be saved in : %s"%RESULTS_FEAT_PER_TSTEP_PATH)

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"pchmm-lr={lr}-seed={seed}-batch_size={batch_size}.csv").format(lr=lr, seed=seed, batch_size=batch_size) for lr in config['lr'] for seed in config['seed'] for batch_size in config['batch_size']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'PC-VAE', 'main.py'),
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test.csv'),
        x_dict_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict.json'),
        y_dict_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="pchmm-lr={lr}-seed={seed}-batch_size={batch_size}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "pchmm-lr={lr}-seed={seed}-batch_size={batch_size}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name did_overheat_binary_label \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --output_filename_prefix {params.fn_prefix} \
        '''
