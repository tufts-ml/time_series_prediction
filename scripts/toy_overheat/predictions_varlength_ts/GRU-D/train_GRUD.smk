'''
Train PC-HMM ON Toy Overheat Binary Classification

Usage
-----
snakemake --cores 1 --snakefile train_GRUD.smk train_and_evaluate_classifier_many_hyperparams

snakemake --snakefile train_GRUD.smk --profile ../../../../profiles/hugheslab_cluster/ 

'''

# Default environment variables
# Can override with local env variables
configfile:"GRUD.json"

TOY_OVERHEAT_VERSION = os.environ.get('TOY_OVERHEAT_VERSION', 'v20200515')
PROJECT_REPO_DIR = os.environ.get("PROJECT_REPO_DIR", os.path.abspath("../../../../"))
PROJECT_CONDA_ENV_YAML = os.path.join(PROJECT_REPO_DIR, "ts_pred.yml")

RESULTS_FEAT_PER_TSTEP_PATH = "/cluster/tufts/hugheslab/prath01/results/toy_overheat/GRUD/"

DATASET_STD_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION)
DATASET_SPLIT_PATH = os.path.join(PROJECT_REPO_DIR, 'datasets', 'toy_overheat', TOY_OVERHEAT_VERSION, 'train_test_split_dir')
print("Results will be saved in : %s"%RESULTS_FEAT_PER_TSTEP_PATH)

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"final_perf_GRUD-missing_handling={missing_handling}-perc_obs={perc_obs}-lr={lr}-dropout={dropout}-l2_penalty={l2_penalty}-seed={seed}.csv").format(lr=lr, seed=seed, perc_obs=perc_obs, dropout=dropout, l2_penalty=l2_penalty, missing_handling=missing_handling) for missing_handling in config['missing_handling'] for perc_obs in config['perc_obs'] for lr in config['lr'] for seed in config['seed'] for dropout in config['dropout'] for l2_penalty in config['l2_penalty']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'src', 'GRU_D', 'main_toy.py'),
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'x_valid_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'y_valid_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_dict_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict_{missing_handling}_observed={perc_obs}_perc.json'),
        y_dict_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict_{missing_handling}_observed={perc_obs}_perc.json')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="GRUD-missing_handling={missing_handling}-perc_obs={perc_obs}-lr={lr}-dropout={dropout}-l2_penalty={l2_penalty}-seed={seed}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "final_perf_GRUD-missing_handling={missing_handling}-perc_obs={perc_obs}-lr={lr}-dropout={dropout}-l2_penalty={l2_penalty}-seed={seed}.csv")
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --outcome_col_name did_overheat_binary_label \
            --output_dir {params.output_dir} \
            --train_csv_files {input.x_train_csv},{input.y_train_csv} \
            --valid_csv_files {input.x_valid_csv},{input.y_valid_csv} \
            --test_csv_files {input.x_test_csv},{input.y_test_csv} \
            --data_dict_files {input.x_dict_json},{input.y_dict_json} \
            --validation_size 0.15 \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --output_filename_prefix {params.fn_prefix} \
            --dropout {wildcards.dropout} \
            --l2_penalty {wildcards.l2_penalty} \
        '''
