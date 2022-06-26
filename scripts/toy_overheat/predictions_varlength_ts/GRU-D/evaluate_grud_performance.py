import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from scipy.stats import multivariate_normal as mvn
import seaborn as sns
import sys

PROJECT_REPO_DIR=os.path.abspath(os.path.join(__file__, '../../../../../'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import get_fenceposts, parse_id_cols, parse_time_cols, parse_feature_cols
from utils import load_data_dict_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fits_dir', type=str, required=True)
    args = parser.parse_args()
        
    #go through all the saved loss plots
    final_perf_df_list = []
    for missing_handling in ['no_imp']:
        for perc_obs in ['20', '40', '60', '80', '100']:
            all_fits_csvs = glob.glob(os.path.join(args.fits_dir, "final_perf_GRUD-missing_handling=%s-perc_obs=%s*"%(missing_handling, perc_obs)))
            losses_per_fit_list = []
            auc_per_fit_list = []
            for fit_csv in all_fits_csvs:
                fit_df = pd.read_csv(fit_csv)
                auc_per_fit_list.append(fit_df['valid_AUC'].values[-1])

            best_fit_ind = np.argmax(auc_per_fit_list)
            best_fit_csv = all_fits_csvs[best_fit_ind]
            best_fit_auc = auc_per_fit_list[best_fit_ind]

            print(best_fit_csv)
            fit_df = pd.read_csv(best_fit_csv)

            final_perf_csv = os.path.join(args.fits_dir, best_fit_csv)
            final_perf_df = pd.read_csv(final_perf_csv)
            final_perf_df['model'] = 'GRU-D'
            final_perf_df['perc_obs'] = perc_obs
            print('--------------------------------------')
            print('perc observed = %s'%(perc_obs))
            print(final_perf_df)
            print('--------------------------------------')
            final_perf_df_list.append(final_perf_df)
    
    final_perf_df = pd.concat(final_perf_df_list).reset_index(drop=True)
    final_perf_df.to_csv('GRUD_performance_toy_data.csv', index=False)
    
        
