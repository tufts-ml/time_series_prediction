import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
import sys

PROJECT_REPO_DIR=os.path.abspath(os.path.join(__file__, '../../../../'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
from feature_transformation import get_fenceposts, parse_id_cols, parse_time_cols, parse_feature_cols
from utils import load_data_dict_json
import json
import tensorflow.keras as keras

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evalution of cnn fits')
    parser.add_argument('--fits_dir', type=str, required=True)
    args = parser.parse_args()
        
    #go through all the saved loss plots and choose the best one

    
    all_fits_csvs = glob.glob(os.path.join(args.fits_dir, "*conv_layers*.csv"))
    
    from IPython import embed; embed()
    losses_per_fit_list = []
    auc_per_fit_list = []
    for fit_csv in all_fits_csvs:
        fit_dict_list = pd.read_csv(fit_csv)
        losses_last_n_epochs = np.asarray(fit_dict_list['val_loss'])[-1] 
        auroc_last_n_epochs = np.asarray(fit_dict_list['val_auc'])[-1] 
        losses_per_fit_list.append(losses_last_n_epochs)
        auc_per_fit_list.append(auroc_last_n_epochs)


    #     best_fit_ind = np.argmin(losses_per_fit_list)
    best_fit_ind = np.argmax(auc_per_fit_list)
    best_fit_csv = all_fits_csvs[best_fit_ind]
    best_fit_df = pd.read_csv(best_fit_csv)

    test_auc = best_fit_df['test_auc'][0]
    print('Best CNN fit : %s'%(best_fit_csv))
    print('Best CNN AUC on test set : %.2f'%(test_auc))
    
    
    
    
 