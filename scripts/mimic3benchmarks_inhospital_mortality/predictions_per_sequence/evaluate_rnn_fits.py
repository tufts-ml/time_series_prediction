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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rnn fits evaluation mimic')
    parser.add_argument('--fits_dir', type=str, required=True)
    args = parser.parse_args()
    
    #go through all the saved loss plots and choose the best one
    all_fits_jsons = glob.glob(os.path.join(args.fits_dir, "*_auprc_best.json"))
    losses_per_fit_list = []
    auprc_per_fit_list = []
    for fit_json in all_fits_jsons:
        fit_dict_list = json.load(open(fit_json))
        if 'auprc_valid' in fit_dict_list[0].keys():
            losses_last_n_epochs = np.median([i['valid_loss'] for i in fit_dict_list[-5:]])
            auprc_last_n_epochs = np.median([i['auprc_valid'] for i in fit_dict_list[-5:]])
            losses_per_fit_list.append(losses_last_n_epochs)
            auprc_per_fit_list.append(auprc_last_n_epochs)


    #     best_fit_ind = np.argmin(losses_per_fit_list)
    best_fit_ind = np.argmax(auprc_per_fit_list)
    best_fit_json = all_fits_jsons[best_fit_ind]
    best_fit_csv = best_fit_json.replace("_auprc_best.json", ".csv")
    best_fit_df = pd.read_csv(best_fit_csv)
    test_auprc = best_fit_df['test_auprc'][0]
    
    best_fit_df = pd.DataFrame(json.load(open(all_fits_jsons[best_fit_ind])))
    train_auprc = best_fit_df['auprc_train'].values[-1] 
    valid_auprc = best_fit_df['auprc_valid'].values[-1] 
    
    print('Best RNN fit : %s'%(best_fit_csv))
    print('Best RNN AUPRC on train set : %.2f'%(train_auprc))
    print('Best RNN AUPRC on valid set : %.2f'%(valid_auprc))
    print('Best RNN AUPRC on test set : %.2f'%(test_auprc))
    from IPython import embed; embed()

    
    
#     # get the mus and covariances of the best fit
#     fit_df = pd.read_csv(best_fit_csv)
    
#     # plot the loss plots
#     f, axs = plt.subplots(2,1)
#     fit_df.plot(x='epochs', y=['hmm_model_loss', 'predictor_loss', 'loss'], ax=axs[0])
#     fit_df.plot(x='epochs', y=['predictor_AUC', 'predictor_accuracy'], ax=axs[1])
#     f.savefig('loss_plots.png')
    
    
 