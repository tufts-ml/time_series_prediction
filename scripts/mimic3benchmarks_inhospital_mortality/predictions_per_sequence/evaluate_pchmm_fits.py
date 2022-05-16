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
    
    for n_states in [5, 10, 20, 30, 40, 50, 60, 70]:
        try:
            print('---------------------')
            print('states = %s'%str(n_states))
            print('---------------------')
            all_fits_csvs = glob.glob(os.path.join(args.fits_dir, "final_perf*n_states=%s-*.csv"%str(n_states)))
            auprc_per_fit_list = []
            for fit_csv in all_fits_csvs:
                fit_perf_df = pd.read_csv(fit_csv)
                if 'valid_AUPRC' in fit_perf_df.columns:
        #             auprc_last_n_epochs = np.median([i['auprc_valid'] for i in fit_dict_list[-5:]])
                    auprc_per_fit_list.append(fit_perf_df['valid_AUPRC'].values[-1])


            #     best_fit_ind = np.argmin(losses_per_fit_list)
            best_fit_ind = np.argmax(auprc_per_fit_list)
            best_fit_csv = all_fits_csvs[best_fit_ind]
        #     best_fit_json = all_fits_jsons[best_fit_ind]
        #     best_fit_csv = best_fit_json.replace("_auprc_best.json", ".csv")
            best_fit_df = pd.read_csv(best_fit_csv)
        #     test_auprc = best_fit_df['test_auprc'][0]

        #     best_fit_df = pd.DataFrame(json.load(open(all_fits_jsons[best_fit_ind])))
            train_auprc = best_fit_df['train_AUPRC'].values[-1] 
            valid_auprc = best_fit_df['valid_AUPRC'].values[-1] 
            test_auprc = best_fit_df['test_AUPRC'].values[-1] 

            print('Best PCHMM fit : %s'%(best_fit_csv))
            print('Best PCHMM AUPRC on train set : %.5f'%(train_auprc))
            print('Best PCHMM AUPRC on valid set : %.5f'%(valid_auprc))
            print('Best PCHMM AUPRC on test set : %.5f'%(test_auprc))
        except:
            print('Still running n_states=%s'%str(n_states))
        print('---------------------')
    from IPython import embed; embed()

    
    
#     # get the mus and covariances of the best fit
#     fit_df = pd.read_csv(best_fit_csv)
    
#     # plot the loss plots
#     f, axs = plt.subplots(2,1)
#     fit_df.plot(x='epochs', y=['hmm_model_loss', 'predictor_loss', 'loss'], ax=axs[0])
#     fit_df.plot(x='epochs', y=['predictor_AUC', 'predictor_accuracy'], ax=axs[1])
#     f.savefig('loss_plots.png')
    
    
 