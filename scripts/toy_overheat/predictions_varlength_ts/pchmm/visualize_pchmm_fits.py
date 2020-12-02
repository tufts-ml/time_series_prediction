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

# Function to plot the toy data
def visualize2D(data_DTN=None, y_N=None, mu_all=None, cov_all=None, levels=3, 
                colorlist=['salmon', 'blue'], markerlist=['$x$', '$o$'], alpha=0.3):
    f, ax = plt.subplots(figsize=(15, 5))
    
    inds_label_0 = np.flatnonzero(y_N==0)
    inds_label_1 = np.flatnonzero(y_N==1)
    
#     n_plot_seqs = 100
    ax.scatter(data_DTN[0, :, inds_label_0], data_DTN[1, :, inds_label_0], 
                marker='x', s=2, c='b', label='y=0')
    ax.scatter(data_DTN[0, :, inds_label_1], data_DTN[1, :, inds_label_1], 
                marker='o', s=2, c='r', label='y=1')
    
    ax.set_ylim([-8,8])
    ax.set_xlim([-5,55])
    fontsize=10
    ax.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
    ax.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    
    ax.set_xlim([-5, 50])
    ax.set_ylim([-5, 5])
    # Plot the paramters
    cmap = sns.diverging_palette(5, 150, l=20, n=mu_all.shape[0], center="dark")
    for i, (mu, cov) in enumerate(zip(mu_all, cov_all)):
        xg = np.linspace(*((max(mu[0] - 5 * cov[0,0], ax.get_xlim()[0]), min(mu[0] + 5 * cov[0,0], ax.get_xlim()[1])) + (1000,)))
        yg = np.linspace(*((max(mu[1] - 5 * cov[1,1], ax.get_ylim()[0]), min(mu[1] + 5 * cov[1,1], ax.get_ylim()[1])) + (1000,)))
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = mvn.pdf(
            np.stack([Xg.flatten(), Yg.flatten()]).T, mean=mu, cov=cov
        ).reshape(Xg.shape)
        ax.contour(Xg, Yg, Zg, levels=levels, colors='black', linewidths=3)

    f.savefig('pchmm_fits.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fits_dir', type=str, required=True)
    args = parser.parse_args()
        
    #go through all the saved loss plots
    all_fits_csvs = glob.glob(os.path.join(args.fits_dir, "pchmm-*.csv"))
    
    losses_per_fit_list = []
    auc_per_fit_list = []
    for fit_csv in all_fits_csvs:
        fit_df = pd.read_csv(fit_csv)
        losses_per_fit_list.append(fit_df['predictor_loss'].to_numpy()[-1])
        auc_per_fit_list.append(fit_df['predictor_AUC'].to_numpy()[-1])
    
    
#     best_fit_ind = np.argmin(losses_per_fit_list)
    best_fit_ind = np.argmin(losses_per_fit_list)
    best_fit_csv = all_fits_csvs[best_fit_ind]
    
    # get the mus and covariances of the best fit
    best_fit_mu = best_fit_csv.replace(".csv", "-fit-mu.npy")
    best_fit_cov = best_fit_csv.replace(".csv", "-fit-cov.npy")
    
    
    fit_df = pd.read_csv(best_fit_csv)
    
    # plot the loss plots
    f, axs = plt.subplots(2,1)
    fit_df.plot(x='epochs', y=['hmm_model_loss', 'predictor_loss', 'loss'], ax=axs[0])
    fit_df.plot(x='epochs', y=['predictor_AUC', 'predictor_accuracy'], ax=axs[1])
    f.savefig('loss_plots.png')
    
#     from IPython import embed; embed()
    
    mu_all = np.load(best_fit_mu)
    cov_all = np.load(best_fit_cov)
    
    features_df = pd.read_csv(os.path.join(args.data_dir, "features_2d_per_tstep.csv"))
    outcomes_df = pd.read_csv(os.path.join(args.data_dir, "outcomes_per_seq.csv"))
    features_dict = load_data_dict_json(os.path.join(args.data_dir, "Spec_Features2DPerTimestep.json"))
    
    id_cols = parse_id_cols(features_dict)
    feature_cols = parse_feature_cols(features_dict)
    
    features_outcomes_df = pd.merge(features_df, outcomes_df, on=id_cols)
    
    # get the data in D x T x N form
    data_DTN_list = []
    fp = get_fenceposts(features_outcomes_df, id_cols)
    n_rows = len(fp)-1
    features_arr = features_outcomes_df[feature_cols].values
    for p in range(n_rows):
        data_DTN_list.append(features_arr[fp[p]:fp[p+1], :])

    data_DTN = np.stack(data_DTN_list).T
    features_outcomes_df.drop_duplicates(subset=id_cols, inplace=True)
    y_N = features_outcomes_df['did_overheat_binary_label'].values
    
    
    visualize2D(data_DTN=data_DTN, y_N=y_N, mu_all=mu_all, cov_all=cov_all, levels=3, 
                colorlist=['salmon', 'blue'], markerlist=['$x$', '$o$'], alpha=0.3)
    
    from IPython import embed; embed()
        