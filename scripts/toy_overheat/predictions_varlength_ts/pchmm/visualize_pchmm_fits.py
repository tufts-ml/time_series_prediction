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
                marker='$x$', color='salmon', linestyle=':', alpha=0.5, label='y=0')
    ax.scatter(data_DTN[0, :, inds_label_1], data_DTN[1, :, inds_label_1], 
                marker='$o$', color='b', linestyle=':', alpha=0.5, label='y=1')
    
    fontsize=10
    ax.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
    ax.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    
    ax.set_xlim([-5, 24])
    ax.set_ylim([-5, 5])
    # Plot the paramters
    cmap = sns.diverging_palette(5, 150, l=20, n=mu_all.shape[0], center="dark")
    for i, (mu, cov) in enumerate(zip(mu_all, cov_all)):
        if len(cov.shape)==1:
            cov = np.diag(cov)

        xg = np.linspace(*((max(mu[0] - 5 * cov[0,0], ax.get_xlim()[0]), min(mu[0] + 5 * cov[0,0], ax.get_xlim()[1])) + (1000,)))
        yg = np.linspace(*((max(mu[1] - 5 * cov[1,1], ax.get_ylim()[0]), min(mu[1] + 5 * cov[1,1], ax.get_ylim()[1])) + (1000,)))
#         xg = np.linspace(mu[0] - 4*cov[0,0], mu[0] + 4*cov[0, 0], 3000)
#         yg = np.linspace(mu[1] - 6*cov[1,1], mu[1] + 6*cov[1, 1], 3000)
        Xg, Yg = np.meshgrid(xg, yg)
        Zg = mvn.pdf(
            np.stack([Xg.flatten(), Yg.flatten()]).T, mean=mu, cov=cov
        ).reshape(Xg.shape)
        
#         r = np.sqrt(np.sum(np.square(x_N2), axis=1))
#         levels = np.percentile(Zg, [50, 95, 99])
        ax.contour(Xg, Yg, Zg, levels=levels, colors='black', linewidths=3)
    
#         from IPython import embed; embed()
    f.savefig('pchmm_fits.png')
    
    
    
def visualize2D_new(data_DTN=None, y_N=None, mu_all=None, cov_all=None, levels=3, 
                colorlist=['salmon', 'blue'], markerlist=['$x$', '$o$'], alpha=0.3, 
                    missing_handling="", perc_obs='20'):
    
    f, ax = plt.subplots(figsize=(15, 5))
    sns.set_context("notebook", font_scale=1.6)
    
    
    inds_label_0 = np.flatnonzero(y_N==0)
    inds_label_1 = np.flatnonzero(y_N==1)
    
#     n_plot_seqs = 100
    ax.scatter(data_DTN[0, :, inds_label_0], data_DTN[1, :, inds_label_0], 
                marker='$x$', color='salmon', linestyle=':', alpha=0.5, label='y=0')
    ax.scatter(data_DTN[0, :, inds_label_1], data_DTN[1, :, inds_label_1], 
                marker='$o$', color='b', linestyle=':', alpha=0.5, label='y=1')
    
    fontsize=10
#     ax.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
#     ax.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    
    ax.set_xlim([-5, 30])
    ax.set_ylim([-5, 3])
    
    
    for ii in range(mu_all.shape[0]):
        D = len(mu_all[ii])
        cov_DD = np.diag(cov_all[ii, :])
        mu_D = mu_all[ii]
        
        # Decompose cov matrix into eigenvalues "lambda[d]" and eigenvectors "U[:,d]"
        lambda_D, U_DD = np.linalg.eig(cov_DD)

        # Verify orthonormal
        assert np.allclose(np.eye(D), np.dot(U_DD, U_DD.T))
        # View eigenvector matrix as a rotation transformation
        rot_DD = U_DD

        # Prep for plotting elliptical contours
        # by creating grid of G different (x,y) points along perfect circle
        # Recall that a perfect circle is swept by considering all radians between [-pi, +pi]
        unit_circle_radian_step_size=0.03
        t_G = np.arange(-np.pi, np.pi, unit_circle_radian_step_size)
        x_G = np.sin(t_G)
        y_G = np.cos(t_G)
        Zcirc_DG = np.vstack([x_G, y_G])

        # Warp circle into ellipse defined by Sigma's eigenvectors
        # Rescale according to eigenvalues
        Zellipse_DG = np.sqrt(lambda_D)[:,np.newaxis] * Zcirc_DG
        # Rotate according to eigenvectors
        Zrotellipse_DG = np.dot(rot_DD, Zellipse_DG)
        
        radius_lengths=[0.3, 0.6, 0.9, 1.2, 1.5]
        
        # Plot contour lines across several radius lengths
        for r in radius_lengths:
            Z_DG = r * Zrotellipse_DG + mu_D[:, np.newaxis]
            plt.plot(
                Z_DG[0], Z_DG[1], '.-',
                color='k',
                markersize=3.0,
                markerfacecolor='k',
                markeredgecolor='k')
    
    if missing_handling=='no_imp':
        ax.set_title("PC-HMM fits handling %d percent missing observations without imputation (Ours) AUROC : %.4f"%(100-int(perc_obs), best_fit_auc)) 
    elif missing_handling=='mean_imp':
        ax.set_title("PC-HMM fits handling %d percent missing observations with mean imputation AUROC : %.4f"%(100-int(perc_obs), best_fit_auc)) 
    elif missing_handling=='ffill_imp':
        ax.set_title("PC-HMM fits handling %d percent missing observations with forward fill imputation AUROC : %.4f"%(100-int(perc_obs), best_fit_auc))
    else:
        ax.set_title("PC-HMM fits : no missing data AUROC : %.4f"%(best_fit_auc))  
        
    ax.grid(True)
    f.savefig(os.path.join('toydata_with_missing_results', 
                           'pchmm_fits_%s_perc_obs=%s.png'%(missing_handling, str(perc_obs))))
    f.savefig(os.path.join('toydata_with_missing_results', 
                           'pchmm_fits_%s_perc_obs=%s.pdf'%(missing_handling, str(perc_obs))), bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pchmm fitting')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--fits_dir', type=str, required=True)
    args = parser.parse_args()
        
    #go through all the saved loss plots
    missing_handling_aka_dict = {'no_imp' : 'no imputation (ours)',
                                'ffill_imp' : 'forward fill imputation',
                                'mean_imp' : 'mean imputation'}
    final_perf_df_list = []
    for missing_handling in ['no_imp', 'ffill_imp', 'mean_imp']:
        for perc_obs in ['20', '40', '60', '80', '100']:
            all_fits_csvs = glob.glob(os.path.join(args.fits_dir, "pchmm-*missing_handling=%s-*perc_obs=%s-*-lamb=*.csv"%(missing_handling, perc_obs)))
            losses_per_fit_list = []
            auc_per_fit_list = []
            for fit_csv in all_fits_csvs:
                fit_df = pd.read_csv(fit_csv)
                lamb = int(fit_csv.replace('.csv', '').split('lamb=')[-1])
                curr_loss = fit_df['hmm_model_loss'].to_numpy()[-1] + (fit_df['predictor_loss'].to_numpy()[-1])/lamb
        #         curr_loss = (fit_df['val_predictor_loss'].to_numpy()[-1])/lamb
                losses_per_fit_list.append(curr_loss)
                auc_per_fit_list.append(fit_df['predictor_AUC'].to_numpy()[-1])


#             best_fit_ind = np.argmax(auc_per_fit_list)
            best_fit_ind = np.argmin(losses_per_fit_list)
            best_fit_csv = all_fits_csvs[best_fit_ind]
            best_fit_auc = auc_per_fit_list[best_fit_ind]

            print(best_fit_csv)
    #         print("%s : AUC : %.5f"%(missing_handling, best_fit_auc))
            #get the mus and covariances of the best fit
            best_fit_mu = best_fit_csv.replace(".csv", "-fit-mu.npy")
            best_fit_cov = best_fit_csv.replace(".csv", "-fit-cov.npy")
            best_fit_eta = best_fit_csv.replace(".csv", "-fit-eta.npy")

            fit_df = pd.read_csv(best_fit_csv)

            final_perf_csv = os.path.join(args.fits_dir, 'final_perf_'+best_fit_csv.split('/')[-1])
            final_perf_df = pd.read_csv(final_perf_csv)
            
            final_perf_df['model'] = 'PCHMM %s'%missing_handling_aka_dict[missing_handling]
            final_perf_df['perc_obs'] = perc_obs
            print('--------------------------------------')
            print('MISSING HANDLING = %s perc observed = %s'%(missing_handling, perc_obs))
            print(final_perf_df)
            print('--------------------------------------')
            
            
            final_perf_df_list.append(final_perf_df)
            
            
            # plot the loss plots
            f, axs = plt.subplots(2,1, figsize = (8, 5))
            fit_df.plot(x='epochs', y=['predictor_loss', 'loss', 
                                       'val_predictor_loss', 'val_loss'], ax=axs[0])
            fit_df.plot(x='epochs', y=['predictor_AUC', 'val_predictor_AUC'], ax=axs[1])
            axs[1].set_ylim([0.5, 1])
            f.savefig('loss_plots_%s.png'%missing_handling)

        #     best_fit_mu = glob.glob(os.path.join(args.fits_dir, "pchmm-lr=100*fit-mu.npy"))[0]
        #     best_fit_cov = glob.glob(os.path.join(args.fits_dir, "pchmm-lr=100*fit-cov.npy"))[0]
        #     best_fit_eta = glob.glob(os.path.join(args.fits_dir, "pchmm-lr=100*fit-eta.npy"))[0]
            mu_all = np.load(best_fit_mu)
            cov_all = np.load(best_fit_cov)
            eta_all = np.load(best_fit_eta)

            features_df = pd.read_csv(os.path.join(args.data_dir, "features_2d_per_tstep_%s_observed=%s_perc.csv"%(missing_handling, perc_obs)))
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

            visualize2D_new(data_DTN=data_DTN, y_N=y_N, mu_all=mu_all, cov_all=cov_all, levels=3, 
                        colorlist=['salmon', 'blue'], markerlist=['$x$', '$o$'], alpha=0.3, 
                            missing_handling=missing_handling,
                           perc_obs=perc_obs)
    
    final_perf_df = pd.concat(final_perf_df_list).reset_index(drop=True)    
    final_perf_df.to_csv('PCHMM_performance_toy_data.csv', index=False)

    from IPython import embed; embed()
        
