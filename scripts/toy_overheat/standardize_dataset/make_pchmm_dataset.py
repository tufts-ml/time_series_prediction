'''
Create toy data


'''
import time
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from scipy.special import softmax
import random
import seaborn as sns

def draw_from_discrete_pmf(states, p, duration):
    ''' Sample from a discrete probability mass function

    Args
    ------
    states : list
        List of states
    p : tuple
        Probabilities for each state
    duration : int
        Duration or "dwell time" of this state
    
    Returns
    -------
    samples : 1D array, size (duration,)
        Each entry is an integer indicator drawn from discrete pmf
    '''
    drawn_state = np.zeros(duration)
    drawn_state[:] = rv_discrete(name='custm', values=(states, p)).rvs(size=1)
    return drawn_state

def generate_state_sequence(T, states, init_proba_K, trans_proba_KK, duration = 2):    
    ''' Generate hidden state assignments following a semi-Markov model

    Args
    ------   
    T : length of time series (scalar) 
    init_proba_K : tuple of initial probabilities
    states : list of states
    
    Returns
    -------
    T x 1 samples drawn from the Markov model
    '''
    # define some initial probabilities of each state
    init_state = draw_from_discrete_pmf(states, init_proba_K, duration)

    # draw T samples from the above model
    drawn_states = []

    while len(drawn_states)<=T:
        if len(drawn_states)==0:
            drawn_states.extend(init_state)
#         elif drawn_states[-1] == 0:
#             drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[0], duration))
#         elif drawn_states[-1] == 1:    
#             drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[1], duration))
#         elif drawn_states[-1] == 2:
#             drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[2], duration))
        else:
            z_prev = drawn_states[-1]
            drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[int(z_prev)], duration))

    drawn_states = np.asarray(drawn_states[:T])
    
    return drawn_states

def generate_data_sequences_given_state_sequences(
        state_sequences_TN, possible_states, mean_KD, cov_KDD):
    ''' Generate data given states

    Returns
    ------
    data_DTN : 3d array, (n_dims, n_timessteps, n_sequences)
        Contains observed features for each timestep
        Any timestep with missing data will be assigned nan
    '''
    K, D = mean_KD.shape
    T, N = state_sequences_TN.shape
    data_DTN = np.nan + np.zeros((D, T, N))
    y_N = np.zeros(N)
    
    for n in range(N):
        for state in possible_states:
            cur_bin_mask_T = state_sequences_TN[:,n] == state
            C = np.sum(cur_bin_mask_T)
            
            data_DTN[:, cur_bin_mask_T, n] = np.random.multivariate_normal(mean_KD[state], cov_KDD[state], size=C).T
            
            # assign y=1 if state is 0 and y dimension of observation is greater than 0 or if state is 3 and y dimension is less than 0
            '''
            if (state == 0)&(C>0)&(np.any(data_DTN[1, cur_bin_mask_T, n]>0)):
                y_N[n]=1
            elif (state == 3)&(C>0)&(np.any(data_DTN[1, cur_bin_mask_T, n]<0)):
                y_N[n]=1
            '''
            if (state == 2)&(C>0)&(np.any(data_DTN[1, cur_bin_mask_T, n]>0)):
                y_N[n]=1
            
    return data_DTN, y_N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Tmin', type=int, default=90,
                        help="Minimum length of any example's time series, default : 100")
    parser.add_argument('--Tmax', type=int, default=110,
                        help='Length of time series, default : 100')
    parser.add_argument('--Nmax', type=int, default=5000, help="Max number of sequences to generate.")
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--n_states', type=int,  default=None)
    parser.add_argument('--output_dir', type=str, default='simulated_data/',
                        help='dir in which to save generated dataset')
    args = parser.parse_args()
    
    # Number of time steps
    Tmin = args.Tmin
    Tmax = args.Tmax

    # define number of channels in each sequence 
    D = 2
    
    # define total number of sequences
    Nmax = args.Nmax
    
    # get number of states and define init probas
    rs = RandomState(args.seed)
    n_states = args.n_states
    states = np.arange(n_states)
    
    # make probability of initializing and transitioning to state 0 and state 3 v.v. low
    '''
    init_proba_K = 5*np.ones(n_states)
#     init_proba_K[0] = 2
#     init_proba_K[3] = 2
    init_proba_K[2] = 4
    init_proba_K = softmax(init_proba_K)
    
#     trans_proba_KK = rs.dirichlet((50*np.ones(n_states)), n_states)
    trans_proba_KK = np.stack([init_proba_K for i in range(n_states)])
    '''
    init_proba_K = np.array([1., 0., 0., 0.])
    trans_proba_KK = np.array([[.5, .5, 0., 0.],
                               [0., .5, .5, 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 0., 1.]])
    
    # fix the HMM means x axis from 0 to n_states*spacing in steps of spacing
    spacing = 8
    mean_all = np.arange(0, n_states*spacing, spacing)
    mean_KD = -1.*np.ones((n_states, D))
    mean_KD[:, 0] = mean_all
    mean_KD[2, 0] = 13
    mean_KD[2, 1] = 0
      
    cov_KDD = np.stack([np.diag([0.3, 1.]) for i in range(n_states)])    
    # define how long to hold a state
    duration = 1
    
    # set random seed to regenerate the same data for reproducability
    np.random.seed(args.seed)
    
    # create a synthetic dataset of sequences and labels.  
    state_sequences_TN = np.nan + np.zeros([Tmax, Nmax])
#     y = -1 * np.ones(Nmax)
    start_time_sec = time.time()
    T = Tmax # fix the number of time-steps for all sequences
    
    
    for j in range(Nmax):
#         T = rs.randint(low=Tmin, high=Tmax+1)
        state_sequences_TN[:T,j] = generate_state_sequence(T, states, init_proba_K, 
                                                          trans_proba_KK, 
                                                          duration)

    # generate the time series data from the state sequence
    data_DTN, y_N = generate_data_sequences_given_state_sequences(
        state_sequences_TN, states, mean_KD, cov_KDD)
    N = data_DTN.shape[2]
    data_DTN_true = data_DTN.copy()
    
    # add missing covariates in 40% of the samples
#     perc_vals_missing = 40
    
    for perc_obs in [20, 40, 60, 80, 100]:
        perc_vals_missing = 100-perc_obs
        n_missing = int(N*T*D*perc_vals_missing/100)
        miss_inds = np.random.choice(data_DTN.size, n_missing, replace=False)
        data_DTN = data_DTN_true.copy()
        data_DTN.ravel()[miss_inds] = np.nan

        seq_list = list()
        seq_list_true = list()
        feature_columns = ['temperature_1', 'temperature_2']
        for n in range(N):
    #         mask_T = np.isfinite(data_DTN[:, :, n])
            true_df = pd.DataFrame(data_DTN_true[:, :, n].T, columns=feature_columns)
            tidy_df = pd.DataFrame(data_DTN[:, :, n].T, columns=feature_columns)

            true_df['timestep'] = np.arange(T)
            tidy_df['timestep'] = np.arange(T)

            true_df['sequence_id'] = n
            tidy_df['sequence_id'] = n

            true_df['did_overheat_binary_label'] = int(y_N[n])
            tidy_df['did_overheat_binary_label'] = int(y_N[n])


            seq_list.append(tidy_df)
            seq_list_true.append(true_df)

        tidy_df = pd.concat(seq_list)
        true_df = pd.concat(seq_list_true)

        true_pertstep_df = true_df[['sequence_id', 'timestep'] + feature_columns].copy()
        tidy_pertstep_df = tidy_df[['sequence_id', 'timestep'] + feature_columns].copy()

        tidy_perseq_df = tidy_df[['sequence_id', 'did_overheat_binary_label']]

        true_pertstep_df.reset_index(drop=True, inplace=True)  
        tidy_pertstep_df.reset_index(drop=True, inplace=True)  
        tidy_perseq_df = tidy_perseq_df.drop_duplicates().copy()

        features_true_df = true_pertstep_df.copy()
        features_without_imputation_df = tidy_pertstep_df.copy()
        features_with_ffill_imputation_df = tidy_pertstep_df.fillna(method='ffill').copy()
        features_with_mean_imputation_df = tidy_pertstep_df.fillna(tidy_pertstep_df.mean()).copy()

        features_true_csv = os.path.join(args.output_dir, 'features_2d_per_tstep_true.csv')
        features_without_imputation_csv = os.path.join(args.output_dir, 
                                                       'features_2d_per_tstep_no_imp_observed=%s_perc.csv'%perc_obs)
        features_with_ffill_imputation_csv = os.path.join(args.output_dir, 'features_2d_per_tstep_ffill_imp_observed=%s_perc.csv'%perc_obs)
        features_with_mean_imputation_csv = os.path.join(args.output_dir, 'features_2d_per_tstep_mean_imp_observed=%s_perc.csv'%perc_obs)
        outcomes_csv = os.path.join(args.output_dir, 'outcomes_per_seq.csv')

        features_true_df.to_csv(features_true_csv, index=False)
        features_without_imputation_df.to_csv(features_without_imputation_csv, index=False)
        features_with_ffill_imputation_df.to_csv(features_with_ffill_imputation_csv, index=False)
        features_with_mean_imputation_df.to_csv(features_with_mean_imputation_csv, index=False)
        tidy_perseq_df.to_csv(outcomes_csv, index=False)

        print("Wrote features to: \n%s \n%s \n%s \n%s"%(features_true_csv, features_without_imputation_csv,
                                                  features_with_ffill_imputation_csv,
                                                  features_with_mean_imputation_csv))
        print("Wrote outcomes to: \n%s"%outcomes_csv)

        # create a plot showing examples of positive and negative labels in the simulated data
        # get examples of time series sequence with label 0 and 1 and plot it
        inds_label_0 = np.flatnonzero(y_N==0)
        inds_label_1 = np.flatnonzero(y_N==1)

        print('Total number of sequences : %s'%(len(y_N)))
        print('Number of negative sequences : %s'%(len(inds_label_0)))
        print('Number of positive sequences : %s'%(len(inds_label_1)))

        features_outcomes_df = pd.merge(features_without_imputation_df, tidy_perseq_df, on=['sequence_id'])
        inds_label_0 = features_outcomes_df['did_overheat_binary_label']==0
        inds_label_1 = features_outcomes_df['did_overheat_binary_label']==1

        fontsize = 10

        for feat_df, feat_aka in [(features_true_df, 'no_missing_observed=%s'%perc_obs),
                                 (features_without_imputation_df, 'no_imputation_observed=%s'%perc_obs),
                                 (features_with_ffill_imputation_df, 'ffill_imputation_observed=%s'%perc_obs),
                                 (features_with_mean_imputation_df, 'mean_imputation_observed=%s'%perc_obs)]:

            f,axs = plt.subplots(1,1, figsize=(15, 5))
            sns.set_context("notebook", font_scale=1.6)
            feature_vals_ND_labels_0 = feat_df[inds_label_0][feature_columns].values
            feature_vals_ND_labels_1 = feat_df[inds_label_1][feature_columns].values


            # plot time series sequence of example with label 0 and 1
            axs.scatter(feature_vals_ND_labels_0[:, 0], feature_vals_ND_labels_0[:, 1], 
                        marker='$x$', color='salmon', linestyle=':', alpha=0.5, label='y=0')
            axs.scatter(feature_vals_ND_labels_1[:, 0], feature_vals_ND_labels_1[:, 1], 
                        marker='$o$', color='b', linestyle=':', alpha=0.5, label='y=1')

            axs.set_xlim([-5, 30])
            axs.set_ylim([-5, 3])
#             axs.set_xlim([-spacing,n_states*spacing])
#             axs.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
#             axs.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)

            axs.set_title('Raw features with %s'%(feat_aka.replace('_', ' ')))
            axs.grid(True)
            plot_png = os.path.join(args.output_dir, 'example_pos_and_neg_sequences_%s.png'%feat_aka)
            plot_pdf = os.path.join(args.output_dir, 'example_pos_and_neg_sequences_%s.pdf'%feat_aka)
            print('Saving generated positive and negative sequence plots to %s'%plot_png)
            f.savefig(plot_png, bbox_inches='tight', pad_inches=0)
            f.savefig(plot_pdf, bbox_inches='tight', pad_inches=0)
    
