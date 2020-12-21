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
            if (state == 0)&(C>0)&(np.any(data_DTN[1, cur_bin_mask_T, n]>0)):
                y_N[n]=1
            elif (state == 3)&(C>0)&(np.any(data_DTN[1, cur_bin_mask_T, n]<0)):
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
    init_proba_K = 5*np.ones(n_states)
    init_proba_K[0] = 2
    init_proba_K[3] = 2
    init_proba_K = softmax(init_proba_K)
    
#     trans_proba_KK = rs.dirichlet((50*np.ones(n_states)), n_states)
    trans_proba_KK = np.stack([init_proba_K for i in range(n_states)])
    
    # fix the HMM means x axis from 0 to n_states*spacing in steps of spacing
    spacing = 5
    mean_all = np.arange(0, n_states*spacing, spacing)
    mean_KD = np.zeros((n_states, D))
    mean_KD[:, 0] = mean_all
      
    cov_KDD = np.stack([np.eye(D) for i in range(n_states)])

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

    seq_list = list()
    feature_columns = ['temperature_1', 'temperature_2']
    for n in range(N):
#         mask_T = np.isfinite(data_DTN[:, :, n])
        tidy_df = pd.DataFrame(data_DTN[:, :, n].T, columns=feature_columns)
        tidy_df['timestep'] = np.arange(T)
        tidy_df['sequence_id'] = n
        tidy_df['did_overheat_binary_label'] = int(y_N[n])
        seq_list.append(tidy_df)
    
    
    tidy_df = pd.concat(seq_list)
    tidy_pertstep_df = tidy_df[['sequence_id', 'timestep'] + feature_columns].copy()
    tidy_perseq_df = tidy_df[['sequence_id', 'did_overheat_binary_label']]
    tidy_perseq_df = tidy_perseq_df.drop_duplicates().copy()

    tidy_pertstep_df.to_csv(
        os.path.join(args.output_dir, 'features_2d_per_tstep.csv'),
        index=False)
    tidy_perseq_df.to_csv(
        os.path.join(args.output_dir, 'outcomes_per_seq.csv'),
        index=False)

    print("Wrote features to:")
    print(
        os.path.join(args.output_dir, 'features_2d_per_tstep.csv'))
    print("Wrote outcomes to:")
    print(
        os.path.join(args.output_dir, 'outcomes_per_seq.csv'))

    
    # create a plot showing examples of positive and negative labels in the simulated data
    # get examples of time series sequence with label 0 and 1 and plot it
    inds_label_0 = np.flatnonzero(y_N==0)
    inds_label_1 = np.flatnonzero(y_N==1)
    
    print('Total number of sequences : %s'%(len(y_N)))
    print('Number of negative sequences : %s'%(len(inds_label_0)))
    print('Number of positive sequences : %s'%(len(inds_label_1)))
    
    fontsize = 8
    f,axs = plt.subplots(1,1, figsize=(15, 5))

    # plot time series sequence of example with label 0
#     n_plot_seqs = 50
#     axs[0].plot(range(Tmax), data_DTN[0,:,inds_label_0[:n_plot_seqs]].T, '-.')
    axs.scatter(data_DTN[0, :, inds_label_0], data_DTN[1, :, inds_label_0], 
                marker='x', s=2, c='b', label='y=0')
    axs.scatter(data_DTN[0, :, inds_label_1], data_DTN[1, :, inds_label_1], 
                marker='o', s=2, c='r', label='y=1')
    
    axs.set_ylim([-4,4])
    axs.set_xlim([-spacing,n_states*spacing+spacing])
    axs.set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
    axs.set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)

    # plot time series sequence of example with label 0
#     axs[1].plot(range(Tmax), data_DTN[0,:,inds_label_1[:n_plot_seqs]].T, '-.')
#     axs[1].set_ylim([-8,8])
#     axs[0].set_xlim([-spacing,n_states*spacing+spacing])
#     axs[1].set_ylabel('Temperature_1 (deg C)', fontsize=fontsize)
#     axs[1].set_xlabel('Temperature_0 (deg C)', fontsize = fontsize)
    
    plot_png = os.path.join(args.output_dir, 'example_pos_and_neg_sequences.png')
    print('Saving generated positive and negative sequence plots to %s'%plot_png)
    f.savefig(plot_png)
