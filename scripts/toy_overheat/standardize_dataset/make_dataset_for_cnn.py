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
    
    # draw T samples from the above model
    while len(drawn_states)<=T:
        if len(drawn_states)==0:
            drawn_states.extend(init_state)
        else:
            z_prev = drawn_states[-1]
            drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[int(z_prev)], duration))

    drawn_states = np.asarray(drawn_states[:T])
    
    return drawn_states

def generate_data_sequences_given_state_sequences(
        state_sequences_TN, possible_states, mean_KD, stddev_KD):
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
            
            data_DTN[:, cur_bin_mask_T, n] = np.random.normal(mean_KD[state], stddev_KD[state], size=C).T
            
            if (state == 2)&(C>0):
                y_N[n]=1
            
    return data_DTN, y_N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Tmax', type=int, default=110,
                        help='Length of time series, default : 100')
    parser.add_argument('--Nmax', type=int, default=5000, help="Max number of sequences to generate.")
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--output_dir', type=str, default='simulated_data/',
                        help='dir in which to save generated dataset')
    args = parser.parse_args()
    
    # Number of time steps
#     Tmin = args.Tmin
    Tmax = args.Tmax

    # define number of channels in each sequence 
    D = 1
    n_states = 3
    
    # define total number of sequences
    Nmax = args.Nmax

    # define 2 states {0 : Background, 1 : state A}
    states = np.arange(n_states)
    init_proba_K = (0.5, 0.49, 0.01)

    # Create a probability transition matrix for the 3 states
    trans_proba_KK = [(0.5, 0.5, 0.0),
                      (0.492, 0.5, 0.008),
                      (0.0, 1.0, 0.0)]

#     mean_overheat = 3

#     stddev_KD = np.ones((n_states, D))
#     stddev_KD[2] = 0.3

#     mean_KD = np.zeros((n_states, D))
#     mean_KD[0, 0] = 0
#     mean_KD[1, 0] = -1
#     mean_KD[2, 0] = mean_overheat
    
    mean_overheat = 5

    stddev_KD = np.ones((n_states, D))
    stddev_KD[2] = 0.3

    mean_KD = np.zeros((n_states, D))
    mean_KD[0, 0] = -4
    mean_KD[1, 0] = -5
    mean_KD[2, 0] = mean_overheat

    # define how long to hold a state
    duration = 2
    
    # set random seed to regenerate the same data for reproducability
    np.random.seed(args.seed)
    
    # create a synthetic dataset of sequences and labels.  
    state_sequences_TN = np.nan + np.zeros([Tmax, Nmax])
    start_time_sec = time.time()
    T = Tmax # fix the number of time-steps for all sequences
    
    
    for j in range(Nmax):
#         T = rs.randint(low=Tmin, high=Tmax+1)
        state_sequences_TN[:T,j] = generate_state_sequence(T, states, init_proba_K, 
                                                          trans_proba_KK, 
                                                          duration)

    # generate the time series data from the state sequence
    data_DTN, y_N = generate_data_sequences_given_state_sequences(
        state_sequences_TN, states, mean_KD, stddev_KD)
    N = data_DTN.shape[2]

    seq_list = list()
    feature_columns = ['temperature']
    for n in range(N):
        mask_T = np.isfinite(data_DTN[0, :, n])
        tidy_df = pd.DataFrame(data_DTN[:, :, n][:, mask_T].T, columns=feature_columns)
        tidy_df['timestep'] = np.arange(np.sum(mask_T))
        tidy_df['sequence_id'] = n
        tidy_df['did_overheat_binary_label'] = int(y_N[n])
        seq_list.append(tidy_df)

    tidy_df = pd.concat(seq_list)
    tidy_pertstep_df = tidy_df[['sequence_id', 'timestep'] + feature_columns].copy()
    tidy_perseq_df = tidy_df[['sequence_id', 'did_overheat_binary_label']]
    tidy_perseq_df = tidy_perseq_df.drop_duplicates().copy()

    tidy_pertstep_df.to_csv(
        os.path.join(args.output_dir, 'cnn_features_per_tstep.csv'),
        index=False)
    tidy_perseq_df.to_csv(
        os.path.join(args.output_dir, 'cnn_outcomes_per_seq.csv'),
        index=False)

    print("Wrote features to:")
    print(
        os.path.join(args.output_dir, 'cnn_features_per_tstep.csv'))
    print("Wrote outcomes to:")
    print(
        os.path.join(args.output_dir, 'cnn_outcomes_per_seq.csv'))

    
    # create a plot showing examples of positive and negative labels in the simulated data
    # get examples of time series sequence with label 0 and 1 and plot it
    inds_label_0 = np.flatnonzero(y_N==0)
    inds_label_1 = np.flatnonzero(y_N==1)
    
    
    print('Total number of sequences : %s'%(len(y_N)))
    print('Number of negative sequences : %s'%(len(inds_label_0)))
    print('Number of positive sequences : %s'%(len(inds_label_1)))
    
    # plot example sequence
    fontsize = 10
    
    inds_list = [inds_label_0, inds_label_1]
    y_labels = ['y=0', 'y=1']
    
    for i in range(len(y_labels)):
        f,axs = plt.subplots(2,1, sharex=True, figsize=(15, 5))
        axs[0].plot(range(Tmax), state_sequences_TN[:, inds_list[i][0]], '.')
        axs[0].set_ylabel('State', fontsize = fontsize)
        axs[0].set_title('State Sequence for %s'%y_labels[i])
        axs[0].set_yticks(np.arange(3))
        axs[0].set_ylim([-.5, 2.5])
        axs[1].plot(range(Tmax), data_DTN[:,:,inds_list[i][0]].T)
        axs[1].set_xlim([0, Tmax])
        axs[1].set_ylim([-10, 10])
        axs[1].set_xlabel('Time', fontsize = fontsize)
        axs[1].set_ylabel('Temperature (deg C)', fontsize = fontsize)
        axs[1].set_title('Generated Sequence for %s'%y_labels[i])

        f.savefig(os.path.join(args.output_dir, 'cnn_example_%s_sequence.png'%y_labels[i]))
    
