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

    while len(drawn_states)<=T:
        if len(drawn_states)==0:
            drawn_states.extend(init_state)
        elif drawn_states[-1] == 0:
            drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[0], duration))
        elif drawn_states[-1] == 1:    
            drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[1], duration))
        elif drawn_states[-1] == 2:
            drawn_states.extend(draw_from_discrete_pmf(states, trans_proba_KK[2], duration))

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
    
    for n in range(N):
        for state in possible_states:
            cur_bin_mask_T = state_sequences_TN[:,n] == state
            C = np.sum(cur_bin_mask_T)
            if state == 0:
                data_DTN[:, cur_bin_mask_T, n] = np.random.normal(mean_KD[0], stddev_KD[0], size=C)
            else:
                data_DTN[:, cur_bin_mask_T, n] = np.random.normal(mean_KD[1], stddev_KD[1], size=C)
            
    return data_DTN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Tmin', type=int, default=90,
                        help="Minimum length of any example's time series, default : 100")
    parser.add_argument('--Tmax', type=int, default=110,
                        help='Length of time series, default : 100')
    parser.add_argument('--Nmax', type=int, default=5000, help="Max number of sequences to generate.")
    parser.add_argument('--min_num_sequences_per_label', type=int,  default=100)
    parser.add_argument('--max_num_sequences_per_label', type=int,  default=500)
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--output_dir', type=str, default='simulated_data/',
                        help='dir in which to save generated dataset')
    args = parser.parse_args()
    
    # Number of time steps
    Tmin = args.Tmin
    Tmax = args.Tmax

    # define number of channels in each sequence 
    D = 1
    
    # define total number of sequences
    Nmax = args.Nmax

    # define 2 states {0 : Background, 1 : state A}
    states = np.arange(2)
    init_proba_K = (1.00, 0.00)

    # Create a probability transition matrix for the 3 states
    trans_proba_KK = [(0.995,0.005),(0.98,0.02)]

    mean_OK = -5
    mean_overheat = +5

    stddev_KD = np.ones((2, D))
    stddev_KD[1] = 0.3

    mean_KD = mean_OK + np.zeros((2, D))
    mean_KD[1, 0] = mean_overheat

    # define how long to hold a state
    duration = 5
    
    # set random seed to regenerate the same data for reproducability
    np.random.seed(args.seed)
    
    # create a synthetic dataset of sequences and labels. 
    # If the sequences have a transition from A to B in any channel label it as 1 
    state_sequences_TN = np.nan + np.zeros([Tmax, Nmax])
    y = -1 * np.ones(Nmax)
    start_time_sec = time.time()
    for j in range(Nmax):
        T = np.random.randint(low=Tmin, high=Tmax+1)
        state_sequences_TN[:T,j] = generate_state_sequence(T, states, init_proba_K, 
                                                          trans_proba_KK, 
                                                          duration)

        # fp : fenceposts for current sequence
        fp = np.hstack([0, np.flatnonzero(np.diff(state_sequences_TN[:T,j]))+1])

        # Label any sequence containing any usage of "signal" state as 1
        # Any sequence that is purely background noise will have label 0
        if np.any(state_sequences_TN[:T, j]):
            y[j] = 1
        else:
            y[j] = 0

        n_pos = np.sum(y==1)
        n_neg = np.sum(y==0)
        too_many_pos = n_pos > args.max_num_sequences_per_label
        too_many_neg = n_neg > args.max_num_sequences_per_label
        if too_many_neg and y[j] == 0:
            y[j] = -1
        if too_many_pos and y[j] == 1:
            y[j] = -1

        has_enough_pos = np.sum(y == 1) >= args.min_num_sequences_per_label
        has_enough_neg = np.sum(y == 0) >= args.min_num_sequences_per_label


        if ((n_pos + n_neg) % 50 == 0) and y[j] >= 0:
            print("Generated %5d pos and %5d neg sequences after %6.1f sec" % (
                n_pos, n_neg, time.time() - start_time_sec))

        if has_enough_neg and has_enough_pos:
            break

    if not (has_enough_pos and has_enough_neg):
        raise ValueError("Did not generate enough positive and negative labeled sequences")
    
    # Keep only the sequences with a valid label
    state_sequences_TN = state_sequences_TN[:, y >= 0].copy()
    y = y[y>= 0].copy()

    # generate the time series data from the state sequence
    data_DTN = generate_data_sequences_given_state_sequences(
        state_sequences_TN, states, mean_KD, stddev_KD)
    
    N = data_DTN.shape[2]

    seq_list = list()
    feature_columns = ['temperature']
    for n in range(N):
        mask_T = np.isfinite(data_DTN[0, :, n])
        tidy_df = pd.DataFrame(data_DTN[:, :, n][:, mask_T].T, columns=feature_columns)
        tidy_df['timestep'] = np.arange(np.sum(mask_T))
        tidy_df['sequence_id'] = n
        tidy_df['did_overheat_binary_label'] = int(y[n])
        seq_list.append(tidy_df)

    tidy_df = pd.concat(seq_list)
    tidy_pertstep_df = tidy_df[['sequence_id', 'timestep'] + feature_columns].copy()
    tidy_perseq_df = tidy_df[['sequence_id', 'did_overheat_binary_label']]
    tidy_perseq_df = tidy_perseq_df.drop_duplicates().copy()

    tidy_pertstep_df.to_csv(
        os.path.join(args.output_dir, 'features_per_tstep.csv'),
        index=False)
    tidy_perseq_df.to_csv(
        os.path.join(args.output_dir, 'outcomes_per_seq.csv'),
        index=False)

    print("Wrote features to:")
    print(
        os.path.join(args.output_dir, 'features_per_tstep.csv'))
    print("Wrote outcomes to:")
    print(
        os.path.join(args.output_dir, 'outcomes_per_seq.csv'))

    '''
    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(time_series_data_DTN.T, y, test_size=0.2, random_state=42) 
    
#     from IPython import embed; embed()

    
    # save in the folder used by the ts predict pipeline
    # Output data
    fdir_train_test = args.save_dir

    if not os.path.exists(fdir_train_test):
        os.mkdir(fdir_train_test)
    
    torch.save(torch.from_numpy(X_train), fdir_train_test + '/X_train.pt')
    torch.save(torch.from_numpy(X_test), fdir_train_test + '/X_test.pt')
    torch.save(torch.from_numpy(y_train), fdir_train_test + '/y_train.pt')
    torch.save(torch.from_numpy(y_test), fdir_train_test + '/y_test.pt')
    print('Done..')
    '''
    
    # create a plot showing examples of positive and negative labels in the simulated data
    # get examples of time series sequence with label 0 and 1 and plot it
    inds_label_0 = np.flatnonzero(y==0)
    inds_label_1 = np.flatnonzero(y==1)
    
    fontsize = 16
    f,axs = plt.subplots(2,1)

    # plot time series sequence of example with label 0
    axs[0].plot(range(Tmax), data_DTN[:,:,inds_label_0[0]].T, '-.')
    axs[0].set_ylim([-8,8])
    axs[0].set_ylabel('Temperature (deg C)', fontsize = fontsize)

    # plot time series sequence of example with label 0
    axs[1].plot(range(Tmax), data_DTN[:,:,inds_label_1[0]].T, '-.')
    axs[1].set_ylim([-8,8])
    axs[1].set_ylabel('Temperature (deg C)', fontsize=fontsize)
    axs[1].set_xlabel('Timestep (t)', fontsize = fontsize)
   
    f.savefig(os.path.join(args.output_dir, 'example_pos_and_neg_sequences.png'))
