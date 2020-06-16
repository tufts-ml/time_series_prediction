# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
import torch
import ast

def draw_from_discrete_pmf(states, p, hold_state):
    '''
    inputs
    ------
     states : list of states
     p : tuple of probabilities for each state
     hold_state : scalar denoting number of replications of the drawn samples
    
    output
    ------
    hold_state x 1 samples drawn from discrete pmf
    '''
    drawn_state = np.zeros(hold_state)
    drawn_state[:] = rv_discrete(name='custm', values=(states, p)).rvs(size=1)
    return drawn_state

def draw_state_sequence(T, init_probas, states, transitionMatrix, hold_state_t = 2):    
    '''
    inputs
    ------   
    T : length of time series (scalar) 
    init_probas : tuple of initial probabilities
    states : list of states
    
    output
    ------
    T x 1 samples drawn from the Markov model
    '''
    # define some initial probabilities of each state
    init_state = draw_from_discrete_pmf(states, init_probas, hold_state_t)

    # draw T samples from the above model
    drawn_states = []

    while len(drawn_states)<=T:
        if len(drawn_states)==0:
            drawn_states.extend(init_state)
        elif drawn_states[-1] == 0:
            drawn_states.extend(draw_from_discrete_pmf(states, transitionMatrix[0], hold_state_t))
        elif drawn_states[-1] == 1:    
            drawn_states.extend(draw_from_discrete_pmf(states, transitionMatrix[1], hold_state_t))
        elif drawn_states[-1] == 2:
            drawn_states.extend(draw_from_discrete_pmf(states, transitionMatrix[2], hold_state_t))

    drawn_states = np.asarray(drawn_states[:T])
    
    return drawn_states

def generate_time_series_from_state_sequences(state_sequences_DTN):
    # define the distribution for samples drawn for each state as :
    # background : N(0,.5), A : N(1,1), B : N(-1,1)
    time_series_data_DTN = state_sequences_DTN.copy()
    
    # get the counts of all the states 0,1 and 2
    for state in range(3):
        counts_per_state = (state_sequences_DTN==state).sum()
        if state==0:
            time_series_data_DTN[state_sequences_DTN==state] = (.5**.5)*np.random.randn(counts_per_state)
        elif state==1:
            time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state) + 2
        else:
            time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state) - 2
            
    return time_series_data_DTN

# NEW collapse functions
def replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound, **kwargs):
    percentile_data_np = data_np[lower_bound:upper_bound,:]
    all_nan_col_ind = np.isnan(percentile_data_np).all(axis = 0)
    if sum(all_nan_col_ind)>0:
        percentile_data_np[:,all_nan_col_ind] = 0 
    return percentile_data_np

def collapse_mean_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)
    return np.nanmean(percentile_data_np, axis=0)
    
def collapse_median_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)  
    return np.nanmedian(percentile_data_np, axis=0)
    
def collapse_standard_dev_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)   
    return np.nanstd(percentile_data_np, axis=0)
    
def collapse_min_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)  
    return np.nanmin(percentile_data_np, axis=0)
    
def collapse_max_np(data_np, lower_bound, upper_bound, **kwargs):
    # replace columns containing all nans to 0 because nanfunc throws error on all nan columns
    percentile_data_np = replace_all_nan_cols_with_zeros(data_np, lower_bound, upper_bound)   
    return np.nanmax(percentile_data_np, axis=0)

def collapse_count_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).sum(axis=0)
 
def collapse_present_np(data_np, lower_bound, upper_bound, **kwargs):
    return (~np.isnan(data_np[lower_bound:upper_bound,:])).any(axis=0)

def collapse_slope_np(data_np, lower_bound, upper_bound, **kwargs): 
    percentile_t_np = np.arange(lower_bound, upper_bound).astype(float)
    percentile_data_np = data_np[lower_bound:upper_bound,:]
    n_cols = percentile_data_np.shape[1]
    collapsed_slope = np.zeros(n_cols)
    
    for col in range(n_cols):
        mask = ~np.isnan(percentile_data_np[:,col])
        if mask.sum():
            xs = percentile_data_np[mask,col]
            ts = percentile_t_np[mask]
            x_mean = np.mean(xs)
            ts -= np.mean(ts)
            xs -= x_mean
            numer = np.sum(ts * xs)
            denom = np.sum(np.square(xs))
            if denom == 0:
                collapsed_slope[col] = 0
            else:
                collapsed_slope[col] = numer/denom
        else:
            collapsed_slope[col] = 0
    return collapsed_slope  



COLLAPSE_FUNCTIONS_np = {
    "mean": collapse_mean_np,
    "std":  collapse_standard_dev_np,
    "median": collapse_median_np,
    "min": collapse_min_np,
    "max": collapse_max_np,
    "slope": collapse_slope_np, 
    "count": collapse_count_np,
    "present": collapse_present_np
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sklearn LogisticRegression')

    parser.add_argument('--T', type=int, default=100,
                        help='Length of time series, default : 100')
    parser.add_argument('--D', type=int, default=1,
                        help='Number of channels')
    parser.add_argument('--N', type=int,  default=200,
                        help='Number of examples')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='simulated_data/',
                        help='dir in which to save generated dataset')
    args = parser.parse_args()
    
    # Number of time steps
    T = args.T
    # define number of channels in each sequence 
    D = args.D
    # define total number of sequences
    N = args.N

    # define 3 states {0 : Background, 1 : state A, 2 : state B}
    states = np.arange(3)
    init_probas = (0.8, 0.1, 0.1)

    # Create a probability transition matrix for the 3 states
    transitionMatrix = [(0.88,0.02,0.1),(0.49,0.02,0.49),(0.88,0.02,0.1)]

    # define how long to hold a state
    hold_state_t = 2

    
    # set random seed to regenerate the same data for reproducability
    np.random.seed(args.seed)
    
    # create a synthetic dataset of sequences and labels. 
    # If the sequences have a transition from A to B in any channel label it as 1 
    state_sequences_DTN = np.zeros([D,T,N])
    y=np.zeros(N)
    for j in range(N):
    #     for i in range(D):
        state_sequences_DTN[:,:,j] = draw_state_sequence(T, init_probas, states, 
                                                          transitionMatrix, 
                                                          hold_state_t)

        # label sequences with state transition from A to B as 1
        for i in range(D):
            fp = np.flatnonzero(np.hstack([0, np.diff(state_sequences_DTN[i,:,j])]))
            for p in range(len(fp)):
                prev = state_sequences_DTN[i, fp[p]-1, j]
                curr = state_sequences_DTN[i, fp[p], j]

                if curr==2.0 and prev==1.0:
                    y[j]=1

    # generate the time series data from the state sequence
    time_series_data_DTN = generate_time_series_from_state_sequences(state_sequences_DTN)
    print('Done creating data! Collapsing..')
    
    # set the collapse features and their ranges
    ops = ['count', 'mean', 'median', 'std', 'min', 'max' ,'slope']
    range_pairs = '[(0, 10), (0, 25), (0, 50), (50, 100), (75, 100), (90, 100), (0, 100)]'
    
    # Initialize matrix to store collapsed ts
    F = len(ast.literal_eval(range_pairs))*len(ops)  
    collapsed_time_series_DFN = np.zeros([D, F, N])
    f=0
    for op in ops:
        for low, high in ast.literal_eval(range_pairs):
            
            lower_bound = (T*low)//100
            upper_bound = (T*high)//100
            if lower_bound == upper_bound:
                upper_bound = lower_bound+1
            
            for n in range(N):
                collapsed_time_series_DFN[:,f,n] = COLLAPSE_FUNCTIONS_np[op](time_series_data_DTN[:,:,n].T, lower_bound, upper_bound)
        f+=1
    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(collapsed_time_series_DFN.T, y, test_size=0.2, random_state=42) 
    
    
    # save in the folder used by the ts predict pipeline
    # Output data
    fdir_train_test = args.save_dir

    if not os.path.exists(fdir_train_test):
        os.mkdir(fdir_train_test)
    
#     from IPython import embed; embed()
    torch.save(torch.from_numpy(X_train[:,:,0]), fdir_train_test + 'X_train.pt')
    torch.save(torch.from_numpy(X_test[:,:,0]), fdir_train_test + 'X_test.pt')
    torch.save(torch.from_numpy(y_train), fdir_train_test + 'y_train.pt')
    torch.save(torch.from_numpy(y_test), fdir_train_test + 'y_test.pt')
    print('Done..')
    
    
    #     # define 2 states {0 : Background, 1 : state A}
#     states = np.arange(2)
#     init_probas = (0.97, 0.03)

#     # Create a probability transition matrix for the 3 states
#     transitionMatrix = [(0.995,0.005),(0.98,0.02)]

#     # define how long to hold a state
#     hold_state_t = 1