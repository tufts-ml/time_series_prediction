# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
import torch

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
#             time_series_data_DTN[state_sequences_DTN==state] = (.5**.5)*np.random.randn(counts_per_state)
            time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state)
        elif state==1:
#             time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state) + 2
            time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state) + 5
        else:
            time_series_data_DTN[state_sequences_DTN==state] = np.random.randn(counts_per_state) - 2
            
    return time_series_data_DTN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sklearn LogisticRegression')

    parser.add_argument('--T', type=int, default=100,
                        help='Length of time series, default : 100')
    parser.add_argument('--D', type=int, default=1,
                        help='Number of channels')
    parser.add_argument('--N', type=int,  default=500,
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

#     # define 3 states {0 : Background, 1 : state A, 2 : state B}
#     states = np.arange(3)
#     init_probas = (0.8, 0.1, 0.1)

#     # Create a probability transition matrix for the 3 states
#     transitionMatrix = [(0.88,0.02,0.1),(0.49,0.02,0.49),(0.88,0.02,0.1)]

#     # define how long to hold a state
#     hold_state_t = 2

    # define 2 states {0 : Background, 1 : state A}
    states = np.arange(2)
    init_probas = (0.97, 0.03)

    # Create a probability transition matrix for the 3 states
    transitionMatrix = [(0.995,0.005),(0.98,0.02)]

    # define how long to hold a state
    hold_state_t = 1
    
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

                if curr==1.0 and prev==0.0:
                    y[j]=1

    # generate the time series data from the state sequence
    time_series_data_DTN = generate_time_series_from_state_sequences(state_sequences_DTN)
    
    print('Done creating data! Performing train - test split to save..')
    
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
    
    
    # create a plot showing examples of positive and negative labels in the simulated data
    # get examples of time series sequence with label 0 and 1 and plot it
    inds_label_0 = np.flatnonzero(y==0)
    inds_label_1 = np.flatnonzero(y==1)
    
    fontsize = 16
    f,axs = plt.subplots(4,1)
    # plot states of example with label 0
    axs[0].plot(range(T),state_sequences_DTN[:,:,inds_label_0[1]].T, '.')
    axs[0].set_title('Example with No State Transition from Background to A : (Label 0)', fontsize = fontsize)
    axs[0].set_yticks(np.arange(3))
    axs[0].set_yticklabels(['Background', 'A', 'B'])
    axs[0].set_ylabel('State', fontsize = fontsize)

    # plot time series sequence of example with label 0
    axs[1].plot(range(T), time_series_data_DTN[:,:,inds_label_0[0]].T, '-.')
    axs[1].set_ylim([-8,8])
    axs[1].set_ylabel('Time series : x(t)', fontsize = fontsize)

    # plot states of example with label 1
    axs[2].set_yticks(np.arange(3))
    axs[2].set_yticklabels(['Background', 'A', 'B'])
    axs[2].plot(range(T),state_sequences_DTN[:,:,inds_label_1[0]].T, '.')
    axs[2].set_title('Example with State Transition from Background to A : (Label 1)', fontsize = fontsize)
    axs[2].set_ylabel('State', fontsize = fontsize)

    # plot time series sequence of example with label 0
    axs[3].plot(range(T), time_series_data_DTN[:,:,inds_label_1[0]].T, '-.')
    axs[3].set_ylim([-8,8])
    axs[3].set_xlabel('Time (t)', fontsize = fontsize)
    axs[3].set_ylabel('Time series : x(t)', fontsize = fontsize)
    for i,ax in enumerate(axs):
        ax.set_xlim([0,100])
        if i<len(axs)-1:
            ax.set_xticks([])

    f.set_size_inches(18.5, 10.5)
    f.savefig(fdir_train_test + '/simulated_data.png')