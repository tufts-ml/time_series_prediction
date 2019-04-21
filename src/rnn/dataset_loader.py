import os
import numpy as np
import pandas as pd
import torch

class TidySequentialDataCSVLoader(object):

    def __init__(self,
            per_tstep_csv_path=None,
            per_seq_csv_path=None,
            x_col_names='__all__',
            idx_col_names='seq_id',
            y_col_name='y',
            y_label_type='per_tstep',
            max_seq_len=None,
            batch_size=None,
            ):
        ''' Load tidy time-series from .csv files

        Returns
        -------
        Nothing

        Post Condition
        --------------
        Updates internal attribute .batches
        '''
        per_tstep_csv_df = pd.read_csv(per_tstep_csv_path)
        if per_seq_csv_path is not None and os.path.exists(per_seq_csv_path):
            per_seq_csv_df = pd.read_csv(per_seq_csv_path)

        ## Parse sequence ids and compute fenceposts
        idx_P = per_tstep_csv_df[idx_col_names].values.copy()
        del per_tstep_csv_df[idx_col_names]
        uvals = np.unique(idx_P)
        seq_fp = list()
        prev_uval = None
        for pp in range(idx_P.size):
            if idx_P[pp] == prev_uval:
                continue
            else:
                seq_fp.append(pp)
                prev_uval = idx_P[pp]
        seq_fp.append(idx_P.size)
        self.seq_fp = seq_fp

        ## Parse sequence lengths
        N = len(seq_fp) - 1
        self.N = N
        self.n_sequences = N
        self.seq_lens_N = np.asarray(
            [seq_fp[n+1] - seq_fp[n] for n in range(N)],
            dtype=np.int64)
        if max_seq_len is None:
            self.max_seq_len = np.max(self.seq_lens_N)
        else:
            self.max_seq_len = int(max_seq_len)

        ## Parse y
        if y_label_type == 'per_tstep':
            y_P = per_tstep_csv_df[y_col_name].values.copy()
            del per_tstep_csv_df[y_col_name]
            # TODO load full seq labels when needed
            laststep_N = np.cumsum(self.seq_lens_N) - 1
            self.y_N = np.asarray(y_P[laststep_N], dtype=np.int64)
        else:
            y_N = per_seq_csv_df[y_col_name].values.copy()
            del per_seq_csv_df[y_col_name]
            self.y_N = np.asarray(y_N, dtype=np.int64)

        ## Parse x
        if x_col_names == '__all__':
            x_PF = per_tstep_csv_df.values.copy()
        else:
            x_PF = per_tstep_csv_df[x_col_names].values.copy()
        self.x_PF = x_PF

        ## Randomly assign seqs to batches
        # TODO do label balancing??
        if batch_size is None:
            batch_size = N
        self.batch_size = batch_size
        self.assign_seqs_to_batches(N, batch_size)


    def assign_seqs_to_batches(self, N, batch_size, random_state=42):
        '''

        Post Condition
        --------------
        self.seq_ids_per_batch : list of lists, size B
            Entry b gives list of all sequences assigned to batch b
        '''
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        n_batch = int(np.ceil(N / batch_size))
        self.n_batch = n_batch

        if self.n_batch == 1:
            randorder_N = np.arange(N)
        else:
            randorder_N = random_state.permutation(N)

        n_per_batch = [batch_size for _ in range(n_batch)]
        n_total = np.sum(n_per_batch)
        assert n_total >= N
        # Trim off excess, so we've assigned exacty N seqs to B batches
        ii = 0
        while n_total > N:
            n_per_batch[ii] -= 1
            ii += 1
            n_total = np.sum(n_per_batch)

        # Fill out the seq_ids_per_batch
        self.n_per_batch = n_per_batch
        self.seq_ids_per_batch = []
        for b in range(n_batch):
            seq_ids_for_batch_b = randorder_N[:n_per_batch[b]]            
            self.seq_ids_per_batch.append(
                seq_ids_for_batch_b)
            randorder_N = randorder_N[n_per_batch[b]:]
        
    def get_single_sequence_data(self, seq_id=0):
        ''' Get labeled data (x,y) for specific sequence

        Returns
        -------
        x : 3D array, size n_seqs x n_time_steps x n_features
            Contains data for single seq, so first dim will be 1
        y : 1D array, size n_seqs
            Contains data for single seq, so first dim will be 1
        '''
        start = self.seq_fp[seq_id]
        stop = self.seq_fp[seq_id+1]
        x_NTF = self.x_PF[start:stop][np.newaxis,:,:]
        y_N = self.y_N[seq_id:seq_id+1] # make it a 1D array of size 1
        return x_NTF, y_N

    def get_batch_data(self, batch_id=0, to_pytorch_tensor=False):
        ''' Get (x,y) data for specific batch
        
        Returns
        -------
        batchx : 3D array, size n_seqs x n_time_steps x n_features
        batchy : 1D array, size n_seqs 
        '''
        seq_ids = self.seq_ids_per_batch[batch_id]
        batch_x = np.zeros((seq_ids.size, self.max_seq_len, self.x_PF.shape[1]), dtype=np.float64)
        batch_y = np.zeros(seq_ids.size, dtype=np.int64)
        for ii, n in enumerate(seq_ids):
            start = self.seq_fp[n]
            stop = self.seq_fp[n+1]
            T = stop - start
            batch_x[ii, :T] = self.x_PF[start:stop]
            batch_y[ii] = self.y_N[n]

        if not to_pytorch_tensor:
            return batch_x, batch_y
        else:
            raise NotImplementedError("TODO convert to torch if needed")

    def __iter__(self, to_pytorch_tensor=False):
        for b in range(self.n_batch):
            x, y = self.get_batch_data(b, to_pytorch_tensor)
            yield x, y

if __name__ == '__main__':
    data_loader = TidySequentialDataCSVLoader('my_dataset.csv')

    for n in range(data_loader.n_sequences):
        X, y = data_loader.get_single_sequence_data(n)
        print("\n#### Sequence %d" % n)
        print("n_timesteps = %d" % (X.shape[1]))
        print("n_features = %d" % (X.shape[2]))
        print("X.shape = %s, y.shape = %s" % (str(X.shape), str(y.shape)))
        print("X:")
        print(X)
        print("y:")
        print(y)





