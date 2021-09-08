import skorch
import numpy as np
import torch

from RNNPerTStepBinaryClassifierModule import RNNPerTStepBinaryClassifierModule
from skorch.utils import TeeGenerator

class RNNPerTStepBinaryClassifier(skorch.NeuralNet):

    def __init__(
            self,
            criterion=torch.nn.CrossEntropyLoss,
            clip=100.0,
            lr=1.00,
            *args,
            **kwargs,
            ):
        self.clip = clip
        kwargs.update(module=RNNPerTStepBinaryClassifierModule)
        super(RNNPerTStepBinaryClassifier, self).__init__(
            criterion=criterion, lr=lr, *args, **kwargs)

    def predict_proba(self, x_NTF):
        ''' Make binary predictions for each example in provided dataset
        Args
        ----
        x_NTF : 3D Numpy array-like, (n_sequences, n_timepoints, n_features)
            Each row is a feature vector
        Returns
        -------
        y_hat_NT2 : 3D array, (n_sequences, n_timepoints, 2)
            y_hat_NT[n, t, :] is the predicted class 0 and class 1 probabilities for the n-th example at time t, summing to 1
        '''
        self.module_.eval()
        
        if isinstance(x_NTF, torch.utils.data.dataset.Subset):
            indices = x_NTF.indices
            y_proba_NT2 = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF.dataset.X[indices])))
        elif isinstance(x_NTF, skorch.dataset.Dataset):
            y_proba_NT2 = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF.X)))
        else:
            y_proba_NT2 = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF)))      
        

        return y_proba_NT2

    def predict(self, x_NTF):
        ''' Make binary predictions for each example in provided dataset
        Args
        ----
        x_NTF : 3D Numpy array-like, (n_sequences, n_timepoints, n_features)
            Each row is a feature vector
        Returns
        -------
        y_hat_NT : 2D array, (n_sequences, n_timepoints)
            y_hat_NT[n, t] is the best guess of binary class label for the n-th example at time t
            Each entry is either 0.0 or 1.0
        '''
        self.module_.eval()
        
        thr=0.5
        if isinstance(x_NTF, torch.utils.data.dataset.Subset):
            indices = x_NTF.indices
            y_hat_NT = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF.dataset.X[indices])))[:,:,1]>=thr
        elif isinstance(x_NTF, skorch.dataset.Dataset):
            y_hat_NT = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF.X)))[:,:,1]>=thr
        else:
            y_hat_NT = torch.exp(self.module_.forward(torch.FloatTensor(x_NTF)))[:,:,1]>=thr
        
        # classify y=1 if p(y=1|x)>=0.5        
        return y_hat_NT


    def calc_bce_loss(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False):
        ''' Calculate BCE Loss for binary classification at each time-point

        Cleanly handles variable-length sequences (though internals a bit messy).

        Args
        ----
        y_true : 2D array (n_sequences, n_timesteps)
            Each row is one sequence, padded to length T = n_timesteps
        y_est_logits_ : 3D array (n_sequences, n_timesteps, 2)
            Entry y_est_logits_[n, t, :] indicates log probability of class 0 and class 1 of the n-th sequence at time t 

        Returns
        -------
        cross_ent_loss : loss for nth sequence = - sum_t(log(p(y_nt | yhat_nt))) = -sum_t(y_nt*log(yhat_nt) + (1-y_nt)*log(1-yhat_nt)), where t is the length of the n-th sequence, and yhat_nt is p(y_nt)=1 
        '''
        
        
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.forward(X)
        
        #consider only estimated logits until the sequence length per example
        keep_inds = torch.logical_not(torch.all(torch.isnan(X), dim=-1))
        seq_lens_N = torch.sum(keep_inds, axis=1)
        
#         cross_ent_loss_per_example_N = torch.sum(-y_est_logits_[:,:,0]*y_true - y_est_logits_[:,:,1]*(1-y_true), dim=1)
        cross_ent_loss_per_example_N = torch.sum(-y_est_logits_[:,:,1]*y_true*keep_inds - y_est_logits_[:,:,0]*(1-y_true)*keep_inds,
                                                 dim=1)/seq_lens_N
        
        cross_ent_loss = torch.sum(cross_ent_loss_per_example_N)
    
        denom = y_true.shape[0]
        
        ## TODO Add the l2 norm on weights and bias
        loss_ = (cross_ent_loss
#             + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
#             + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom * np.log(2.0))
        
        if return_y_logproba:
            return loss_, y_est_logits_
        else:
            return loss_
    

    def train_step_single(self, X, y, **fit_params):
        ''' Perform one gradient descent update step on provided batch (X,y)
        Returns
        -------
        info_dict : dict
            Contains summary metrics and some intermediate predicted values for debugging.
        '''
        self.module_.train()
        self.optimizer_.zero_grad()
        
        loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)

        loss_.backward()
        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=X,
            y=y,
        )
        
#         y_pred_ = self.predict(X.type(torch.DoubleTensor))
        y_pred_ = self.predict(X)
        
        return {
            'loss': loss_,
            'y_pred' : y_pred_,
            'y_logproba' : y_logproba_
            }


    def validation_step(self, X, y, **fit_params):
        ''' Perform one evaluation step on provided batch (X,y)
        Returns
        -------
        info_dict : dict
            Contains summary metrics and some intermediate predicted values for debugging.
        '''
        self.module_.eval()

        with torch.no_grad():
            loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)

                
            y_pred_ = self.predict(X)
            
            return {
                'loss': loss_,
                'y_pred' : y_pred_,
                'y_logproba' : y_logproba_
                }
