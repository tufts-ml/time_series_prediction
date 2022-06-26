import skorch
import numpy as np
import torch

from RNNPerTStepBinaryClassifierModule import RNNPerTStepBinaryClassifierModule
from skorch.utils import TeeGenerator

class RNNPerTStepBinaryClassifier(skorch.NeuralNet):

    def __init__(
            self,
            criterion=torch.nn.CrossEntropyLoss,
            l2_penalty_weights=0.0,
            min_precision=0.7,
            constraint_lambda=10000,
            clip=100.0,
            scoring='cross_entropy_loss',

            lr=1.00,
            *args,
            **kwargs,
            ):
        self.clip = clip
        self.scoring=scoring
        self.min_precision = min_precision
        self.constraint_lambda=constraint_lambda
        self.l2_penalty_weights=l2_penalty_weights
        self.gamma_fp = 7.00 
        self.delta_fp = 0.021 
        self.m_fp = 8.264 
        self.b_fp = 2.092 
        self.gamma_tp =  7.00 
        self.delta_tp = 0.035 
        self.m_tp = 5.19 
        self.b_tp = -3.54 

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
        
        cross_ent_loss_per_example_N = torch.sum(-y_est_logits_[:,:,1]*y_true[:, :max(seq_lens_N)]*keep_inds[:, :max(seq_lens_N)] - y_est_logits_[:,:,0]*(1-y_true[:, :max(seq_lens_N)])*keep_inds[:, :max(seq_lens_N)], dim=1)/seq_lens_N  
        
        cross_ent_loss = torch.sum(cross_ent_loss_per_example_N)
        
        denom = y_true.shape[0]
        
        weights_ = torch.cat([self.module_.rnn.weight_ih_l0.flatten(), 
                              self.module_.rnn.weight_hh_l0.flatten(), 
                              self.module_.output.weight.flatten()])
#         bias_ = torch.cat([self.module_.hidden_layer.bias, self.module_.output_layer.bias])
        
        
        ## TODO Add the l2 norm on weights and bias
        loss_ = (cross_ent_loss
            + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))

#             + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom * np.log(2.0))
        
        if return_y_logproba:
            return loss_, y_est_logits_
        else:
            return loss_
    
    
    def calc_fp_tp_bounds_better(self, y_true_, y_est_logits_, keep_inds):
                
        fp_upper_bound_N = (1+self.gamma_fp*self.delta_fp)*torch.sigmoid(self.m_fp*y_est_logits_[:, :, 1][keep_inds]+self.b_fp)
        fp_upper_bound = torch.sum(fp_upper_bound_N[y_true_[keep_inds]==0])
        
        tp_lower_bound_N = (1+self.gamma_tp*self.delta_tp)*torch.sigmoid(self.m_tp*y_est_logits_[:, :, 1][keep_inds]+self.b_tp)
        tp_lower_bound = torch.sum(tp_lower_bound_N[y_true_[keep_inds]==1])      
        
        return fp_upper_bound, tp_lower_bound
    
    def calc_fp_tp_bounds_loose(self, y_true_, y_est_logits_, keep_inds):
        scores = 1.0-(torch.sign(y_true_[keep_inds]-.001)*y_est_logits_[:, :, 1][keep_inds])
        hinge_loss = torch.clamp(scores, min=0.0) # torch way for ag_np.max(0.0, scores)
        
        fp_upper_bound = torch.sum(hinge_loss[y_true_[keep_inds]==0])
        tp_lower_bound = torch.sum(1-(hinge_loss[y_true_[keep_inds]==1]))
        
        return fp_upper_bound, tp_lower_bound  
    
    
    def calc_fp_tp_bounds_ideal(self, y_true_, y_est_logits_):
        
        fp_upper_bound_N = (y_true_[keep_inds]==0)&(y_est_logits_[:, :, 1][keep_inds]>=0)
        fp_upper_bound = torch.sum(fp_upper_bound_N)
        
        tp_lower_bound_N = (y_true_[keep_inds]==1)&(y_est_logits_[:, :, 1][keep_inds]>=0)
#         tp_lower_bound = torch.sum(tp_lower_bound_N) 
        tp_lower_bound = torch.sum(self.delta_tp + tp_lower_bound_N)  
        
        
        return fp_upper_bound, tp_lower_bound
    
    
    def calc_surrogate_loss(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False, alpha=0.85, lamb=1, bounds='tight'):
        
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.forward(X, apply_log_softmax=False)
        
        #consider only estimated logits until the sequence length per example
        keep_inds = torch.logical_not(torch.all(torch.isnan(X), dim=-1))
        seq_lens_N = torch.sum(keep_inds, axis=1)
        
        
        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)
        
        # handle variable length sequences
        y_true_ = y_true[:, :max(seq_lens_N)]
        keep_inds = keep_inds[:, :max(seq_lens_N)]
        
#         weights_ = torch.cat([self.module_.hidden_layer.weight.flatten(), self.module_.output_layer.weight.flatten()])
#         bias_ = torch.cat([self.module_.hidden_layer.bias, self.module_.output_layer.bias])
        
        if bounds=='tight':
            fp_upper_bound, tp_lower_bound = self.calc_fp_tp_bounds_better(y_true_, y_est_logits_, keep_inds)
        elif bounds=='loose':
            fp_upper_bound, tp_lower_bound = self.calc_fp_tp_bounds_loose(y_true_, y_est_logits_, keep_inds)
        
        frac_alpha = alpha/(1-alpha)
#         surr_loss_tight = -tp_lower_bound*(1+lamb) + lamb*frac_alpha*fp_upper_bound
        g_theta = -tp_lower_bound + frac_alpha*fp_upper_bound
        
        if g_theta>=0:
            penalty = g_theta
        else:
            penalty = 0
        surr_loss_tight = -tp_lower_bound + lamb*penalty        
        denom = y_true_.shape[0]
        
        weights_ = torch.cat([self.module_.rnn.weight_ih_l0.flatten(), 
                              self.module_.rnn.weight_hh_l0.flatten(), 
                              self.module_.output.weight.flatten()])        
        
        loss_ = (
            surr_loss_tight
            + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
#             + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom)
        
        if return_y_logproba:
            yproba_ = torch.nn.functional.log_softmax(y_est_logits_, dim=-1)
            return loss_, yproba_
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
        
        if self.scoring=='cross_entropy_loss':
            loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)
        elif self.scoring=='surrogate_loss_tight':
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True, 
                                                          alpha=self.min_precision, 
                                                          lamb=self.constraint_lambda,
                                                          bounds='tight')


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
            if self.scoring=='cross_entropy_loss':
                loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)
            elif self.scoring=='surrogate_loss_tight':
                loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True, 
                                                              alpha=self.min_precision, 
                                                              lamb=self.constraint_lambda,
                                                              bounds='tight')


                
            y_pred_ = self.predict(X)
            
            return {
                'loss': loss_,
                'y_pred' : y_pred_,
                'y_logproba' : y_logproba_
                }

