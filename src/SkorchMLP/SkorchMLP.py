import skorch
import numpy as np
import torch
import sklearn.linear_model

from scipy.special import expit as logistic_sigmoid
from skorch.utils import TeeGenerator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

from SkorchMLPModule import SkorchMLPModule


class SkorchMLP(skorch.NeuralNet):

    def __init__(
            self,
            n_features=0,
            n_hiddens=32,
            n_layers=2,
            dropout=0.0,
            l2_penalty_weights=0.0,
            l2_penalty_bias=0.0,
            clip=0.25,
            lr=1.00,
            batch_size=-1,
            max_epochs=100,
            criterion=torch.nn.NLLLoss,
            min_precision=0.9,
            constraint_lambda=100,
            loss_name='cross_entropy_loss',
            incremental_min_precision=False,
            initialization_gain=1.0,
            *args,
            **kwargs
            ):
        ## Fill in keyword arguments
        self.n_features = n_features
        self.l2_penalty_weights = l2_penalty_weights
        self.l2_penalty_bias = l2_penalty_bias
        self.clip = clip
        self.loss_name=loss_name
        self.min_precision = min_precision
        self.constraint_lambda=constraint_lambda
        self.gamma_fp = 7.00 #2.001 #4.00 #1.001
        self.delta_fp = 0.021 #0.006 #0.012 #0.003
        self.m_fp = 8.264 #170.73 #32.47  #1231.95
        self.b_fp = 2.092 #5.12 #3.33  #12.708
        self.gamma_tp =  7.00 #2.001 #4.00  #1.001
        self.delta_tp = 0.035 #0.01 #0.02  #0.005
        self.m_tp = 5.19 #92.3 #17.00 #680.072
        self.b_tp = -3.54 #-4.61 #-3.97 #-5.297
        self.incremental_min_precision=incremental_min_precision
        self.initialization_gain=initialization_gain
        self.precision_ind = None
        self.n_layers=n_layers
        kwargs.update(dict(module=SkorchMLPModule, 
                           module__n_features=n_features, 
                           module__n_hiddens=n_hiddens,
                           module__n_layers=n_layers,
                           module__initialization_gain=initialization_gain,
                           module__dropout=dropout,
                           criterion=criterion, 
                           lr=lr,           
                           batch_size=batch_size,
                           max_epochs=max_epochs))
        super(SkorchMLP, self).__init__(
            *args, **kwargs)
#         self.initialize()

    def predict_proba(self, x_NF):
        ''' Make probabilistic predictions for each example in provided dataset
        Args
        ----
        x_NF : 2D Numpy array-like, (n_examples, n_features)
        Returns
        -------
        y_proba_N2 : 2D array, (n_examples, 2)
        '''
        self.module_.eval()
        
        if isinstance(x_NF, skorch.dataset.Dataset):
            y_logproba_ = self.module_.forward(torch.DoubleTensor(x_NF.X))
        else:
            y_logproba_ = self.module_.forward(torch.DoubleTensor(x_NF))
        
        y_logproba_N1 = y_logproba_.detach().numpy()
        y_proba_N2 = np.empty((len(y_logproba_N1), 2))
        y_proba_N2[:,1] = np.exp(y_logproba_N1[:,0])
        y_proba_N2[:,0] = 1 - y_proba_N2[:,1]
        return y_proba_N2

    def predict(self, x_NF):
        ''' Make binary predictions for each example in provided dataset
        Args
        ----
        x_NF : 2D Numpy array-like, (n_examples, n_features)
            Each row is a feature vector
        Returns
        -------
        y_hat_N : 2D array, (n_examples, 1)
            Best guess of binary class label for each example
            Each row is either 0.0 or 1.0
        '''
        self.module_.eval()
        if isinstance(x_NF, torch.utils.data.dataset.Subset):
            indices = x_NF.indices
            y_decision_scores_N1 = self.module_.forward(torch.DoubleTensor(x_NF.dataset.X[indices]), apply_logsigmoid=False)
        elif isinstance(x_NF, skorch.dataset.Dataset):
            y_decision_scores_N1 = self.module_.forward(torch.DoubleTensor(x_NF.X), apply_logsigmoid=False)
        else:
            y_decision_scores_N1 = self.module_.forward(torch.DoubleTensor(x_NF), apply_logsigmoid=False)
        
        # classify y=1 if w'x+b>=0
        y_hat_N1 = y_decision_scores_N1.detach().numpy() >= 0
        return np.asarray(y_hat_N1, dtype=np.float64)


    def calc_bce_loss(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False):
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.forward(X.type(torch.DoubleTensor), apply_logsigmoid=False)[:,0]

        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)
        
        weights_arr = []
        bias_arr = []
        for layer_num in range(self.n_layers):
            weights_arr.append(self.module_.hidden_layer._modules['fc_%d'%layer_num].weight.flatten())
            bias_arr.append(self.module_.hidden_layer._modules['fc_%d'%layer_num].bias)
        
        
#         weights_ = torch.cat([self.module_.hidden_layer.weight.flatten(), self.module_.output_layer.weight.flatten()])
#         bias_ = torch.cat([self.module_.hidden_layer.bias, self.module_.output_layer.bias])
        
        weights_ = torch.cat(weights_arr)
        bias_ = torch.cat(bias_arr)
        
        ry_N = torch.sign(y_true_-0.01)*y_est_logits_
        cross_ent_loss = -1.0*torch.sum(torch.nn.functional.logsigmoid(ry_N))
        
    
        denom = y_true_.shape[0]
    
        loss_ = (cross_ent_loss
            + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
            + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom * np.log(2.0))
        
        if return_y_logproba:
            yproba_ = torch.nn.functional.logsigmoid(y_est_logits_)
            return loss_, yproba_
        else:
            return loss_
    
    def calc_fp_tp_bounds(self, y_true_, y_est_logits_):
        sigmoid_loss = 1.2*torch.sigmoid(-((torch.sign(y_true_-.001)*8*y_est_logits_))+1.8)
        fp_upper_bound = torch.sum(sigmoid_loss[y_true_==0])
        tp_lower_bound = torch.sum(1-(sigmoid_loss[y_true_==1]))
        
        return fp_upper_bound, tp_lower_bound
    
    def calc_fp_tp_bounds_better(self, y_true_, y_est_logits_):
        
        fp_upper_bound_N = (1+self.gamma_fp*self.delta_fp)*torch.sigmoid(self.m_fp*y_est_logits_+self.b_fp)
        fp_upper_bound = torch.sum(fp_upper_bound_N[y_true_==0])
        
        tp_lower_bound_N = (1+self.gamma_tp*self.delta_tp)*torch.sigmoid(self.m_tp*y_est_logits_+self.b_tp)
        tp_lower_bound = torch.sum(tp_lower_bound_N[y_true_==1])      
        
        return fp_upper_bound, tp_lower_bound
    
    def calc_fp_tp_bounds_loose(self, y_true_, y_est_logits_):
        scores = 1.0-(torch.sign(y_true_-.001)*y_est_logits_)
        hinge_loss = torch.clamp(scores, min=0.0) # torch way for ag_np.max(0.0, scores)
        
        fp_upper_bound = torch.sum(hinge_loss[y_true_==0])
        tp_lower_bound = torch.sum(1-(hinge_loss[y_true_==1]))
        
        return fp_upper_bound, tp_lower_bound  
    
    
    def calc_fp_tp_bounds_ideal(self, y_true_, y_est_logits_):
        
        fp_upper_bound_N = (y_true_==0)&(y_est_logits_>=0)
        fp_upper_bound = torch.sum(fp_upper_bound_N)
        
        tp_lower_bound_N = (y_true_==1)&(y_est_logits_>=0)
#         tp_lower_bound = torch.sum(tp_lower_bound_N) 
        tp_lower_bound = torch.sum(self.delta_tp + tp_lower_bound_N)  
        
        
        return fp_upper_bound, tp_lower_bound
    
    
    def calc_surrogate_loss(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False, alpha=0.85, lamb=1, bounds='tight'):
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.forward(X.type(torch.DoubleTensor), apply_logsigmoid=False)[:,0]

        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)

        weights_arr = []
        bias_arr = []
        for layer_num in range(self.n_layers):
            weights_arr.append(self.module_.hidden_layer._modules['fc_%d'%layer_num].weight.flatten())
            bias_arr.append(self.module_.hidden_layer._modules['fc_%d'%layer_num].bias)
        
        
#         weights_ = torch.cat([self.module_.hidden_layer.weight.flatten(), self.module_.output_layer.weight.flatten()])
#         bias_ = torch.cat([self.module_.hidden_layer.bias, self.module_.output_layer.bias])
        
        weights_ = torch.cat(weights_arr)
        bias_ = torch.cat(bias_arr)
        
        if bounds=='tight':
            fp_upper_bound, tp_lower_bound = self.calc_fp_tp_bounds_better(y_true_, y_est_logits_)
        elif bounds=='loose':
            fp_upper_bound, tp_lower_bound = self.calc_fp_tp_bounds_loose(y_true_, y_est_logits_)
        
        frac_alpha = alpha/(1-alpha)
#         surr_loss_tight = -tp_lower_bound*(1+lamb) + lamb*frac_alpha*fp_upper_bound
        g_theta = -tp_lower_bound + frac_alpha*fp_upper_bound
        
        if g_theta>=0:
            penalty = g_theta
        else:
            penalty = 0
        surr_loss_tight = -tp_lower_bound + lamb*penalty        
        denom = y_true_.shape[0]
        loss_ = (
            surr_loss_tight
            + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
            + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom)

        if return_y_logproba:
            yproba_ = torch.nn.functional.logsigmoid(y_est_logits_)
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

#         loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)
        
        if self.loss_name == 'cross_entropy_loss':
            loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)
        elif (self.loss_name == 'surrogate_loss_tight')&(self.incremental_min_precision):
            current_epoch = self.history[-1]['epoch']
            # slowly increment the min precision over epochs
            n_increments = 4
            epochs_per_increment = int(self.max_epochs/n_increments)
            min_precision_grid = np.arange(self.min_precision/n_increments, 
                                           self.min_precision+self.min_precision/n_increments, 
                                           self.min_precision/n_increments)
            min_precision_increment_epochs_grid = np.arange(epochs_per_increment, self.max_epochs+epochs_per_increment, epochs_per_increment)
            
            if (current_epoch>1)&(self.precision_ind==None):#handles case when weights are transferred from BCE
                self.precision_ind = len(min_precision_grid)-1
                print('Min precision at epoch %.1f set to %.4f'%(current_epoch, min_precision_grid[self.precision_ind]))
            elif current_epoch==1:
                self.precision_ind = 0
                print('Starting with min precision = %.1f'%min_precision_grid[self.precision_ind])
            elif (current_epoch in min_precision_increment_epochs_grid)&(self.precision_ind<len(min_precision_grid)-1):
                self.precision_ind = np.searchsorted(min_precision_increment_epochs_grid, current_epoch)+1
                print('Min precision at epoch %.1f set to %.4f'%(current_epoch, min_precision_grid[self.precision_ind]))

            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                alpha=min_precision_grid[self.precision_ind],
                                                                lamb=self.constraint_lambda, bounds='tight')                
                
                
        elif (self.loss_name == 'surrogate_loss_tight')&(not(self.incremental_min_precision)):
            
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                alpha=self.min_precision,
                                                                lamb=self.constraint_lambda, bounds='tight') 
        elif (self.loss_name == 'surrogate_loss_loose'):
            
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                alpha=self.min_precision,
                                                                lamb=self.constraint_lambda, bounds='loose') 
            
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=X,
            y=y,
        )
        
        y_pred_ = self.predict(X.type(torch.DoubleTensor))
        
        return {
            'loss': loss_,
            'y_pred' : y_pred_,
            'y_logproba_': y_logproba_
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
            if self.loss_name == 'cross_entropy_loss':
                loss_, y_logproba_ = self.calc_bce_loss(y, X=X, return_y_logproba=True)
            elif (self.loss_name == 'surrogate_loss_tight')&(self.incremental_min_precision):
                current_epoch = self.history[-1]['epoch']
                # slowly increment the min precision over epochs
                n_increments = 4
                epochs_per_increment = int(self.max_epochs/n_increments)
                min_precision_grid = np.arange(self.min_precision/n_increments, 
                                               self.min_precision+self.min_precision/n_increments, 
                                               self.min_precision/n_increments)
                min_precision_increment_epochs_grid = np.arange(epochs_per_increment, self.max_epochs+epochs_per_increment, epochs_per_increment)

                if (current_epoch>1)&(self.precision_ind==None):#handles case when weights are transferred from BCE
                    self.precision_ind = len(min_precision_grid)-1
                    print('Min precision at epoch %.1f set to %.4f'%(current_epoch, min_precision_grid[self.precision_ind]))
                elif current_epoch==1:
                    self.precision_ind = 0
                    print('Starting with min precision = %.1f'%min_precision_grid[self.precision_ind])
                elif (current_epoch in min_precision_increment_epochs_grid)&(self.precision_ind<len(min_precision_grid)-1):
                    self.precision_ind = np.searchsorted(min_precision_increment_epochs_grid, current_epoch)+1
                    print('Min precision at epoch %.1f set to %.4f'%(current_epoch, min_precision_grid[self.precision_ind]))

                loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                    alpha=min_precision_grid[self.precision_ind],
                                                                    lamb=self.constraint_lambda, bounds='tight')                


            elif (self.loss_name == 'surrogate_loss_tight')&(not(self.incremental_min_precision)):

                loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                    alpha=self.min_precision,
                                                                    lamb=self.constraint_lambda, bounds='tight') 
            elif self.loss_name == 'surrogate_loss_loose':
                loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True, 
                                                                alpha=self.min_precision, lamb=self.constraint_lambda,
                                                                   bounds='loose')
                
            y_pred_ = self.predict(X.type(torch.DoubleTensor))
            
            return {
                'loss': loss_,
                'y_pred' : y_pred_,
                'y_logproba_': y_logproba_
                }

if __name__ == '__main__':
    N = 200   # n_examples
    F = 3   # n_features

    np.random.seed(0)
    torch.random.manual_seed(0)

    mlp_clf = SkorchMLP(
        l2_penalty_weights=1e-10,
        l2_penalty_bias=0.0,
        n_features=F,
        dropout=0.0,
        n_hiddens=32,
        lr=0.5,
        train_split=None,
        max_epochs=100, 
        loss_name='cross_entropy_loss')
    


    print("Random Data!")
    # Generate random data
    x_NF = np.random.randn(N, F)

    true_w_F = np.arange(F) + 1
    true_y_proba_N = logistic_sigmoid(np.dot(x_NF, true_w_F))
    
    true_y_N = np.asarray(
        true_y_proba_N >= np.random.rand(N),
        dtype=np.float64)


    # fit classifier
    mlp_clf.fit(x_NF, true_y_N)