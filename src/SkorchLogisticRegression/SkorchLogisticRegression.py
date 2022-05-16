import skorch
import numpy as np
import torch
import sklearn.linear_model

from scipy.special import expit as logistic_sigmoid
from skorch.utils import TeeGenerator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score

from SkorchLogisticRegressionModule import SkorchLogisticRegressionModule


class SkorchLogisticRegression(skorch.NeuralNet):

    def __init__(
            self,
            n_features=0,
            l2_penalty_weights=0.0,
            l2_penalty_bias=0.0,
            clip=0.25,
            lr=1.00,
            batch_size=-1,
            max_epochs=100,
            criterion=torch.nn.NLLLoss,
            loss_name='cross_entropy_loss',
            min_precision=0.9,
            constraint_lambda=100,
            incremental_min_precision=False,
            tighten_bounds=False,
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
        self.gamma_fp = 7.00
        self.delta_fp = 0.03  
        self.m_fp = 6.85 
        self.b_fp = 1.59 
        self.gamma_tp =  7.00 
        self.delta_tp = 0.03 
        self.m_tp = 6.85 
        self.b_tp = -3.54 
        self.precision_ind = None
        self.tighten_bounds = tighten_bounds
        self.bounds_dict = dict(gamma_fp = [7.00, 6.00, 5.00, 4.00, 3.00],
                                delta_fp = [0.021, 0.018, 0.015, 0.012, 0.009],
                                m_fp = [8.264, 12.091, 18.897, 32.477, 64.85],
                                b_fp = [2.092, 2.426, 2.828, 3.336, 4.026],
                                gamma_tp = [7.00, 6.00, 5.00, 4.00, 3.00],
                                delta_tp = [0.035, 0.030, 0.025, 0.020, 0.015],
                                m_tp = [5.19, 6.192, 9.778, 17.009, 34.448],
                                b_tp = [-3.54, -3.646, -3.784, -3.97, -4.229])
        
        self.incremental_min_precision=incremental_min_precision
        self.initialization_gain=initialization_gain
        kwargs.update(dict(module=SkorchLogisticRegressionModule, 
                           module__n_features=n_features,
                           module__initialization_gain=initialization_gain,
                           criterion=criterion, 
                           lr=lr,           
                           batch_size=batch_size,
                           max_epochs=max_epochs))
        super(SkorchLogisticRegression, self).__init__(
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
            y_decision_scores_N1 = self.module_.linear_transform_layer.forward(torch.DoubleTensor(x_NF.dataset.X[indices]))
        elif isinstance(x_NF, skorch.dataset.Dataset):
            y_decision_scores_N1 = self.module_.linear_transform_layer.forward(torch.DoubleTensor(x_NF.X))
        else:
            y_decision_scores_N1 = self.module_.linear_transform_layer.forward(torch.DoubleTensor(x_NF))

        # classify y=1 if w'x+b>=0
        y_hat_N1 = y_decision_scores_N1.detach().numpy() >= 0
        return np.asarray(y_hat_N1, dtype=np.float64)


    def calc_bce_loss(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False):
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.linear_transform_layer.forward(X.type(torch.DoubleTensor))[:,0]

        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)

        weights_ = self.module_.linear_transform_layer.weight
        bias_ = self.module_.linear_transform_layer.bias
        
        
        ry_N = torch.sign(y_true_-0.01)*y_est_logits_
        cross_ent_loss = -1.0*torch.sum(torch.nn.functional.logsigmoid(ry_N))# add 1e-15 to avoid precision problems
        
    
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

    def calc_fp_tp_bounds_loose(self, y_true_, y_est_logits_):
        scores = 1.0-(torch.sign(y_true_-.001)*y_est_logits_)
        hinge_loss = torch.clamp(scores, min=0.0) # torch way for ag_np.max(0.0, scores)
        
        fp_upper_bound = torch.sum(hinge_loss[y_true_==0])
        tp_lower_bound = torch.sum(1-(hinge_loss[y_true_==1]))
        
        return fp_upper_bound, tp_lower_bound    
    
    def calc_fp_tp_bounds_better(self, y_true_, y_est_logits_):
        
        current_epoch = self.history[-1]['epoch']
        
        fp_upper_bound_N = (1+self.gamma_fp*self.delta_fp)*torch.sigmoid(self.m_fp*y_est_logits_+self.b_fp)
        fp_upper_bound = torch.sum(fp_upper_bound_N[y_true_==0])
        
        tp_lower_bound_N = (1+self.gamma_tp*self.delta_tp)*torch.sigmoid(self.m_tp*y_est_logits_+self.b_tp)
        tp_lower_bound = torch.sum(tp_lower_bound_N[y_true_==1])      
        
        return fp_upper_bound, tp_lower_bound
    
    
    def calc_fp_tp_bounds_ideal(self, y_true_, y_est_logits_):
        
        fp_upper_bound_N = (y_true_==0)&(y_est_logits_>=0)
        fp_upper_bound = torch.sum(fp_upper_bound_N)
        
        tp_lower_bound_N = (y_true_==1)&(y_est_logits_>=0)
#         tp_lower_bound = torch.sum(tp_lower_bound_N) 
        tp_lower_bound = torch.sum(self.delta_tp + tp_lower_bound_N)  
        
        
        return fp_upper_bound, tp_lower_bound
    
    
    def calc_surrogate_loss(self, y_true, y_est_logits_=None, X=None, return_y_logproba=False, 
                            alpha=0.8, lamb=1, bounds='tight'):
        
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.linear_transform_layer.forward(X.type(torch.DoubleTensor))[:,0]

        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)

        weights_ = self.module_.linear_transform_layer.weight
        bias_ = self.module_.linear_transform_layer.bias
        
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
                
                
        elif (self.loss_name == 'surrogate_loss_tight')&(not(self.incremental_min_precision))&(not(self.tighten_bounds)):
            
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                alpha=self.min_precision,
                                                                lamb=self.constraint_lambda, bounds='tight') 
            
        elif (self.loss_name == 'surrogate_loss_tight')&(self.tighten_bounds):
            
            # slowly tighten the bounds every num_epochs/4 (TODO : Change this to something more adaptive)
            current_epoch = self.history[-1]['epoch']
            n_increments = 4
            epochs_per_increment = int(self.max_epochs/n_increments)
            
            bound_ind = int(current_epoch/epochs_per_increment)
            if (current_epoch<self.max_epochs):
                self.gamma_fp = self.bounds_dict['gamma_fp'][bound_ind]
                self.delta_fp = self.bounds_dict['delta_fp'][bound_ind]
                self.m_fp = self.bounds_dict['m_fp'][bound_ind]
                self.b_fp = self.bounds_dict['b_fp'][bound_ind]
                self.gamma_tp = self.bounds_dict['gamma_tp'][bound_ind]
                self.delta_tp = self.bounds_dict['delta_tp'][bound_ind]
                self.m_tp = self.bounds_dict['m_tp'][bound_ind]
                self.b_tp = self.bounds_dict['b_tp'][bound_ind]
            
            if (current_epoch%epochs_per_increment==0):
                print('tightening bounds at epoch %d'%current_epoch)
            
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                    alpha=self.min_precision,
                                                    lamb=self.constraint_lambda, bounds='tight')
            
        
        elif (self.loss_name == 'surrogate_loss_loose'):
            
            loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                alpha=self.min_precision,
                                                                lamb=self.constraint_lambda, bounds='loose') 
            
        loss_.backward()
#         torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
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
            
            elif (self.loss_name == 'surrogate_loss_tight')&(not(self.incremental_min_precision))&(not(self.tighten_bounds)):

                loss_, y_logproba_ = self.calc_surrogate_loss(y, X=X, return_y_logproba=True,
                                                                    alpha=self.min_precision,
                                                                    lamb=self.constraint_lambda, bounds='tight') 
                
            elif (self.loss_name == 'surrogate_loss_tight')&(self.tighten_bounds):
                # slowly tighten the bounds every num_epochs/4 (TODO : Change this to something more adaptive)
            # slowly tighten the bounds every num_epochs/4 (TODO : Change this to something more adaptive)
                current_epoch = self.history[-1]['epoch']
                n_increments = 4
                epochs_per_increment = int(self.max_epochs/n_increments)

                bound_ind = int(current_epoch/epochs_per_increment)
                if (current_epoch<self.max_epochs):
                    self.gamma_fp = self.bounds_dict['gamma_fp'][bound_ind]
                    self.delta_fp = self.bounds_dict['delta_fp'][bound_ind]
                    self.m_fp = self.bounds_dict['m_fp'][bound_ind]
                    self.b_fp = self.bounds_dict['b_fp'][bound_ind]
                    self.gamma_tp = self.bounds_dict['gamma_tp'][bound_ind]
                    self.delta_tp = self.bounds_dict['delta_tp'][bound_ind]
                    self.m_tp = self.bounds_dict['m_tp'][bound_ind]
                    self.b_tp = self.bounds_dict['b_tp'][bound_ind]

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
                'y_pred' : y_pred_
                }


if __name__ == '__main__':
    N = 200   # n_examples
    F = 3   # n_features

    np.random.seed(0)
    torch.random.manual_seed(0)

    lr_clf = SkorchLogisticRegression(
        l2_penalty_weights=1e-10,
        l2_penalty_bias=0.0,
        n_features=F,
        lr=0.5,
        train_split=None,
        max_epochs=100)
    lr_clf.initialize()

    print("Weights:")
    print(lr_clf.module_.linear_transform_layer.weight)
    print("Bias:")
    print(lr_clf.module_.linear_transform_layer.bias)

    print("Random Data!")
    # Generate random data
    x_NF = np.random.randn(N, F)

    true_w_F = np.arange(F) + 1
    true_y_proba_N = logistic_sigmoid(np.dot(x_NF, true_w_F))
    
    true_y_N = np.asarray(
        true_y_proba_N >= np.random.rand(N),
        dtype=np.float64)



    clf = sklearn.linear_model.LogisticRegression(
        C=0.5/lr_clf.l2_penalty_weights,
        solver='lbfgs')
    clf.fit(x_NF, true_y_N)

    lr_clf.fit(x_NF, true_y_N)

    print("EST BY SKORCH w:")
    print(
        lr_clf.module_.linear_transform_layer.weight.detach().numpy())
    print(
        lr_clf.module_.linear_transform_layer.bias.detach().numpy())

    print("EST BY SKLEARN w:")
    print(clf.coef_.flatten())
    print(clf.intercept_)
    print("TRUE w:")
    print(true_w_F)


    '''
    y_proba_N2 = lr_clf.predict_proba(x_NF)
    for n in range(N):
        print("==== Example %d" % n)
        print("x[n=%d]" % n)
        print(x_NF[n])
        print("y_proba[n=%d]" % n)
        print(y_proba_N2[n])
    '''
     
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4]                  
    params = {'lr': [0.01, 0.02],  
           'module__l2_penalty_weights' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e2, 1e3, 1e4],
          }    
    roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    
    classifier = GridSearchCV(lr_clf, params, cv=5, scoring = roc_auc_scorer)
    classifier.fit(x_NF, true_y_N)
