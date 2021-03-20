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
        kwargs.update(dict(module=SkorchLogisticRegressionModule, 
                           module__n_features=n_features, 
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
        y_logproba_ = self.module_.forward(torch.DoubleTensor(x_NF))
        y_logproba_N1 = y_logproba_.detach().numpy()
        y_proba_N2 = np.empty((x_NF.shape[0], 2))
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
#         loss_ = (
#             torch.nn.functional.binary_cross_entropy_with_logits(
#                 y_est_logits_, y_true_) * denom
#             + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
#             + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
#             ) / (denom * np.log(2.0))
        
    
        loss_ = (cross_ent_loss
            + self.l2_penalty_weights * torch.sum(torch.mul(weights_, weights_))
            + self.l2_penalty_bias * torch.sum(torch.mul(bias_, bias_))
            ) / (denom * np.log(2.0))
        
        if return_y_logproba:
            yproba_ = torch.nn.functional.logsigmoid(y_est_logits_)
            return loss_, yproba_
        else:
            return loss_

    def calc_surrogate_loss_tight(
            self, y_true, y_est_logits_=None, X=None, return_y_logproba=False, alpha=0.8, lamb=1):
        if y_est_logits_ is None:
            y_est_logits_ = self.module_.linear_transform_layer.forward(X.type(torch.DoubleTensor))[:,0]

        if isinstance(y_true, torch.Tensor):
            # Make sure we cast from possible Float to Double
            y_true_ = y_true.type(torch.DoubleTensor)
        else:
            y_true_ = torch.DoubleTensor(y_true)

        weights_ = self.module_.linear_transform_layer.weight
        bias_ = self.module_.linear_transform_layer.bias
        
        sigmoid_loss = 1.2*torch.sigmoid(-((torch.sign(y_true_-.001)*8*y_est_logits_))+1.8)
        fp_upper_bound = torch.sum(sigmoid_loss[y_true_==0])
        tp_lower_bound = torch.sum(1-(sigmoid_loss[y_true_==1]))
        
        frac_alpha = alpha/(1-alpha)
        surr_loss_tight = -tp_lower_bound*(1+lamb) + lamb*frac_alpha*fp_upper_bound
        
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
        elif self.loss_name == 'surrogate_loss_tight':
            loss_, y_logproba_ = self.calc_surrogate_loss_tight(y, X=X, return_y_logproba=True, 
                                                                alpha=self.min_precision, lamb=self.constraint_lambda)
            
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
            elif self.loss_name == 'surrogate_loss_tight':
                loss_, y_logproba_ = self.calc_surrogate_loss_tight(y, X=X, return_y_logproba=True)
                
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