import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.metrics import (roc_auc_score, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import sys
cwd = sys.path[0]
sys.path.append(os.path.dirname(cwd))
from utils import load_data_dict_json
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sklearn lightgbm')

# Optimization options
    parser.add_argument('--train_np_files', type=str, required=True)
    parser.add_argument('--test_np_files', type=str, required=True)
    parser.add_argument('--valid_np_files', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--output_filename_prefix', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics')


    args = parser.parse_args()
    
    # read the data dictionaries
    print('Reading train-test data...')
    

    x_train_np_filename, y_train_np_filename = args.train_np_files.split(',')
    x_test_np_filename, y_test_np_filename = args.test_np_files.split(',')
    x_valid_np_filename, y_valid_np_filename = args.valid_np_files.split(',')
    
    
    X_train = np.load(x_train_np_filename)
    X_test = np.load(x_test_np_filename)
    y_train = np.load(y_train_np_filename)
    y_test = np.load(y_test_np_filename)
    X_valid = np.load(x_valid_np_filename)
    y_valid = np.load(y_valid_np_filename)
    
    # get only the labelled data
    labelled_inds_tr = ~np.isnan(y_train)
    labelled_inds_va = ~np.isnan(y_valid)
    labelled_inds_te = ~np.isnan(y_test)
    
    X_train_labelled = X_train[labelled_inds_tr].copy()
    y_train_labelled = y_train[labelled_inds_tr]
    
    X_valid_labelled = X_valid[labelled_inds_va]
    y_valid_labelled = y_valid[labelled_inds_va]  
    
    
    X_test_labelled = X_test[labelled_inds_te]
    y_test_labelled = y_test[labelled_inds_te]  
    
    
    # build pipeline
    step_list = list()
    scaler = StandardScaler()
    step_list.append(('standardize', scaler))
    
    ## start training
#     hyperparameters = dict(n_estimators=[64],
#                            max_features=[0.66],
#                            min_samples_leaf=[512],
#                            max_leaf_nodes=[32])
    
    hyperparameters = dict(n_estimators=[64, 100],
                           max_features=[0.33, 0.66],
                           min_samples_leaf=[128, 512, 1024],
                           max_leaf_nodes=[32, 128])


    # grid search
    forest = RandomForestClassifier(class_weight='balanced')
    classifier = GridSearchCV(forest, hyperparameters, cv=5, 
                              verbose=5, scoring=['roc_auc', 'average_precision'],
                             refit='average_precision')
    
#     print('Training lightGBM...')
#     clf = lgb.LGBMClassifier(learning_rate=0.05, class_weight='balanced', subsample_for_bin=200000, 
#                                   min_data_in_leaf=args.min_samples_per_leaf, num_leaves=args.max_leaves,
#                                   feature_fraction=args.frac_features_for_clf,
#                              bagging_fraction=args.frac_training_samples_per_tree,
#                                   bagging_freq=1, first_metric_only=True, num_iterations=500)
    
    step_list.append(('classifier',classifier))
    prediction_pipeline = Pipeline(step_list)
    prediction_pipeline.fit(X_train_labelled, y_train_labelled) 
           
    # save the scaler
    pickle_file = os.path.join(args.output_dir, 'scaler.pkl')
    dump(scaler, open(pickle_file, 'wb'))
    
    y_train_pred_probas = prediction_pipeline.predict_proba(X_train_labelled)[:,1]
    auprc_train = average_precision_score(y_train_labelled, y_train_pred_probas)
    auroc_train = roc_auc_score(y_train_labelled, y_train_pred_probas)

    y_valid_pred_probas = prediction_pipeline.predict_proba(X_valid_labelled)[:,1]
    auprc_valid = average_precision_score(y_valid_labelled, y_valid_pred_probas)
    auroc_valid = roc_auc_score(y_valid_labelled, y_valid_pred_probas)    

    y_test_pred_probas = prediction_pipeline.predict_proba(X_test_labelled)[:,1]
    auprc_test = average_precision_score(y_test_labelled, y_test_pred_probas)
    auroc_test = roc_auc_score(y_test_labelled, y_test_pred_probas)     
    
    
    final_perf_save_file = os.path.join(args.output_dir, 'final_perf_'+args.output_filename_prefix+'.csv')
    final_model_perf_df = pd.DataFrame([{'train_AUPRC' : auprc_train, 
                                       'valid_AUPRC' : auprc_valid, 
                                       'test_AUPRC' : auprc_test,
                                       'train_AUROC' : auroc_train, 
                                       'valid_AUROC' : auroc_valid, 
                                       'test_AUROC' : auroc_test}])
    print(final_model_perf_df)
    final_model_perf_df.to_csv(final_perf_save_file, index=False)
    
    # save model to disk
    model_file = os.path.join(args.output_dir, args.output_filename_prefix+'_trained_model.joblib')
    dump(prediction_pipeline, open(model_file, 'wb'))
    print('saved lightGBM model to : %s'%model_file)
    
    

 