import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score, make_scorer, precision_score, recall_score, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from yattag import Doc
import matplotlib.pyplot as plt
import sys
cwd = sys.path[0]
sys.path.append(os.path.dirname(cwd))
from feature_transformation import parse_feature_cols, parse_output_cols, parse_id_cols
from utils import load_data_dict_json
from pickle import dump
from split_dataset import Splitter
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
# from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sklearn Logistic Regression')

    parser.add_argument('--train_csv_files', type=str, required=True,
                        help='csv files for training')
    parser.add_argument('--test_csv_files', type=str, required=True,
                        help='csv files for testing')
    parser.add_argument('--valid_csv_files', type=str, default=None, required=False,
                        help='csv files for testing')
    parser.add_argument('--outcome_col_name', type=str, required=True,
                        help='outcome column name')  
    parser.add_argument('--output_dir', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--output_filename_prefix', type=str, required=True,
                        help='save dir of trained classifier and associated performance metrics') 
    parser.add_argument('--data_dict_files', type=str, required=True,
                        help='dict files for features and outcomes') 
    parser.add_argument('--validation_size', type=float, default=0.2, help='Validation size') 
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight Decay') 
    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
                        help='columns for splitter') 
    parser.add_argument('--n_splits', type=int, default=2)
    parser.add_argument('--n_estimators', type=int, default=25)
    parser.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False)


    args = parser.parse_args()
    
    # read the data dictionaries
    print('Reading train-test data...')
    
    # read the data dict JSONs and parse the feature and outcome columns
    x_data_dict_file, y_data_dict_file = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_data_dict_file)
    y_data_dict = load_data_dict_json(y_data_dict_file)
    
    feature_cols = parse_feature_cols(x_data_dict)
    key_cols = parse_id_cols(x_data_dict)
    
    df_by_split = dict()
    for split_name, csv_files in [
            ('train', args.train_csv_files.split(',')),
            ('test', args.test_csv_files.split(','))]:
        cur_df = None
        for csv_file in csv_files:

            # TODO use json data dict to load specific columns as desired types
            more_df =  pd.read_csv(csv_file)
            if cur_df is None:
                cur_df = more_df
            else:
                if args.merge_x_y:
                    cur_df = cur_df.merge(more_df, on=key_cols)
                else:
                    cur_df = pd.concat([cur_df, more_df], axis=1)
                    cur_df = cur_df.loc[:,~cur_df.columns.duplicated()]
        
        df_by_split[split_name] = cur_df
 
    outcome_col_name = args.outcome_col_name
    
    # Prepare data for classification
    try:
        x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    except KeyError:
        feature_cols = [col.replace('_to_', '-') for col in feature_cols]
        x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
    y_train = np.ravel(df_by_split['train'][outcome_col_name])
    
    x_test = df_by_split['test'][feature_cols].values.astype(np.float32)
    y_test = np.ravel(df_by_split['test'][outcome_col_name])        
    
    if args.valid_csv_files is None:
        # get the validation set
        splitter = Splitter(
            size=args.validation_size, random_state=41,
            n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
        # Assign training instances to splits by provided keys
        key_train = splitter.make_groups_from_df(df_by_split['train'][key_cols])


        # get the train and validation splits
        for ss, (tr_inds, va_inds) in enumerate(splitter.split(x_train, y_train, groups=key_train)):
            x_tr = x_train[tr_inds].copy()
            y_tr = y_train[tr_inds].copy()
            x_valid = x_train[va_inds]
            y_valid = y_train[va_inds]

        x_train = x_tr
        y_train = y_tr
        del(x_tr, y_tr)
    
    else:
        x_valid_csv, y_valid_csv = args.valid_csv_files.split(',')
        x_valid_df = pd.read_csv(x_valid_csv)
        y_valid_df = pd.read_csv(y_valid_csv)
        
        x_valid = x_valid_df[feature_cols].values.astype(np.float32)
        y_valid = np.ravel(y_valid_df[outcome_col_name])
        
    split_dict = {'N_train' : len(x_train),
                 'N_valid' : len(x_valid),
                 'N_test' : len(x_test),
                 'pos_frac_train' : y_train.sum()/len(y_train),
                 'pos_frac_valid' : y_valid.sum()/len(y_valid),
                 'pos_frac_test' : y_test.sum()/len(y_test)
                 }
    
    print(split_dict)
    
    
#     # normalize data
#     scaler = StandardScaler()
#     scaler.fit(x_train)
#     x_train_transformed = scaler.transform(x_train) 
#     x_valid_transformed = scaler.transform(x_valid)
#     x_test_transformed = scaler.transform(x_test) 
    
    
    # build pipeline
    step_list = list()
    scaler = StandardScaler()
    step_list.append(('standardize', scaler))
    
    fixed_precision = 0.2
    thr_list = [0.5]
    
    ## start training
    
    print('Training sklearn logistic regression...')
    clf = LogisticRegression(penalty='l2', C=args.weight_decay, fit_intercept=True, class_weight='balanced', 
                             random_state=41, verbose=1, solver='saga', max_iter=500, tol=2e-3)
    
    step_list.append(('classifier',clf))
    

    prediction_pipeline = Pipeline(step_list)
    prediction_pipeline.fit(x_train, y_train)
        
#     lr_clf = clf.fit(X=x_train_transformed, y=y_train)
    

    # search multiple decision thresolds and pick the threshold that performs best on validation set
    print('Searching thresholds that maximize recall at fixed precision %.4f'%fixed_precision)

    y_valid_proba_vals = prediction_pipeline.predict_proba(x_valid)
    unique_probas = np.unique(y_valid_proba_vals)
    thr_grid = np.linspace(np.percentile(unique_probas,1), np.percentile(unique_probas, 99), 100)

    precision_scores_G, recall_scores_G = [np.zeros(thr_grid.size), np.zeros(thr_grid.size)]
#     y_train_valid_pred_probas = prediction_pipeline.predict_proba(x_train_valid_transformed)
    for gg, thr in enumerate(thr_grid): 
        curr_thr_y_preds = y_valid_proba_vals[:,1]>=thr_grid[gg] 
        precision_scores_G[gg] = precision_score(y_valid, curr_thr_y_preds)
        recall_scores_G[gg] = recall_score(y_valid, curr_thr_y_preds) 
    
    keep_inds = precision_scores_G>=fixed_precision
    if keep_inds.sum()==0:
        fixed_precision = fixed_precision-0.05
        keep_inds = precision_scores_G>=fixed_precision
        print('Trying with precision %.2f'%fixed_precision)
    
    if keep_inds.sum()>0:
        precision_scores_G = precision_scores_G[keep_inds]
        recall_scores_G = recall_scores_G[keep_inds]
        thr_grid = thr_grid[keep_inds]
        best_ind = np.argmax(recall_scores_G)
        best_thr = thr_grid[best_ind]
        thr_list.append(best_thr)

        thr_perf_df = pd.DataFrame(np.vstack([
                    thr_grid[np.newaxis,:],
                    precision_scores_G[np.newaxis,:],
                    recall_scores_G[np.newaxis,:]]).T,
                columns=['thr', 'precision_score', 'recall_score'])
        print(thr_perf_df)
        print('Chosen threshold : '+str(best_thr))
    else:
        print('Could not find thresholds achieving fixed precision of %.2f. Evaluating with threshold 0.5'%fixed_precision)
        
 
        
    # save the scaler
    pickle_file = os.path.join(args.output_dir, 'scaler.pkl')
    dump(scaler, open(pickle_file, 'wb'))
    
    
    # print precision on train and validation
    f_out_name = os.path.join(args.output_dir, args.output_filename_prefix+'.txt')
    f_out = open(f_out_name, 'w')
    for thr in thr_list : 
        y_train_pred_probas = prediction_pipeline.predict_proba(x_train)[:,1]
        y_train_pred = y_train_pred_probas>=thr
        precision_train = precision_score(y_train, y_train_pred)
        recall_train = recall_score(y_train, y_train_pred)
        
        y_valid_pred_probas = prediction_pipeline.predict_proba(x_valid)[:,1]
        y_valid_pred = y_valid_pred_probas>=thr
        precision_valid = precision_score(y_valid, y_valid_pred)
        recall_valid = recall_score(y_valid, y_valid_pred)       
        
        y_test_pred_probas = prediction_pipeline.predict_proba(x_test)[:,1]
        y_test_pred = y_test_pred_probas>=thr
        precision_test = precision_score(y_test, y_test_pred)
        recall_test = recall_score(y_test, y_test_pred)
        
        threshold = thr
        TP_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 1))
        FP_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 1))
        TN_train = np.sum(np.logical_and(y_train == 0, y_train_pred == 0))
        FN_train = np.sum(np.logical_and(y_train == 1, y_train_pred == 0))
        
        TP_valid = np.sum(np.logical_and(y_valid == 1, y_valid_pred == 1))
        FP_valid = np.sum(np.logical_and(y_valid == 0, y_valid_pred == 1))
        TN_valid = np.sum(np.logical_and(y_valid == 0, y_valid_pred == 0))
        FN_valid = np.sum(np.logical_and(y_valid == 1, y_valid_pred == 0))          
        
        TP_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 1))
        FP_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 1))
        TN_test = np.sum(np.logical_and(y_test == 0, y_test_pred == 0))
        FN_test = np.sum(np.logical_and(y_test == 1, y_test_pred == 0))    

        print_st_tr = "LR Training performance at threshold %.5f : | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%( thr, precision_train, recall_train, TP_train, FP_train, TN_train, FN_train)
        print_st_va = "LR Validation performance at threshold %.5f : | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(thr, precision_valid, recall_valid, TP_valid, FP_valid, TN_valid, FN_valid)
        print_st_te = "LR Test performance  at threshold %.5f :  | Precision %.3f | Recall %.3f | TP %5d | FP %5d | TN %5d | FN %5d"%(thr, precision_test, recall_test,TP_test, FP_test, TN_test, FN_test) 

        print(print_st_tr)
        print(print_st_va)
        print(print_st_te)
        f_out.write(print_st_tr + '\n' + print_st_va + '\n' + print_st_te)
    f_out.close()
    
    
    perf_dict = {'N_train':len(x_train),
                'precision_train':precision_train,
                'recall_train':recall_train,
                'TP_train':TP_train,
                'FP_train':FP_train,
                'TN_train':TN_train,
                'FN_train':FN_train,
                'N_train':len(x_train),
                'precision_valid':precision_valid,
                'recall_valid':recall_valid,
                'TP_valid':TP_valid,
                'FP_valid':FP_valid,
                'TN_valid':TN_valid,
                'FN_valid':FN_valid,
                'N_valid':len(x_valid),
                'precision_test':precision_test,
                'recall_test':recall_test,
                'TP_test':TP_test,
                'FP_test':FP_test,
                'TN_test':TN_test,
                'FN_test':FN_test,
                'N_test':len(x_test),
                'threshold':threshold}
    perf_df = pd.DataFrame([perf_dict])
    perf_csv = os.path.join(args.output_dir, args.output_filename_prefix+'_perf.csv')
    print('Saving performance on train and test set to %s'%(perf_csv))
    perf_df.to_csv(perf_csv, index=False)
    
    # save model to disk
    model_file = os.path.join(args.output_dir, args.output_filename_prefix+'_trained_model.joblib')
    dump(prediction_pipeline, open(model_file, 'wb'))
    print('saved Logistic regression pickle model to : %s'%model_file)
    
    # save as onnx
    initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    onx = convert_sklearn(prediction_pipeline, initial_types=initial_type)
    onx_file = os.path.join(args.output_dir, args.output_filename_prefix+'.onnx')
    
    with open(onx_file, "wb") as f:
        f.write(onx.SerializeToString())
    
    print('saved Logistic regression onnx model to : %s'%onx_file)
    
    