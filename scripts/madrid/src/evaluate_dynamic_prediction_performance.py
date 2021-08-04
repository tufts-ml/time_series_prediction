import sys, os
import argparse
import numpy as np 
import pandas as pd
import json
import time
import skorch
import sys
import glob
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, accuracy_score, log_loss, 
                            balanced_accuracy_score, confusion_matrix, 
                            roc_auc_score, make_scorer, precision_score, recall_score,
                            average_precision_score, precision_recall_curve)
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchLogisticRegression'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'SkorchMLP'))

from sklearn.preprocessing import StandardScaler
from SkorchLogisticRegression import SkorchLogisticRegression
from SkorchMLP import SkorchMLP
import matplotlib.pyplot as plt

from feature_transformation import parse_feature_cols, parse_output_cols, parse_id_cols
from utils import load_data_dict_json
from pickle import load
from split_dataset import Splitter
from skorch.dataset import Dataset
from skorch.helper import predefined_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating dynamic performance of all models')

    parser.add_argument('--train_csv_files', type=str, required=True,
                        help='csv files for training')
    parser.add_argument('--test_csv_files', type=str, required=True,
                        help='csv files for testing')
    parser.add_argument('--valid_csv_files', type=str, default=None, required=False,
                        help='csv files for testing')
    parser.add_argument('--clf_models_dir', type=str, required=True,
                        help='directory where clf models are saved')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory where model performances are saved')
    parser.add_argument('--outcome_col_name', type=str, required=True,
                        help='outcome column name')  
    parser.add_argument('--data_dict_files', type=str, required=True,
                        help='dict files for features and outcomes') 
    parser.add_argument('--merge_x_y', default=True,
                                type=lambda x: (str(x).lower() == 'true'), required=False)
    parser.add_argument('--validation_size', type=float, help='Validation size', default=0.2) 
    parser.add_argument('--key_cols_to_group_when_splitting', type=str,
                        help='columns for splitter', default=None) 

    args = parser.parse_args()

    # read the data dict JSONs and parse the feature and outcome columns
    x_data_dict_file, y_data_dict_file = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_data_dict_file)
    y_data_dict = load_data_dict_json(y_data_dict_file)
    
    feature_cols = parse_feature_cols(x_data_dict)
    key_cols = parse_id_cols(x_data_dict)
    
    '''
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
    '''
    outcome_col_name = args.outcome_col_name
    x_train_csv, y_train_csv = args.train_csv_files.split(',')
    x_train_df = pd.read_csv(x_train_csv)
    y_train_df = pd.read_csv(y_train_csv)

    x_train = x_train_df[feature_cols].values.astype(np.float32)
    y_train = np.ravel(y_train_df[outcome_col_name])
    
    x_test_csv, y_test_csv = args.test_csv_files.split(',')
    x_test_df = pd.read_csv(x_test_csv)
    y_test_df = pd.read_csv(y_test_csv)

    x_test = x_test_df[feature_cols].values.astype(np.float32)
    y_test = np.ravel(y_test_df[outcome_col_name])
    
    
    
    # Prepare data for classification
#     x_train = df_by_split['train'][feature_cols].values.astype(np.float32)
#     y_train = np.ravel(df_by_split['train'][outcome_col_name])
    
#     x_test = df_by_split['test'][feature_cols].values.astype(np.float32)
#     y_test = np.ravel(df_by_split['test'][outcome_col_name])        
    
    if args.valid_csv_files is None:
        # get the validation set
        splitter = Splitter(
            size=args.validation_size, random_state=41,
            n_splits=args.n_splits, cols_to_group=args.key_cols_to_group_when_splitting)
        # Assign training instances to splits by provided keys
        key_train = splitter.make_groups_from_df(x_train_df[key_cols])


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
    
    # normalize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_transformed = scaler.transform(x_train)
    del x_train
    x_valid_transformed = scaler.transform(x_valid)
    del x_valid
    x_test_transformed = scaler.transform(x_test) 
    del x_test
    
    x_train_valid_transformed = np.vstack([x_train_transformed, x_valid_transformed])
    y_train_valid = np.concatenate([y_train, y_valid])
    
    
    # add the window end timestamps in train, valid and test
    x_train_df['window_end_timestamp'] = pd.to_datetime(x_train_df['admission_timestamp'])+pd.to_timedelta(x_train_df['window_end'], 'h')
    x_valid_df['window_end_timestamp'] = pd.to_datetime(x_valid_df['admission_timestamp'])+pd.to_timedelta(x_valid_df['window_end'], 'h')
    x_test_df['window_end_timestamp'] = pd.to_datetime(x_test_df['admission_timestamp'])+pd.to_timedelta(x_test_df['window_end'], 'h')
    
    n_features = len(feature_cols)
    models_dict = {'logistic regression' : {'dirname':'skorch_logistic_regression', 
                                            'model_constructor': SkorchLogisticRegression(n_features=n_features),
                                            'prefix' : '',
                                           'model_color' : 'r'},
#                  'random forest' : {'dirname': 'random_forest',
#                                    'model_constructor' : None,
#                                    'prefix' : '',
#                                     'model_color' : 'g'},
                 'lightGBM' : {'dirname': 'lightGBM',
                                   'model_constructor' : None,
                                   'prefix' : '',
                                    'model_color' : 'g'},
                  'MLP 1 layer' : {'dirname' : 'skorch_mlp',
                                   'model_constructor' : SkorchMLP(n_features=n_features,
                                                         n_hiddens=32,
                                                         n_layers=1),
                                  'prefix' : '*n_layers=1',
                                  'model_color' : 'b'},
                   
                  'MLP 2 layer' : {'dirname' : 'skorch_mlp',
                                   'model_constructor' : SkorchMLP(n_features=n_features,
                                                         n_hiddens=32,
                                                         n_layers=2),
                                  'prefix' : '*n_layers=2',
                                   'model_color' : 'k'},
#                    'Support Vector Classifier' :{'dirname' : 'SVC',
#                                                  'model_constructor' : None,
#                                                  'prefix' : ''}
                  }
    
    perf_dict_list = []
    pr_f_tr, pr_axs_tr = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_va, pr_axs_va = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_te, pr_axs_te = plt.subplots(1, 1, figsize=(10, 8))
    for model_name in models_dict.keys():
        model_perf_csvs = glob.glob(os.path.join(args.clf_models_dir, models_dict[model_name]['dirname'], 
                                                 models_dict[model_name]['prefix']+'*_perf.csv'))
        G = len(model_perf_csvs)
        precision_scores_train_valid_G = np.zeros(G)
        recall_scores_train_valid_G = np.zeros(G)
        precision_scores_train_G = np.zeros(G)
        precision_scores_valid_G = np.zeros(G)
        recall_scores_train_G = np.zeros(G)
        recall_scores_valid_G = np.zeros(G)
        auprc_scores_train_G = np.zeros(G)
        auprc_scores_valid_G = np.zeros(G)

        # choose the hyperparamater that achieves max recall on stacked train and validation and achieve atleast 30% precision on stacked train and validation
        for i, model_perf_csv in enumerate(model_perf_csvs):
            model_perf_df = pd.read_csv(model_perf_csv)
            thr = model_perf_df['threshold'][0]

            if models_dict[model_name]['model_constructor'] is not None:
                clf = models_dict[model_name]['model_constructor']
                clf.initialize()
                model_param_file = model_perf_csv.replace('_perf.csv', 'params.pt')
                clf.load_params(model_param_file)
            else:
                model_param_file = model_perf_csv.replace('_perf.csv', '_trained_model.joblib')
                clf = load(open(model_param_file, 'rb'))
                
            y_train_valid_proba_vals = clf.predict_proba(x_train_valid_transformed)
            precision_scores_train_valid_G[i] = precision_score(y_train_valid, y_train_valid_proba_vals[:,1]>=thr)
            recall_scores_train_valid_G[i] = recall_score(y_train_valid, y_train_valid_proba_vals[:,1]>=thr)
            
            y_train_proba_vals = clf.predict_proba(x_train_transformed)
            precision_scores_train_G[i] = precision_score(y_train, y_train_proba_vals[:,1]>=thr)
            recall_scores_train_G[i] = recall_score(y_train, y_train_proba_vals[:,1]>=thr) 
            auprc_scores_train_G[i] = average_precision_score(y_train, y_train_proba_vals[:,1])
            
            y_valid_proba_vals = clf.predict_proba(x_valid_transformed)
            precision_scores_valid_G[i] = precision_score(y_valid, y_valid_proba_vals[:,1]>=thr)
            recall_scores_valid_G[i] = recall_score(y_valid, y_valid_proba_vals[:,1]>=thr)
            auprc_scores_valid_G[i] = average_precision_score(y_valid, y_valid_proba_vals[:,1])
                
        best_model_auprc_ind = np.argmax(auprc_scores_valid_G)
#         best_model_auprc_train = auprc_scores_train_G[best_model_auprc_ind]
#         best_model_auprc_valid = auprc_scores_valid_G[best_model_auprc_ind]
        
        best_model_perf_csv = model_perf_csvs[best_model_auprc_ind]
        best_model_perf_df = pd.read_csv(best_model_perf_csv)
        best_model_threshold = best_model_perf_df['threshold'][0]

        if models_dict[model_name]['model_constructor'] is not None:
            best_model_clf_file = best_model_perf_csv.replace('_perf.csv', 'params.pt')
            best_model_clf = models_dict[model_name]['model_constructor']
            best_model_clf.initialize()
            best_model_clf.load_params(best_model_clf_file)
        else:
            best_model_clf_file = best_model_perf_csv.replace('_perf.csv', '_trained_model.joblib')
            best_model_clf = load(open(best_model_clf_file, 'rb'))

        # predict probas
        y_train_pred_probas = best_model_clf.predict_proba(x_train_transformed)
        y_valid_pred_probas = best_model_clf.predict_proba(x_valid_transformed)
        y_test_pred_probas = best_model_clf.predict_proba(x_test_transformed)

        # get precision and recall on train, valid and test
        best_model_auprc_train = average_precision_score(y_train, y_train_pred_probas[:,1])
        best_model_auprc_valid = average_precision_score(y_valid, y_valid_pred_probas[:,1])
        best_model_auprc_test = average_precision_score(y_test, y_test_pred_probas[:,1])
        
        perf_dict = {'model' : model_name,
                     'best_model_auprc_train' : best_model_auprc_train,
                     'best_model_auprc_valid' : best_model_auprc_valid,
                     'best_model_auprc_test' : best_model_auprc_test,
                     'best_model_file' : best_model_clf_file
                    }
        
        print(perf_dict)
        perf_dict_list.append(perf_dict)
        
        # create the precision recall plot
        precs_train, recs_train, thresholds_train = precision_recall_curve(y_train, y_train_pred_probas[:,1])
        precs_valid, recs_valid, thresholds_valid = precision_recall_curve(y_valid, y_valid_pred_probas[:,1])
        precs_test, recs_test, thresholds_test = precision_recall_curve(y_test, y_test_pred_probas[:,1])  
        
        linewidth=1.0
        pr_axs_tr.plot(recs_train, precs_train, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_train), linewidth=linewidth)
        
        pr_axs_va.plot(recs_valid, precs_valid, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_valid), linewidth=linewidth)
        
        pr_axs_te.plot(recs_test, precs_test, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_test), linewidth=linewidth)
        
    ticks = np.arange(0.0, 1.1, 0.1)
    ticklabels = ['%.1f'%x for x in ticks]
    lims = [-0.05, 1.05]
    for ax in [pr_axs_tr, pr_axs_va, pr_axs_te]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        ax.legend()
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    pr_f_tr.savefig(os.path.join(args.output_dir, 'pr_curve_train.png'))
    pr_f_va.savefig(os.path.join(args.output_dir,'pr_curve_valid.png'))
    pr_f_te.savefig(os.path.join(args.output_dir,'pr_curve_test.png'))
    from IPython import embed; embed()
    '''
        min_precision_train = 0.3
        min_precision_valid = 0.2
        
#         keep_inds = (precision_scores_train_G >= min_precision_train)&(precision_scores_valid_G >= min_precision_valid)
        keep_inds = precision_scores_train_valid_G>=0.3
        if keep_inds.sum()==0:
            keep_inds = precision_scores_train_valid_G>=0.15
        
        if keep_inds.sum()>0:
#             precision_scores_G_filtered = precision_scores_G[keep_inds]
            recall_scores_valid_G_filtered = recall_scores_valid_G[keep_inds]
            model_perf_csvs_filtered = np.array(model_perf_csvs)[keep_inds]

            best_model_ind = np.argmax(recall_scores_valid_G_filtered)
            best_model_perf_csv = model_perf_csvs_filtered[best_model_ind]
            best_model_perf_df = pd.read_csv(best_model_perf_csv)
            best_model_threshold = best_model_perf_df['threshold'][0]
#             best_model_train_valid_precision = precision_scores_G_filtered[best_model_ind]
#             best_model_train_valid_recall = recall_scores_G_filtered[best_model_ind]

            if models_dict[model_name]['model_constructor'] is not None:
                best_model_clf_file = best_model_perf_csv.replace('_perf.csv', 'params.pt')
                best_model_clf = models_dict[model_name]['model_constructor']
                best_model_clf.initialize()
                best_model_clf.load_params(best_model_clf_file)
            else:
                best_model_clf_file = best_model_perf_csv.replace('_perf.csv', '_trained_model.joblib')
                best_model_clf = load(open(best_model_clf_file, 'rb'))

            # predict probas
            y_train_pred_probas = best_model_clf.predict_proba(x_train_transformed)
            y_valid_pred_probas = best_model_clf.predict_proba(x_valid_transformed)
            y_test_pred_probas = best_model_clf.predict_proba(x_test_transformed)

            # get precision and recall on train, valid and test
            best_model_precision_train = precision_score(y_train, y_train_pred_probas[:,1]>=best_model_threshold)
            best_model_precision_valid = precision_score(y_valid, y_valid_pred_probas[:,1]>=best_model_threshold)
            best_model_precision_test = precision_score(y_test, y_test_pred_probas[:,1]>=best_model_threshold)

            best_model_recall_train = recall_score(y_train, y_train_pred_probas[:,1]>=best_model_threshold)
            best_model_recall_valid = recall_score(y_valid, y_valid_pred_probas[:,1]>=best_model_threshold)
            best_model_recall_test = recall_score(y_test, y_test_pred_probas[:,1]>=best_model_threshold)

            perf_dict = {'model' : model_name,
                        'precision_train' : best_model_precision_train,
                        'precision_valid' : best_model_precision_valid,
                        'precision_test' : best_model_precision_test,
                        'recall_train' : best_model_recall_train,
                        'recall_valid' : best_model_recall_valid,
                        'recall_test' : best_model_recall_test,
                        'best_model_file' : best_model_clf_file,
                        'best_model_threshold' : best_model_threshold
                        }
        
            perf_dict_list.append(perf_dict)
        print(perf_dict)
    perf_df = pd.DataFrame(perf_dict_list)
    print(perf_df)
    
    ## plot the preicision and recall over time 
    window_ends = x_train_df['window_end'].unique() 
    all_models_precision_train_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))                     
    all_models_precision_valid_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    all_models_precision_test_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    
    all_models_recall_train_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    all_models_recall_valid_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    all_models_recall_test_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    
    all_models_frac_pos_train_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    all_models_frac_pos_valid_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    all_models_frac_pos_test_per_window_dict = dict(zip(models_dict.keys(), [np.zeros(len(window_ends)) for i in range(len(models_dict.keys()))]))
    
    for model_name in models_dict.keys():
        model_ind = perf_df['model']==model_name
        best_model_clf_file = perf_df.loc[model_ind, 'best_model_file'].values[0]
        best_model_threshold = perf_df.loc[model_ind, 'best_model_threshold'].values[0]
        if models_dict[model_name]['model_constructor'] is not None:
            best_model_clf = models_dict[model_name]['model_constructor']
            best_model_clf.initialize()
            best_model_clf.load_params(best_model_clf_file)
        else:
            best_model_clf = load(open(best_model_clf_file, 'rb'))
         
        for ii, window_end in enumerate(window_ends) :
            x_tr = x_train_df[x_train_df['window_end']==window_end][feature_cols].values.astype(np.float32)
            y_tr = np.ravel(y_train_df[y_train_df['window_end']==window_end][outcome_col_name])
            x_te = x_test_df[x_test_df['window_end']==window_end][feature_cols].values.astype(np.float32)
            y_te = np.ravel(y_test_df[y_test_df['window_end']==window_end][outcome_col_name])
            x_va = x_valid_df[x_valid_df['window_end']==window_end][feature_cols].values.astype(np.float32)
            y_va = np.ravel(y_valid_df[y_valid_df['window_end']==window_end][outcome_col_name])
            x_tr_transformed = scaler.transform(x_tr)
            x_va_transformed = scaler.transform(x_va)
            x_te_transformed = scaler.transform(x_te)
            
            y_tr_pred_probas = best_model_clf.predict_proba(x_tr_transformed)
            y_va_pred_probas = best_model_clf.predict_proba(x_va_transformed)
            y_te_pred_probas = best_model_clf.predict_proba(x_te_transformed)
            
            all_models_precision_train_per_window_dict[model_name][ii] = precision_score(y_tr, y_tr_pred_probas[:,1]>=best_model_threshold) 
            all_models_precision_valid_per_window_dict[model_name][ii] = precision_score(y_va, y_va_pred_probas[:,1]>=best_model_threshold)
            all_models_precision_test_per_window_dict[model_name][ii] = precision_score(y_te, y_te_pred_probas[:,1]>=best_model_threshold)

            all_models_recall_train_per_window_dict[model_name][ii] = recall_score(y_tr, y_tr_pred_probas[:,1]>=best_model_threshold) 
            all_models_recall_valid_per_window_dict[model_name][ii] = recall_score(y_va, y_va_pred_probas[:,1]>=best_model_threshold)
            all_models_recall_test_per_window_dict[model_name][ii] = recall_score(y_te, y_te_pred_probas[:,1]>=best_model_threshold) 
            
            all_models_frac_pos_train_per_window_dict[model_name][ii] = y_tr.sum()/len(y_tr)
            all_models_frac_pos_valid_per_window_dict[model_name][ii] = y_va.sum()/len(y_va)
            all_models_frac_pos_test_per_window_dict[model_name][ii] = y_te.sum()/len(y_te)
    
    # plot 3x3 grid of precision recall and frac positive on train, validation and test
    f, axs = plt.subplots(3, 3, sharex=True, figsize=(15, 12))
    model_colors = ['r', 'b', 'g', 'k', 'm']
    for ii, (metric, metric_tr_perf_dict, metric_val_perf_dict, metric_te_perf_dict) in enumerate([('precision',
                                                                                    all_models_precision_train_per_window_dict,
                                                                                   all_models_precision_valid_per_window_dict,
                                                                                   all_models_precision_test_per_window_dict),
                                                                                  ('recall',
                                                                                  all_models_recall_train_per_window_dict,
                                                                                  all_models_recall_valid_per_window_dict,
                                                                                  all_models_recall_test_per_window_dict),
                                                                                  ('frac positive',
                                                                                  all_models_frac_pos_train_per_window_dict,
                                                                                  all_models_frac_pos_valid_per_window_dict,
                                                                                  all_models_frac_pos_test_per_window_dict)]):
        
        for jj, model_name in enumerate(models_dict.keys()):
            axs[ii, 0].plot(window_ends, metric_tr_perf_dict[model_name], model_colors[jj]+'-o', label=model_name, linewidth=2)
            axs[ii, 1].plot(window_ends, metric_val_perf_dict[model_name], model_colors[jj]+'-o', label=model_name, linewidth=2)
            axs[ii, 2].plot(window_ends, metric_te_perf_dict[model_name], model_colors[jj]+'-o', label=model_name, linewidth=2)
        
        axs[0, 2].legend()
        axs[2, ii].set_xlabel('hours of data observed')
        axs[ii, 0].set_ylabel(metric)
    axs[0, 0].set_title('Train')
    axs[0, 1].set_title('Valid')
    axs[0, 2].set_title('Test')
    
    output_png = os.path.join(args.output_dir, 'all_clfs_dynamic_precision_recall_per_window.png')
    output_csv = os.path.join(args.output_dir, 'all_clfs_full_dataset_precision_recall.csv')
    f.savefig(output_png)
    perf_df.to_csv(output_csv, index=False)
    
    print('Performance files saved to :\n%s\n%s'%(output_csv, output_png))

    from IPython import embed; embed()
    # get the first admission timestamp(t0) and the last deterioration/discharge timestamp(tend) in train, valid and test
    min_ts_tr = pd.to_datetime(x_train_df['admission_timestamp'].min())
    min_ts_va = pd.to_datetime(x_valid_df['admission_timestamp'].min()) 
    min_ts_te = pd.to_datetime(x_test_df['admission_timestamp'].min())
    
    max_ts_tr = pd.to_datetime(x_train_df['window_end_timestamp'].max())
    max_ts_va = pd.to_datetime(x_valid_df['window_end_timestamp'].max())  
    max_ts_te = pd.to_datetime(x_test_df['window_end_timestamp'].max()) 
    
    # create an array of non-overlapping windows of size=1 week from t0 to tend
    prediction_freq = '7D'
    prediction_window_ends_ts_tr = pd.date_range(min_ts_tr, max_ts_tr, freq=prediction_freq).values
    prediction_window_ends_ts_va = pd.date_range(min_ts_va, max_ts_va, freq=prediction_freq).values
    prediction_window_ends_ts_te = pd.date_range(min_ts_te, max_ts_te, freq=prediction_freq).values
    
    # for every week starting from t0 until tend, run each classifier and get the TP, TN, FP and FN
    alarms_perf_dict_list = []
    for model_name in models_dict.keys():
        model_ind = perf_df['model']==model_name
        best_model_clf_file = perf_df.loc[model_ind, 'best_model_file'].values[0]
        best_model_threshold = perf_df.loc[model_ind, 'best_model_threshold'].values[0]
        if models_dict[model_name]['model_constructor'] is not None:
            best_model_clf = models_dict[model_name]['model_constructor']
            best_model_clf.initialize()
            best_model_clf.load_params(best_model_clf_file)
        else:
            best_model_clf = load(open(best_model_clf_file, 'rb'))
        
        print('Evaluating distribution of TP, FP, TN, FN every %s with %s'%(prediction_freq, model_name))
        
        for split, x_df, y_df, split_prediction_window_ends in [('train', x_train_df, y_train_df, prediction_window_ends_ts_tr),
                                                               ('valid', x_valid_df, y_valid_df, prediction_window_ends_ts_va),
                                                               ('test', x_test_df, y_test_df, prediction_window_ends_ts_te)]:
            
            N_pred_segments = len(split_prediction_window_ends)-1
            curr_alarms_perf_dict = {'split' : split,
                                    'TP_arr' : np.zeros(N_pred_segments),
                                    'TN_arr' : np.zeros(N_pred_segments),
                                    'FP_arr' : np.zeros(N_pred_segments),
                                    'FN_arr' : np.zeros(N_pred_segments),
                                    'model' : model_name,
                                    'N_preds' : np.zeros(N_pred_segments),
                                    'mean_alarms_window_before_deterioration' : None,
                                    'mean_alarms_window_before_no_deterioration' : None}
            
            N_alarms_window_before_deterioration = np.zeros(N_pred_segments)
            N_alarms_window_before_no_deterioration = np.zeros(N_pred_segments)
            for ii in range(N_pred_segments):
                pred_segment_start = split_prediction_window_ends[ii]
                pred_segment_end = split_prediction_window_ends[ii+1]
            
                keep_inds = (x_df['window_end_timestamp']>=pred_segment_start)&(x_df['window_end_timestamp']<=pred_segment_end)
                if keep_inds.sum()>0:
                    curr_x = x_df[keep_inds][feature_cols].values.astype(np.float32)
                    curr_y = np.ravel(y_df[keep_inds][outcome_col_name])

                    curr_x_transformed = scaler.transform(curr_x)
                    curr_y_pred_probas = best_model_clf.predict_proba(curr_x_transformed)
                    curr_y_preds = curr_y_pred_probas[:,1]>=best_model_threshold

                    TP = np.logical_and(curr_y==1, curr_y_preds==1).sum()
                    FP = np.logical_and(curr_y==0, curr_y_preds==1).sum()
                    FN = np.logical_and(curr_y==1, curr_y_preds==0).sum()
                    TN = np.logical_and(curr_y==0, curr_y_preds==0).sum()

                    curr_alarms_perf_dict['TP_arr'][ii] = TP
                    curr_alarms_perf_dict['FP_arr'][ii] = FP
                    curr_alarms_perf_dict['FN_arr'][ii] = FN
                    curr_alarms_perf_dict['TN_arr'][ii] = TN
                    curr_alarms_perf_dict['N_preds'][ii] = len(curr_y)
                    
                    
                    # predict on last window before clinical deterioration
                    x_df_window_before_deterioration = x_df[keep_inds].drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
                    y_df_window_before_deterioration = pd.merge(x_df_window_before_deterioration, y_df, 
                                                                on=key_cols+['window_start', 'window_end', 'admission_timestamp'],
                                                                how='inner')[y_df.columns]
                    
                    curr_x_window_before_deterioration = x_df_window_before_deterioration[feature_cols].values.astype(np.float32)
                    curr_y_window_before_deterioration = np.ravel(y_df_window_before_deterioration[outcome_col_name])
                    
                    curr_x_window_before_deterioration_transformed = scaler.transform(curr_x_window_before_deterioration)
                    curr_y_window_before_deterioration_pred_probas = best_model_clf.predict_proba(curr_x_window_before_deterioration_transformed)
                    curr_y_window_before_deterioration_preds = curr_y_window_before_deterioration_pred_probas[:,1]>=best_model_threshold
                    
                    N_alarms_window_before_deterioration[ii] = np.logical_and(curr_y_window_before_deterioration==1,
                                                                          curr_y_window_before_deterioration_preds==1).sum()
                    
                    N_alarms_window_before_no_deterioration[ii] = np.logical_and(curr_y_window_before_deterioration==0,
                                                                             curr_y_window_before_deterioration_preds==1).sum()                    
            curr_alarms_perf_dict['mean_alarms_window_before_deterioration'] = np.mean(N_alarms_window_before_deterioration)
            curr_alarms_perf_dict['mean_alarms_window_before_no_deterioration'] = np.mean(N_alarms_window_before_no_deterioration)
                    
            alarms_perf_dict_list.append(curr_alarms_perf_dict)    
            
    alarms_perf_df = pd.DataFrame(alarms_perf_dict_list, columns = curr_alarms_perf_dict.keys())
    alarms_csv = os.path.join(args.output_dir, 'alarm_stats.csv')
    alarms_perf_df.to_csv(alarms_csv, index=False)
    print('Alarm stats saved to : %s'%alarms_csv)
    
    ## plot the distribution of alarms
    print('Plotting Distribution of Alarms on train, validation and test')
    hist_dict = {'True Positive Alarms' : {'colname' : 'TP_arr',
                                          'n_bins' : 20},
                'False Positive Alarms' : {'colname' : 'FP_arr',
                                          'n_bins' : 20},
                'False Negative Alarms' : {'colname' : 'FN_arr',
                                          'n_bins' : 20},
#                 'True Negative Alarms' : {'colname' : 'TN_arr',
#                                           'n_bins' : 20}
                }
    
    
    for kk, split in enumerate(alarms_perf_df['split'].unique()):
        f, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ii, (alarm_type, alarm_type_dict) in enumerate(hist_dict.items()):
            all_counts_arr = np.concatenate(alarms_perf_df[alarm_type_dict['colname']].values)  
            bins = np.linspace(min(all_counts_arr), max(all_counts_arr), alarm_type_dict['n_bins']).astype(int)
            keep_inds = alarms_perf_df['split']==split
            alarms_split_perf_df = alarms_perf_df[keep_inds]

            for jj, model in enumerate(alarms_split_perf_df['model'].values):
                model_ind = alarms_split_perf_df['model']==model
                x = alarms_split_perf_df.loc[model_ind, alarm_type_dict['colname']].values[0]
                axs[ii].hist(x=x, bins=bins, label=model, alpha=0.7)
                axs[ii].legend()
                axs[ii].set_xlabel(alarm_type)
                axs[ii].set_ylabel('Number of weeks')
        if split == 'train':
            min_ts = min_ts_tr
            max_ts = max_ts_tr
        elif split == 'valid':
            min_ts = min_ts_va
            max_ts = max_ts_va
        elif split == 'test':
            min_ts = min_ts_te
            max_ts = max_ts_te
        f.suptitle('Weekly Distribution of Alarms on %s set (from %s to %s)'%(split, min_ts, max_ts), fontsize=16)
        alarms_png = os.path.join(args.output_dir, 'alarms_dist_%s'%split)
        f.savefig(alarms_png)
        print('%s set alarms saved to %s'%(split, alarms_png))
        
    from IPython import embed; embed()
    '''
