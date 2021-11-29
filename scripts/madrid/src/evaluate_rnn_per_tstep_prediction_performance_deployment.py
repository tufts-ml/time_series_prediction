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
                            average_precision_score, precision_recall_curve, roc_curve)
DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))

from dataset_loader import TidySequentialDataCSVLoader
from RNNPerTStepBinaryClassifier import RNNPerTStepBinaryClassifier
import matplotlib.pyplot as plt

from feature_transformation import parse_feature_cols, parse_output_cols, parse_id_cols
from utils import load_data_dict_json
from pickle import load
import seaborn as sns
import torch

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
    x_train_csv_filename, y_train_csv_filename = args.train_csv_files.split(',')
    x_valid_csv_filename, y_valid_csv_filename = args.valid_csv_files.split(',')
    x_test_csv_filename, y_test_csv_filename = args.test_csv_files.split(',')
    x_dict, y_dict = args.data_dict_files.split(',')
    x_data_dict = load_data_dict_json(x_dict)
    y_data_dict = load_data_dict_json(y_dict)
    

    # get the id and feature columns
    id_cols = parse_id_cols(x_data_dict)
    feature_cols = parse_feature_cols(x_data_dict)
    outcome_col_name = args.outcome_col_name
    # extract data
    train_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_train_csv_filename,
        y_csv_path=y_train_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=outcome_col_name,
        y_label_type='per_tstep'
    )

    valid_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_valid_csv_filename,
        y_csv_path=y_valid_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=outcome_col_name,
        y_label_type='per_tstep'
    )
    
    test_vitals = TidySequentialDataCSVLoader(
        x_csv_path=x_test_csv_filename,
        y_csv_path=y_test_csv_filename,
        x_col_names=feature_cols,
        idx_col_names=id_cols,
        y_col_name=outcome_col_name,
        y_label_type='per_tstep'
    )

    x_train, y_train = train_vitals.get_batch_data(batch_id=0)
    x_valid, y_valid = valid_vitals.get_batch_data(batch_id=0)
    x_test, y_test = test_vitals.get_batch_data(batch_id=0)
    
#     x_train_df = pd.read_csv(x_train_csv)
    y_train_df = pd.read_csv(y_train_csv_filename)

    
#     x_test_df = pd.read_csv(x_test_csv)
    y_test_df = pd.read_csv(y_test_csv_filename)    

#     x_valid_df = pd.read_csv(x_valid_csv)
    y_valid_df = pd.read_csv(y_valid_csv_filename)
        
    y_train_df['window_end_timestamp'] = pd.to_datetime(y_train_df['admission_timestamp'])+pd.to_timedelta(y_train_df['stop'], 'h')
    y_valid_df['window_end_timestamp'] = pd.to_datetime(y_valid_df['admission_timestamp'])+pd.to_timedelta(y_valid_df['stop'], 'h')
    y_test_df['window_end_timestamp'] = pd.to_datetime(y_test_df['admission_timestamp'])+pd.to_timedelta(y_test_df['stop'], 'h')
    
    split_dict = {'N_train' : len(x_train),
                 'N_valid' : len(x_valid),
                 'N_test' : len(x_test),
                 'pos_frac_train' : y_train_df[outcome_col_name].sum()/len(y_train_df),
                 'pos_frac_valid' : y_valid_df[outcome_col_name].sum()/len(y_valid_df),
                 'pos_frac_test' : y_test_df[outcome_col_name].sum()/len(y_test_df),
                 'N_patients_train' : len(y_train_df.patient_id.unique()),
                 'N_patients_valid' : len(y_valid_df.patient_id.unique()),
                 'N_patients_test' : len(y_test_df.patient_id.unique()),
                 'N_admissions_train' : len(y_train_df.hospital_admission_id.unique()),
                 'N_admissions_valid' : len(y_valid_df.hospital_admission_id.unique()),
                 'N_admissions_test' : len(y_test_df.hospital_admission_id.unique()),                 
                 }
    
    print(split_dict)
    

    
    
    n_features = len(feature_cols)
    models_dict = {
        'GRU-RNN' : {'dirname':'rnn_per_tstep', 
                                            'model_constructor': RNNPerTStepBinaryClassifier(module__rnn_type='GRU',
                                                                                     module__n_layers=2,
                                                                                     module__n_hiddens=128,
                                                                                     module__n_inputs=x_test.shape[-1]),
                                           'prefix' : '',
                                           'model_color' : 'r', 
                                           'model_marker' : 's'},
                  }
    
    
    perf_dict_list = []
    pr_f_tr, pr_axs_tr = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_va, pr_axs_va = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_te, pr_axs_te = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_tr, auc_axs_tr = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_va, auc_axs_va = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_te, auc_axs_te = plt.subplots(1, 1, figsize=(10, 8))
    
    
    for model_name in models_dict.keys():
        
        model_perf_csvs = glob.glob(os.path.join(args.clf_models_dir, models_dict[model_name]['dirname'], 
                                                 models_dict[model_name]['prefix']+'*.csv'))
        G = len(model_perf_csvs)
        auprc_scores_train_G = np.zeros(G)
        auprc_scores_valid_G = np.zeros(G)
        auprc_scores_test_G = np.zeros(G)
        auroc_scores_train_G = np.zeros(G)
        auroc_scores_valid_G = np.zeros(G)
        auroc_scores_test_G = np.zeros(G)

        # choose the hyperparamater that achieves max auprc
        for i, model_perf_csv in enumerate(model_perf_csvs):
            model_perf_df = pd.read_csv(model_perf_csv)
            thr = model_perf_df['threshold'][0]

            clf = models_dict[model_name]['model_constructor']
            clf.initialize()
            model_param_file = model_perf_csv.replace('.csv', 'params.pt')
            clf.load_params(model_param_file) 

            splits = ['train', 'valid', 'test']
            auroc_per_split,  auprc_per_split = [np.zeros(len(splits)), np.zeros(len(splits))]

            for ii, (X, y) in enumerate([(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]):
                keep_inds = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X)), dim=-1))
                y_pred_proba_pos = clf.predict_proba(X)[keep_inds][:,1].detach().numpy()
                auroc_per_split[ii] = roc_auc_score(y[keep_inds], y_pred_proba_pos)
                auprc_per_split[ii] = average_precision_score(y[keep_inds], y_pred_proba_pos)
    
            auroc_scores_train_G[i], auroc_scores_valid_G[i], auroc_scores_test_G[i] = auroc_per_split
            auprc_scores_train_G[i], auprc_scores_valid_G[i], auprc_scores_test_G[i] = auprc_per_split
            
            
        best_model_auprc_ind = np.argmax(auprc_scores_valid_G)

        best_model_perf_csv = model_perf_csvs[best_model_auprc_ind]
        best_model_perf_df = pd.read_csv(best_model_perf_csv)
        best_model_threshold = best_model_perf_df['threshold'][0]
           
        best_model_param_file = best_model_perf_csv.replace('.csv', 'params.pt')
        clf = models_dict[model_name]['model_constructor']
        clf.initialize()
        clf.load_params(best_model_param_file) 
        
        auroc_per_split,  auprc_per_split = [np.zeros(len(splits)), np.zeros(len(splits))]
        pred_proba_vals_per_split = []
        for ii, (X, y) in enumerate([(x_train, y_train), (x_valid, y_valid), (x_test, y_test)]):
            keep_inds = torch.logical_not(torch.all(torch.isnan(torch.FloatTensor(X)), dim=-1))
            y_pred_proba_pos = clf.predict_proba(X)[keep_inds][:,1].detach().numpy()
            auroc_per_split[ii] = roc_auc_score(y[keep_inds], y_pred_proba_pos)
            auprc_per_split[ii] = average_precision_score(y[keep_inds], y_pred_proba_pos) 
            pred_proba_vals_per_split.append(y_pred_proba_pos)
        
        # get precision and recall on train, valid and test
        best_model_auprc_train, best_model_auprc_valid, best_model_auprc_test = auprc_per_split
        best_model_auroc_train, best_model_auroc_valid, best_model_auroc_test = auroc_per_split
        y_train_proba_vals, y_valid_proba_vals, y_test_proba_vals = pred_proba_vals_per_split
        
        
        perf_dict = {'model' : model_name,
                     'best_model_auprc_train' : best_model_auprc_train,
                     'best_model_auprc_valid' : best_model_auprc_valid,
                     'best_model_auprc_test' : best_model_auprc_test,
                     'best_model_train_pred_probas' : y_train_proba_vals,
                     'best_model_valid_pred_probas' : y_valid_proba_vals,
                     'best_model_test_pred_probas' : y_test_proba_vals,
                     'best_model_file' : best_model_param_file
                    }
        
        print(perf_dict)
        perf_dict_list.append(perf_dict)
        
        perf_df = pd.DataFrame(perf_dict_list)
        print(perf_df)
        
        # create the precision recall plot
        precs_train, recs_train, thresholds_train = precision_recall_curve(y_train_df[outcome_col_name].values, y_train_proba_vals)
        precs_valid, recs_valid, thresholds_valid = precision_recall_curve(y_valid_df[outcome_col_name].values, y_valid_proba_vals)
        precs_test, recs_test, thresholds_test = precision_recall_curve(y_test_df[outcome_col_name].values, y_test_proba_vals)  
        
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train_df[outcome_col_name].values, y_train_proba_vals)
        fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid_df[outcome_col_name].values, y_valid_proba_vals)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test_df[outcome_col_name].values, y_test_proba_vals)
        
        linewidth=1.0
        pr_axs_tr.plot(recs_train, precs_train, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_train), linewidth=linewidth)
        
        pr_axs_va.plot(recs_valid, precs_valid, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_valid), linewidth=linewidth)
        
        pr_axs_te.plot(recs_test, precs_test, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_test), linewidth=linewidth)
        
        auc_axs_tr.plot(fpr_train, tpr_train, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_train), linewidth=linewidth)
        
        auc_axs_va.plot(fpr_valid, tpr_valid, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_valid), linewidth=linewidth)
        
        auc_axs_te.plot(fpr_test, tpr_test, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_test), linewidth=linewidth)
    
    perf_df.to_pickle(os.path.join(args.output_dir, 'performance_of_best_rnn.pkl'))
    print('Saved the best model performance on full dataset to :\n%s'%(os.path.join(args.output_dir, 'performance_of_best_rnn.pkl')))
    
    
    ticks = np.arange(0.0, 1.1, 0.1)
    ticklabels = ['%.1f'%x for x in ticks]
    lims = [-0.05, 1.05]
    fontsize = 12
    for (ax, ax_title) in [(pr_axs_tr, 'Train Precision Recall Curve'), 
                           (pr_axs_va, 'Validation Precision Recall Curve'),
                           (pr_axs_te, 'Test Precision Recall Curve')]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, fontsize=fontsize)
        ax.set_title(ax_title, fontsize = fontsize+3)
        ax.legend(fontsize=fontsize)
        ax.set_xlabel('Recall', fontsize=fontsize)
        ax.set_ylabel('Precision', fontsize=fontsize)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    pr_f_tr.savefig(os.path.join(args.output_dir, 'pr_curve_train.png'))
    pr_f_va.savefig(os.path.join(args.output_dir,'pr_curve_valid.png'))
    pr_f_te.savefig(os.path.join(args.output_dir,'pr_curve_test.png'))
    
    for (ax, ax_title) in [(auc_axs_tr, 'Train ROC Curve'), 
                           (auc_axs_va, 'Validation ROC Curve'),
                           (auc_axs_te, 'Test ROC Curve')]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=fontsize)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, fontsize=fontsize)
        ax.set_title(ax_title, fontsize = fontsize+3)
        ax.legend(fontsize=fontsize)
        ax.set_xlabel('FPR', fontsize=fontsize)
        ax.set_ylabel('TPR', fontsize=fontsize)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    auc_f_tr.savefig(os.path.join(args.output_dir, 'roc_curve_train.png'))
    auc_f_va.savefig(os.path.join(args.output_dir,'roc_curve_valid.png'))
    auc_f_te.savefig(os.path.join(args.output_dir,'roc_curve_test.png'))    
    
    print('Saved pr, roc curves on train, valid, test to : %s'%args.output_dir)
    
    # get the first admission timestamp(t0) and the last deterioration/discharge timestamp(tend) in train, valid and test
    min_ts_tr = pd.to_datetime(y_train_df['admission_timestamp'].min())
    min_ts_va = pd.to_datetime(y_valid_df['admission_timestamp'].min()) 
    min_ts_te = pd.to_datetime(y_test_df['admission_timestamp'].min())
    
    max_ts_tr = pd.to_datetime(y_train_df['window_end_timestamp'].max())
    max_ts_va = pd.to_datetime(y_valid_df['window_end_timestamp'].max())  
    max_ts_te = pd.to_datetime(y_test_df['window_end_timestamp'].max()) 
    
    # create an array of non-overlapping windows of size=1 week from t0 to tend
    prediction_freq = '7D'
    prediction_window_ends_ts_tr = pd.date_range(min_ts_tr, max_ts_tr, freq=prediction_freq).values
    prediction_window_ends_ts_va = pd.date_range(min_ts_va, max_ts_va, freq=prediction_freq).values
    prediction_window_ends_ts_te = pd.date_range(min_ts_te, max_ts_te, freq=prediction_freq).values
    
    # for every week starting from t0 until tend, run each classifier and get the TP, TN, FP and FN
    alarms_perf_dict_list = []
    for model_name in models_dict.keys():
        model_ind = perf_df['model']==model_name
        
        print('Evaluating distribution of TP, FP, TN, FN every %s with %s'%(prediction_freq, model_name))
        
        # get the predicted probabilities on train, valid and test for this classifier
        y_train_pred_probas = perf_df.loc[model_ind, 'best_model_train_pred_probas'].values[0]
        y_valid_pred_probas = perf_df.loc[model_ind, 'best_model_valid_pred_probas'].values[0]
        y_test_pred_probas = perf_df.loc[model_ind, 'best_model_test_pred_probas'].values[0]
        
        
        unique_probas = np.unique(y_valid_pred_probas)
        thr_grid_G = np.linspace(np.percentile(unique_probas,1), max(unique_probas), 1000)
        
        for split, y_df, split_prediction_window_ends, y_pred_probas in [('train', y_train_df,
                                                                                prediction_window_ends_ts_tr, y_train_pred_probas),
                                                                               ('valid', y_valid_df,
                                                                                prediction_window_ends_ts_va, y_valid_pred_probas),
                                                                               ('test', y_test_df,
                                                                                prediction_window_ends_ts_te, y_test_pred_probas)]:
            
            N_pred_segments = len(split_prediction_window_ends)-1
            
            curr_alarms_perf_dict = {'split' : split,
                                     'TP_arr' : np.zeros((N_pred_segments, len(thr_grid_G))),
                                     'TN_arr' : np.zeros((N_pred_segments, len(thr_grid_G))),
                                     'FP_arr' : np.zeros((N_pred_segments, len(thr_grid_G))),
                                     'FN_arr' : np.zeros((N_pred_segments, len(thr_grid_G))),
                                     'model' : model_name,
                                     'threshold_grid' : thr_grid_G,  
                                     'N_preds' : np.zeros(N_pred_segments),
                                     'N_adms' : np.zeros(N_pred_segments),
                                     'N_patients' : np.zeros(N_pred_segments),
                                     'precision_arr' : np.zeros((N_pred_segments, len(thr_grid_G)))+np.nan,
                                     'recall_arr' : np.zeros((N_pred_segments, len(thr_grid_G)))+np.nan}
            
            
#             N_alarms_window_before_deterioration = np.zeros(N_pred_segments)
#             N_alarms_window_before_no_deterioration = np.zeros(N_pred_segments)
            for ii in range(N_pred_segments):
                pred_segment_start = split_prediction_window_ends[ii]
                pred_segment_end = split_prediction_window_ends[ii+1]
            
                keep_inds = (y_df['window_end_timestamp']>=pred_segment_start)&(y_df['window_end_timestamp']<=pred_segment_end)
                if keep_inds.sum()>0:
#                     curr_x = x_df[keep_inds][feature_cols].values.astype(np.float32)
                    curr_unique_adms = y_df[keep_inds]['hospital_admission_id'].unique()
                    curr_unique_patients = y_df[keep_inds]['patient_id'].unique()
                    curr_y = np.ravel(y_df[keep_inds][outcome_col_name])
                    
                    curr_y_pred_probas = y_pred_probas[keep_inds]
                    curr_y_preds = curr_y_pred_probas[:, np.newaxis]>=thr_grid_G
                    
                    TP = np.logical_and((curr_y==1)[:, np.newaxis], curr_y_preds==1).sum(axis=0)
                    FP = np.logical_and((curr_y==0)[:, np.newaxis], curr_y_preds==1).sum(axis=0)
                    FN = np.logical_and((curr_y==1)[:, np.newaxis], curr_y_preds==0).sum(axis=0)
                    TN = np.logical_and((curr_y==0)[:, np.newaxis], curr_y_preds==0).sum(axis=0)

                    curr_alarms_perf_dict['TP_arr'][ii, :] = TP
                    curr_alarms_perf_dict['FP_arr'][ii, :] = FP
                    curr_alarms_perf_dict['FN_arr'][ii, :] = FN
                    curr_alarms_perf_dict['TN_arr'][ii, :] = TN
                    curr_alarms_perf_dict['N_preds'][ii] = len(curr_y)
                    curr_alarms_perf_dict['precision_arr'][ii, :] = TP/(TP+FP)
                    curr_alarms_perf_dict['recall_arr'][ii, :] = TP/(TP+FN)
                    curr_alarms_perf_dict['N_adms'][ii] = len(curr_unique_adms)
                    curr_alarms_perf_dict['N_patients'][ii] = len(curr_unique_patients)
                    
            alarms_perf_dict_list.append(curr_alarms_perf_dict)    
    
    alarms_perf_df = pd.DataFrame(alarms_perf_dict_list, columns = curr_alarms_perf_dict.keys())
    alarms_csv = os.path.join(args.output_dir, 'rnn_alarm_stats.csv')
    alarms_perf_df.to_pickle(alarms_csv)
    print('Alarm stats saved to : %s'%alarms_csv)
    
    from IPython import embed; embed()