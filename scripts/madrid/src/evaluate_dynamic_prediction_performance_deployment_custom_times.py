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
import seaborn as sns
import onnxruntime as rt
import joblib

def read_csv_with_float32_dtypes(filename, nrows=None):
    # Sample 100 rows of data to determine dtypes.
    df_test = pd.read_csv(filename, nrows=100)

    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    
    if nrows is not None:
        df = pd.read_csv(filename, dtype=float32_cols, nrows=nrows)
    else:
        df = pd.read_csv(filename, dtype=float32_cols)
    
    return df

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
#     parser.add_argument('--mews_train_csv_file', type=str, required=True,
#                         help='mews training subject scores')
#     parser.add_argument('--mews_valid_csv_file', type=str, required=True,
#                         help='mews validation subject scores')
#     parser.add_argument('--mews_test_csv_file', type=str, required=True,
#                         help='mews test subject scores')
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
    
    outcome_col_name = args.outcome_col_name
    x_train_csv, y_train_csv = args.train_csv_files.split(',')
    x_train_df = read_csv_with_float32_dtypes(x_train_csv)
    y_train_df = pd.read_csv(y_train_csv)

    x_train = x_train_df[feature_cols].values.astype(np.float32)
    y_train = np.ravel(y_train_df[outcome_col_name])
    del(x_train_df)
    
    x_test_csv, y_test_csv = args.test_csv_files.split(',')
    x_test_df = read_csv_with_float32_dtypes(x_test_csv)
    y_test_df = pd.read_csv(y_test_csv)
    
    x_test = x_test_df[feature_cols].values.astype(np.float32)
    y_test = np.ravel(y_test_df[outcome_col_name])
    del(x_test_df)
    
#     mews_train_df = pd.read_csv(args.mews_train_csv_file)
#     mews_valid_df = pd.read_csv(args.mews_valid_csv_file)
#     mews_test_df = pd.read_csv(args.mews_test_csv_file)
    
    # Prepare data for classification    
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
        x_valid_df = read_csv_with_float32_dtypes(x_valid_csv)
        y_valid_df = pd.read_csv(y_valid_csv)
        
        x_valid = x_valid_df[feature_cols].values.astype(np.float32)
        y_valid = np.ravel(y_valid_df[outcome_col_name])
        del(x_valid_df)
        
    split_dict = {'N_train' : len(x_train),
                 'N_valid' : len(x_valid),
                 'N_test' : len(x_test),
                 'pos_frac_train' : y_train.sum()/len(y_train),
                 'pos_frac_valid' : y_valid.sum()/len(y_valid),
                 'pos_frac_test' : y_test.sum()/len(y_test),
                 'N_patients_train' : len(y_train_df.patient_id.unique()),
                 'N_patients_valid' : len(y_valid_df.patient_id.unique()),
                 'N_patients_test' : len(y_test_df.patient_id.unique()),
                 'N_admissions_train' : len(y_train_df.hospital_admission_id.unique()),
                 'N_admissions_valid' : len(y_valid_df.hospital_admission_id.unique()),
                 'N_admissions_test' : len(y_test_df.hospital_admission_id.unique()),                 
                 }
    
    print(split_dict)
    
    
    # add the window end timestamps in train, valid and test
    y_train_df['window_end_timestamp'] = pd.to_datetime(y_train_df['admission_timestamp'])+pd.to_timedelta(y_train_df['stop'], 'h')
    y_valid_df['window_end_timestamp'] = pd.to_datetime(y_valid_df['admission_timestamp'])+pd.to_timedelta(y_valid_df['stop'], 'h')
    y_test_df['window_end_timestamp'] = pd.to_datetime(y_test_df['admission_timestamp'])+pd.to_timedelta(y_test_df['stop'], 'h')
    
    
#     del(x_train_df, x_valid_df, x_test_df)
    
    # labs, vitals, med orders best *min_samples_per_leaf=1024-max_leaves=128-n_estimators=100-frac_features_for_clf=0.33-frac_training_samples_per_tree=0.66
    
    # labs, vitals, med orders best 12pm 8pm *min_samples_per_leaf=4096-max_leaves=128-n_estimators=100-frac_features_for_clf=0.66-frac_training_samples_per_tree=0.66
    
    results_file_suffix = ''#'_HIL_trained_on_HUF'
    n_features = len(feature_cols)
    models_dict = {
        'logistic regression' : {'dirname':'skorch_logistic_regression/CustomTimes_10_6', 
                                            'model_constructor': None,
                                            'prefix' : '*scoring=cross_entropy_loss*features_to_include=vitals_only',
                                           'model_color' : 'r', 
                                           'model_marker' : 's'},
                 'lightGBM' : {'dirname': 'lightGBM/CustomTimes_10_6',
                                   'model_constructor' : None,
                                   'prefix' : '*vitals_only',
                                    'model_color' : 'g',
                                    'model_marker' : 'o'},
#                   'MLP 1 layer' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=1),
#                                   'prefix' : '*surrogate_loss_tight*n_hiddens=16*n_layers=1*warm_start=true',
#                                   'model_color' : 'b',
#                                   'model_marker' : '^'},
#                   'MLP 1 layer (BCE + thr search)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=1),
#                                   'prefix' : '*cross_entropy_loss*n_hiddens=16*min_precision=0.5*n_layers=1*warm_start=true*vitals_only',
#                                   'model_color' : 'b',
#                                   'model_marker' : '^'},
#                   'MLP 1 layer (Hinge Bound)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=1),
#                                   'prefix' : '*surrogate_loss_loose*n_hiddens=16*min_precision=0.5*n_layers=1*warm_start=true*vitals_only',
#                                   'model_color' : 'b',
#                                   'model_marker' : '^'},
#                   'MLP 1 layer (Sigmoid Bound)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=1),
#                                   'prefix' : '*surrogate_loss_tight*n_hiddens=16*min_precision=0.5*n_layers=1*warm_start=true*vitals_only',
#                                   'model_color' : 'b',
#                                   'model_marker' : '^'},
#                   'MLP 2 layer' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=8,
#                                                          n_layers=2),
#                                   'prefix' : '*surrogate_loss_tight*n_layers=2*warm_start=true',
#                                    'model_color' : 'k',
#                                   'model_marker' : 'x'},
#                   'MLP 2 layer (BCE + thr search)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=2),
#                                   'prefix' : '*cross_entropy_loss*n_hiddens=16*min_precision=0.5*n_layers=2*warm_start=true',
#                                   'model_color' : 'k',
#                                   'model_marker' : '^'},
#                   'MLP 2 layer (Hinge Bound)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=2),
#                                   'prefix' : '*surrogate_loss_loose*n_hiddens=16*min_precision=0.5*n_layers=2*warm_start=true',
#                                   'model_color' : 'k',
#                                   'model_marker' : '^'},
#                   'MLP 2 layer (Sigmoid Bound)' : {'dirname' : 'skorch_mlp/CustomTimes_10_6',
#                                    'model_constructor' : SkorchMLP(n_features=n_features,
#                                                          n_hiddens=16,
#                                                          n_layers=2),
#                                   'prefix' : '*surrogate_loss_tight*n_hiddens=16*min_precision=0.5*n_layers=2*warm_start=true',
#                                   'model_color' : 'k',
#                                   'model_marker' : '^'},
#                   'MEWS' : {'dirname' : None,
#                                    'model_constructor' : None,
#                                   'prefix' : None,
#                                    'model_color' : 'm',
#                                   'model_marker' : '.'}
                  }
    
    perf_dict_list = []
    pr_f_tr, pr_axs_tr = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_va, pr_axs_va = plt.subplots(1, 1, figsize=(10, 8))
    pr_f_te, pr_axs_te = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_tr, auc_axs_tr = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_va, auc_axs_va = plt.subplots(1, 1, figsize=(10, 8))
    auc_f_te, auc_axs_te = plt.subplots(1, 1, figsize=(10, 8))
    
    
    for model_name in models_dict.keys():
        print('Evaluating with %s'%model_name)
        if model_name != 'MEWS':
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

            # choose the hyperparamater that achieves max auprc
            for i, model_perf_csv in enumerate(model_perf_csvs):
                model_perf_df = pd.read_csv(model_perf_csv)
                thr = model_perf_df['threshold'][0]
                
                model_param_file = model_perf_csv.replace('_perf.csv', '.onnx')
                sess = rt.InferenceSession(model_param_file)
                input_name = sess.get_inputs()[0].name
                proba_label_name = sess.get_outputs()[1].name


                try:
                    y_train_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_train})[0]
                    y_train_proba_vals = np.asarray([i[1] for i in y_train_probas_list_of_dicts])

                    y_valid_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_valid})[0]
                    y_valid_proba_vals = np.asarray([i[1] for i in y_valid_probas_list_of_dicts])

                    y_test_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_test})[0]
                    y_test_proba_vals = np.asarray([i[1] for i in y_test_probas_list_of_dicts])

                except:
                    y_train_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_train})[0]
                    y_train_proba_vals = np.asarray([i[0] for i in y_train_probas_list_of_dicts])

                    y_valid_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_valid})[0]
                    y_valid_proba_vals = np.asarray([i[0] for i in y_valid_probas_list_of_dicts])

                    y_test_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_test})[0]
                    y_test_proba_vals = np.asarray([i[0] for i in y_test_probas_list_of_dicts])               
                
                
                precision_scores_train_G[i] = precision_score(y_train, y_train_proba_vals>=thr)
                recall_scores_train_G[i] = recall_score(y_train, y_train_proba_vals>=thr) 
                auprc_scores_train_G[i] = average_precision_score(y_train, y_train_proba_vals)

                precision_scores_valid_G[i] = precision_score(y_valid, y_valid_proba_vals>=thr)
                recall_scores_valid_G[i] = recall_score(y_valid, y_valid_proba_vals>=thr)
                auprc_scores_valid_G[i] = average_precision_score(y_valid, y_valid_proba_vals)
            
            from IPython import embed; embed()
            best_model_auprc_ind = np.argmax(auprc_scores_valid_G)

            best_model_perf_csv = model_perf_csvs[best_model_auprc_ind]
            best_model_perf_df = pd.read_csv(best_model_perf_csv)
            best_model_threshold = best_model_perf_df['threshold'][0]

            if models_dict[model_name]['model_constructor'] is not None:
                best_model_clf_file = best_model_perf_csv.replace('_perf.csv', 'params.pt')
                best_model_clf = models_dict[model_name]['model_constructor']
                best_model_clf.initialize()
                best_model_clf.load_params(best_model_clf_file)
            else:
                best_model_clf_file = best_model_perf_csv.replace('_perf.csv', '.onnx')
                sess = rt.InferenceSession(best_model_clf_file)

            # predict probas
            try:
                y_train_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_train})[0]
                y_train_proba_vals = np.asarray([i[1] for i in y_train_probas_list_of_dicts])

                y_valid_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_valid})[0]
                y_valid_proba_vals = np.asarray([i[1] for i in y_valid_probas_list_of_dicts])

                y_test_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_test})[0]
                y_test_proba_vals = np.asarray([i[1] for i in y_test_probas_list_of_dicts])
                
            except:
                y_train_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_train})[0]
                y_train_proba_vals = np.asarray([i[0] for i in y_train_probas_list_of_dicts])

                y_valid_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_valid})[0]
                y_valid_proba_vals = np.asarray([i[0] for i in y_valid_probas_list_of_dicts])

                y_test_probas_list_of_dicts = sess.run([proba_label_name], {input_name: x_test})[0]
                y_test_proba_vals = np.asarray([i[0] for i in y_test_probas_list_of_dicts])
        
        else : #If scores are MEWS scores
            y_train_proba_vals = mews_train_df['mews_score']
            y_valid_proba_vals = mews_valid_df['mews_score']
            y_test_proba_vals = mews_test_df['mews_score']
            best_model_clf_file = None
            
        # get precision and recall on train, valid and test
        best_model_auprc_train = average_precision_score(y_train, y_train_proba_vals)
        best_model_auprc_valid = average_precision_score(y_valid, y_valid_proba_vals)
        best_model_auprc_test = average_precision_score(y_test, y_test_proba_vals)
     
        best_model_auroc_train = roc_auc_score(y_train, y_train_proba_vals)
        best_model_auroc_valid = roc_auc_score(y_valid, y_valid_proba_vals)
        best_model_auroc_test = roc_auc_score(y_test, y_test_proba_vals)
        
        perf_dict = {'model' : model_name,
                     'best_model_auprc_train' : best_model_auprc_train,
                     'best_model_auprc_valid' : best_model_auprc_valid,
                     'best_model_auprc_test' : best_model_auprc_test,
                     'best_model_train_pred_probas' : y_train_proba_vals,
                     'best_model_valid_pred_probas' : y_valid_proba_vals,
                     'best_model_test_pred_probas' : y_test_proba_vals,
                     'best_model_file' : best_model_clf_file
                    }
        
        print(perf_dict)
        perf_dict_list.append(perf_dict)
        
        perf_df = pd.DataFrame(perf_dict_list)
        print(perf_df)        
        # create the precision recall plot
        precs_train, recs_train, thresholds_train = precision_recall_curve(y_train, y_train_proba_vals)
        precs_valid, recs_valid, thresholds_valid = precision_recall_curve(y_valid, y_valid_proba_vals)
        precs_test, recs_test, thresholds_test = precision_recall_curve(y_test, y_test_proba_vals)  
        
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba_vals)
        fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_proba_vals)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba_vals)
        
        linewidth=1.0
        pr_axs_tr.plot(recs_train, precs_train, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_train), linewidth=linewidth)
        
        pr_axs_va.plot(recs_valid, precs_valid, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_valid), linewidth=linewidth)
        
        pr_axs_te.plot(recs_test, precs_test, models_dict[model_name]['model_color']+'-o', label = '%s, AUPRC : %.2f'%(model_name, best_model_auprc_test), linewidth=linewidth)
        
        auc_axs_tr.plot(fpr_train, tpr_train, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_train), linewidth=linewidth)
        
        auc_axs_va.plot(fpr_valid, tpr_valid, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_valid), linewidth=linewidth)
        
        auc_axs_te.plot(fpr_test, tpr_test, models_dict[model_name]['model_color']+'-o', label = '%s, AUROC : %.2f'%(model_name, best_model_auroc_test), linewidth=linewidth)
    
    perf_pkl = os.path.join(args.output_dir, 'performance_of_best_clfs%s.pkl'%results_file_suffix)
    perf_df.to_pickle(perf_pkl)
    print('Saved the best model performance on full dataset to :\n%s'%perf_pkl)
    
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
        
        
#         if model_name=='lightGBM':
        unique_probas = np.unique(y_valid_pred_probas)
        thr_grid_G = np.linspace(np.percentile(unique_probas,1), max(unique_probas), 10000)
            
#         else:
#             unique_probas = np.unique(y_valid_pred_probas)
#             thr_grid_linspace = np.linspace(np.percentile(unique_probas,1), max(unique_probas), 3000)
#             thr_grid_G = np.unique(np.union1d(unique_probas, thr_grid_linspace))
        
        
        for split, x_np, y_df, split_prediction_window_ends, y_pred_probas in [('train', x_train, y_train_df,
                                                                                prediction_window_ends_ts_tr, y_train_pred_probas),
                                                                               ('valid', x_valid, y_valid_df,
                                                                                prediction_window_ends_ts_va, y_valid_pred_probas),
                                                                               ('test', x_test, y_test_df,
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
    alarms_csv = os.path.join(args.output_dir, 'alarm_stats_%s%s.csv'%(prediction_freq, results_file_suffix))
    alarms_perf_df.to_pickle(alarms_csv)
    print('Alarm stats saved to : %s'%alarms_csv)
    
    from IPython import embed; embed()
    # plot the distribution of predicted probabilities per week with the lightgbm model
    model_ind = perf_df['model']=='lightGBM'
    
    # get the predicted probabilities on train, valid and test for this classifier
    y_train_pred_probas = perf_df.loc[model_ind, 'best_model_train_pred_probas'].values[0]
    y_valid_pred_probas = perf_df.loc[model_ind, 'best_model_valid_pred_probas'].values[0]
    y_test_pred_probas = perf_df.loc[model_ind, 'best_model_test_pred_probas'].values[0]    
    
    bins = np.arange(0, 1.1, 0.1) 
    N_pred_segments_tr = len(prediction_window_ends_ts_tr)-1
    N_pred_segments_va = len(prediction_window_ends_ts_va)-1
    N_pred_segments_te = len(prediction_window_ends_ts_te)-1
    
    pred_probas_hist_train, pred_probas_hist_valid, pred_probas_hist_test = [np.zeros((N_pred_segments_tr, len(bins)-1)),
                                                                             np.zeros((N_pred_segments_va, len(bins)-1)),
                                                                             np.zeros((N_pred_segments_te, len(bins)-1))]
    
    for split, x_np, y_df, split_prediction_window_ends, y_pred_probas, pred_probas_hist in [('train', x_train, y_train_df,
                                                                                              prediction_window_ends_ts_tr,
                                                                                              y_train_pred_probas,
                                                                                              pred_probas_hist_train),
                                                                                             ('valid', x_valid, y_valid_df,
                                                                                              prediction_window_ends_ts_va,
                                                                                              y_valid_pred_probas,
                                                                                              pred_probas_hist_valid),
                                                                                             ('test', x_test, y_test_df,
                                                                                              prediction_window_ends_ts_te,
                                                                                              y_test_pred_probas,
                                                                                              pred_probas_hist_test)]:

        N_pred_segments = len(split_prediction_window_ends)-1   
        
        for ii in range(N_pred_segments):
            pred_segment_start = split_prediction_window_ends[ii]
            pred_segment_end = split_prediction_window_ends[ii+1]
            
            keep_inds = (y_df['window_end_timestamp']>=pred_segment_start)&(y_df['window_end_timestamp']<=pred_segment_end)
            if keep_inds.sum()>0:
                curr_y_pred_probas = y_pred_probas[keep_inds]
                pred_probas_hist[ii, :], bin_edges = np.histogram(curr_y_pred_probas, bins=bins, density=False)
                
                
    f, axs = plt.subplots(1, 1)
    sns.set_context("notebook", font_scale=1.25)
    sns.set_style("whitegrid")
    colors = ['r', 'g', 'b']
    yticks = np.arange(0, 2300, 100)
    for ii, (split, pred_probas_hist) in enumerate([('train', pred_probas_hist_train), 
                                    ('valid', pred_probas_hist_valid), 
                                    ('test', pred_probas_hist_test)]):
#         axs.plot(bins[1:], np.median(pred_probas_hist, axis=0), label=split)
        pos_err = np.percentile(pred_probas_hist, 95, axis=0)-np.median(pred_probas_hist, axis=0)
        neg_err = np.median(pred_probas_hist, axis=0) - np.percentile(pred_probas_hist, 5, axis=0)
        err = np.vstack([pos_err, neg_err])
        axs[ii].plot(bins[1:], np.median(pred_probas_hist, axis=0), label=split, color='k')
        axs[ii].fill_between(x=bins[1:], y1=np.median(pred_probas_hist, axis=0)-neg_err, 
                             y2=np.median(pred_probas_hist, axis=0)+pos_err, color=colors[ii])
        axs[ii].set_xlabel('Predicted probability')
        axs[0].set_ylabel('Number of predictions')
        axs[ii].set_title(split)
        axs[ii].set_yticks(yticks)
        axs[ii].set_xlim([0, 1])
        axs[ii].set_ylim([0, 1000])
    
    f.savefig('predicted_probas_hist.png')
    from IPython import embed; embed()
    
    # plot feature importance from LightGBM model
    lgbm_model_name = perf_df.iloc[0, -1].replace('.onnx', '_trained_model.joblib')
    lgbm_model = joblib.load(lgbm_model_name)
    feature_importances = lgbm_model.steps[1][1].feature_importances_
    feature_importance_df = pd.DataFrame(sorted(zip(feature_importances, feature_cols)), columns=['Importance','Feature'])
    f, axs = plt.subplots(1, 1, figsize=(8,8))
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.00)
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.sort_values(by="Importance", ascending=False)[:15], ax=axs)
    axs.set_title('LightGBM Features Importances')
    plt.tight_layout()
    f.savefig('lgbm_importances.png')
    
    
    # get the preicisons per shift for all thresholds (S x T)
    perf_per_shift_list = []
    for model_name in models_dict.keys():
        ind = (alarms_perf_df.split=='valid')&(alarms_perf_df.model==model_name)
        valid_recall_SG = alarms_perf_df.loc[ind, 'recall_arr'].values[0] 
        valid_precision_SG = alarms_perf_df.loc[ind, 'precision_arr'].values[0]
        valid_thresh_grid_G = alarms_perf_df.loc[ind, 'threshold_grid'].values[0]

        # Get the median precision across shifts for each threshold
        all_nan_inds = np.isnan(valid_precision_SG).all(axis=0) 
        valid_precision_SG[:, all_nan_inds] = 0

        # for median precisions per shift of 15%, 30% and 45%, get the threshold with the highest recall per shift 
        fixed_precs_list = [0.05, 0.10, 0.20, 0.30]
        neg_tol = 0.02
        pos_tol = 0.05
        median_precision_per_threshold_G = np.nanpercentile(valid_precision_SG, 50, axis=0)
        median_recall_per_threshold_G = np.nanpercentile(valid_recall_SG, 50, axis=0)
        
        for fixed_prec in fixed_precs_list:
            keep_inds = (median_precision_per_threshold_G>(fixed_prec-neg_tol))&(median_precision_per_threshold_G<(fixed_prec+pos_tol))
            if keep_inds.sum()>0:
                curr_median_precisions_G = median_precision_per_threshold_G[keep_inds]
                curr_median_recalls_G = median_recall_per_threshold_G[keep_inds]
                curr_thresh_grid_G = valid_thresh_grid_G[keep_inds]
                chosen_thr_ind = np.argmax(curr_median_recalls_G)

                curr_precisions_SG = valid_precision_SG[:, keep_inds]
                curr_recalls_SG = valid_recall_SG[:, keep_inds]

                chosen_thresh_precisions_S = curr_precisions_SG[:, chosen_thr_ind]
                chosen_thresh_recalls_S = curr_recalls_SG[:, chosen_thr_ind]

                # keep the 5th, 50th and 95th percentile of recall scores for the chosen threshold
                curr_perf_per_shift_dict = {'fixed_precision' : fixed_prec,
                                           'precision_5th_percentile' : np.nanpercentile(chosen_thresh_precisions_S, 5),
                                           'precision_50th_percentile' : np.nanpercentile(chosen_thresh_precisions_S, 50),
                                           'precision_95th_percentile' : np.nanpercentile(chosen_thresh_precisions_S, 95),
                                           'recall_5th_percentile' : np.nanpercentile(chosen_thresh_recalls_S, 5),
                                           'recall_50th_percentile' : np.nanpercentile(chosen_thresh_recalls_S, 50),
                                           'recall_95th_percentile' : np.nanpercentile(chosen_thresh_recalls_S, 95),
                                           'model' : model_name,
                                           'chosen threshold' : curr_thresh_grid_G[chosen_thr_ind],
                                           'precisions_at_chosen_threshold' : chosen_thresh_precisions_S,
                                           'recalls_at_chosen_threshold' : chosen_thresh_recalls_S,
                                           }

                perf_per_shift_list.append(curr_perf_per_shift_dict) 
    
    perf_per_shift_df = pd.DataFrame(perf_per_shift_list)
    
    f, axs = plt.subplots(1, 1, figsize=(8, 8))
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.25)
    xticks = np.arange(0, 1.1, 0.1)
    xticklabels = ['%.1f'%i for i in xticks]
    for model_name in models_dict.keys():
        model_inds = perf_per_shift_df.model==model_name
        x = perf_per_shift_df.loc[model_inds, 'precision_50th_percentile']
        y = perf_per_shift_df.loc[model_inds, 'recall_50th_percentile']
        xerr = np.zeros((2, len(x)))
        xerr[0, :] = x - perf_per_shift_df.loc[model_inds, 'precision_5th_percentile']
        xerr[1, :] = perf_per_shift_df.loc[model_inds, 'precision_95th_percentile'] - x
        yerr = np.zeros((2, len(x)))
        yerr[0, :] = y - perf_per_shift_df.loc[model_inds, 'recall_5th_percentile']
        yerr[1, :] = perf_per_shift_df.loc[model_inds, 'recall_95th_percentile'] - y

        
        markers, caps, bars =axs.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=10, label=model_name, color=models_dict[model_name]['model_color'], marker=models_dict[model_name]['model_marker'])
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        markers.set_alpha(0.5)
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticklabels)
        axs.set_yticks(xticks)
        axs.set_yticklabels(xticklabels)
        axs.legend()
        axs.set_xlabel('Precision in %s shifts'%prediction_freq)
        axs.set_ylabel('Recall in %s day shifts'%prediction_freq)
    
    save_fname = os.path.join(args.output_dir, 'valid_precision_recall_per_shift_errorbars.png')
    print('Performance per shift saved to : \n%s'%save_fname)
    f.savefig(save_fname, bbox_inches='tight', pad_inches=0)
    
    # plot the precision recall over weeks for lightGBM having 20% precision in atleast 50% of the weeks
    model_inds_lgbm = (perf_per_shift_df.model=='lightGBM')&(perf_per_shift_df.fixed_precision==0.2)
    model_inds_mews = (perf_per_shift_df.model=='MEWS')&(perf_per_shift_df.fixed_precision==0.05)
    model_inds_lr = (perf_per_shift_df.model=='logistic regression')&(perf_per_shift_df.fixed_precision==0.2)
 
    f, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True) 
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.25)        
    for model_name, model_inds in [('logistic regression', model_inds_lr),
                                 ('lightGBM', model_inds_lgbm),
                                 ('MEWS', model_inds_mews)]:
        
        # Get precisions and recalls at the chosen threshold
        precs = perf_per_shift_df.loc[model_inds, 'precisions_at_chosen_threshold'].values[0]
        recs = perf_per_shift_df.loc[model_inds, 'recalls_at_chosen_threshold'].values[0]
        
        # get rolling average of precisions
        precs_df = pd.DataFrame({'week' : range(len(precs)), 'precision' : precs})
        rolling_precs = precs_df['precision'].rolling(window=6).mean().values
        
        # get rolling average of recalls
        recs_df = pd.DataFrame({'week' : range(len(recs)), 'recall' : recs})
        rolling_recs = recs_df['recall'].rolling(window=6).mean().values
        


        axs[0].plot(range(len(precs)), precs, models_dict[model_name]['model_color']+'-o', label=model_name)
        axs[1].plot(range(len(recs)), recs, models_dict[model_name]['model_color']+'-o', label=model_name) 
        axs[0].plot(range(len(precs)), rolling_precs, models_dict[model_name]['model_color']+'-', alpha=0.5) 
        axs[1].plot(range(len(recs)), rolling_recs, models_dict[model_name]['model_color']+'-', alpha=0.5) 
    


    axs[0].legend(fontsize=fontsize) 
    axs[1].legend(fontsize=fontsize)

    axs[0].set_ylabel('precision', fontsize=fontsize)
    axs[1].set_ylabel('recall', fontsize=fontsize)
    axs[1].set_xlabel('Week', fontsize=fontsize)
    plt.suptitle('Precision - Recall Over Weeks', fontsize=fontsize+3)
    save_fname = os.path.join(args.output_dir, 'precision_recall_over_weeks.png')
    print('precision recall over weeks saved to : \n%s'%save_fname)
    f.savefig(save_fname, bbox_inches='tight', pad_inches=0)
    from IPython import embed; embed()
    '''
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
