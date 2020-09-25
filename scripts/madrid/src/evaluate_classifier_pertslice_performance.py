'''
Evaluate performance of a single trained classifier on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))

#import LR model before importing other packages because joblib files act weird when certain packages are loaded
from feature_transformation import *
from merge_features_all_tslices import merge_data_dicts
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             average_precision_score, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve, precision_recall_curve)
from utils import load_data_dict_json
import ast
from filter_admissions_by_tslice import get_preprocessed_data
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--collapsed_tslice_folder', type=str, 
                        help='folder where collapsed features from each tslice are stored')
    parser.add_argument('--tslice_folder', type=str, 
                        help='folder where features filtered by tslice are stored')
    parser.add_argument('--evaluation_tslices', type=str,
                        help='evaluation tslices separated by spaces')
    parser.add_argument('--preproc_data_dir', type=str,
                        help='folder where the preprocessed data is stored')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--random_seed_list', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--output_dir', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    
    
    args = parser.parse_args()
    
    models = ['logistic_regression', 'random_forest']
    clf_models_dict = dict.fromkeys(models)
    for model in models:
        clf_model_file = os.path.join(args.clf_models_dir, model, '%s_trained_model.joblib'%model)
        clf_model = load(clf_model_file)
        clf_models_dict[model] = clf_model
    clf_models_dict['mews'] = pd.read_csv(os.path.join(args.clf_models_dir, 'mews', 'mews_best_threshold.csv' ))
    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.clf_train_test_split_dir, 'y_dict.json')
    
    # import the y dict to get the id cols
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    
    tslice_folders = os.path.join(args.tslice_folder, 'TSLICE=')
    collapsed_tslice_folders = os.path.join(args.collapsed_tslice_folder, 'TSLICE=')
    outcome_col = args.outcome_column_name
    tslices_list = args.evaluation_tslices.split(' ')
    y_test_ids_df = y_test_df[id_cols].drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    # get demographics csv and data_dict
    # for each patient get their vitals, labs, demographics
    _,_,_,_,demographics_df, demographics_data_dict,_,_ = get_preprocessed_data(args.preproc_data_dir)
    
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice
        collapsed_tslice_folder = collapsed_tslice_folders + tslice
        # get test set collapsed labs and vitals
        collapsed_vitals_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedVitalsPerSequence.csv'))
        collapsed_labs_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedLabsPerSequence.csv'))
        mews_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'MewsScoresPerSequence.csv'))
        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'clinical_deterioration_outcomes_filtered_%s_hours.csv'%tslice))
        collapsed_vitals_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedVitalsPerSequence.json'))
        collapsed_labs_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedLabsPerSequence.json'))
        mews_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_MewsScoresPerSequence.json'))
        test_vitals_df = pd.merge(collapsed_vitals_df, y_test_ids_df, on=id_cols)
        test_labs_df = pd.merge(collapsed_labs_df, y_test_ids_df, on=id_cols)
        test_mews_df = pd.merge(mews_df, y_test_ids_df, on=id_cols)


        # merge them
        test_collapsed_features_df = pd.merge(test_vitals_df, test_labs_df, on=id_cols, how='inner')
        test_features_df = pd.merge(test_collapsed_features_df, demographics_df, on=id_cols)
        if p==0:
            test_features_dict = merge_data_dicts([collapsed_vitals_data_dict, 
                                                 collapsed_labs_data_dict, demographics_data_dict])

        test_outcomes_df = pd.merge(test_features_df[id_cols], outcomes_df, on=id_cols, how='inner')

    #     # get performance metrics
        feature_cols = parse_feature_cols(test_features_dict['schema'])
        mews_score_col = parse_feature_cols(mews_data_dict['schema'])
        x_test = test_features_df[feature_cols].values
        y_test = test_outcomes_df[outcome_col].values
        mews_test = test_mews_df[mews_score_col].values
        
        # bootstrap test set inds without replacement
        for model in clf_models_dict.keys():
            print('Evaluating %s on tslice=%s'%(model, tslice))
            roc_auc_np = np.zeros(len(random_seed_list))
            balanced_accuracy_np = np.zeros(len(random_seed_list))
            log_loss_np = np.zeros(len(random_seed_list))
            avg_precision_np = np.zeros(len(random_seed_list))
            for k, seed in enumerate(random_seed_list):
                random.seed(int(seed))
                rnd_inds = random.sample(range(x_test.shape[0]), int(0.8*x_test.shape[0])) 
                curr_y_test = y_test[rnd_inds]
                    
                if model== 'mews':
                    y_score = mews_test[rnd_inds]
                    mews_best_threshold = float(clf_models_dict['mews'].loc[0])
                    y_pred = (y_score > mews_best_threshold)*1
                    y_pred_proba = y_score/np.max(y_score)

                else:
                    curr_x_test = x_test[rnd_inds, :]
                    y_pred = clf_models_dict[model].predict(curr_x_test)
                    y_pred_proba = clf_models_dict[model].predict_proba(curr_x_test)[:, 1]
                    y_score = y_pred_proba

                roc_auc_np[k] = roc_auc_score(curr_y_test, y_score)
                balanced_accuracy_np[k] = balanced_accuracy_score(curr_y_test, y_pred)
                log_loss_np[k] = log_loss(curr_y_test, y_pred_proba, normalize=True) / np.log(2)
                avg_precision_np[k] = average_precision_score(curr_y_test, y_score)

            for prctile in prctile_vals:
                row_dict = dict()
                row_dict['model'] = model
                row_dict['percentile'] = prctile
                row_dict['tslice'] = tslice
                row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)

                perf_df = perf_df.append(row_dict, ignore_index=True)        
    
    
    # plot performance as a function of hours from admission, % data observed and hours from deterioration  
    print('Saving plots to : \n%s'%(args.clf_models_dir))
    neg_tslices_list = [i for i in tslices_list if '-' in i]
    perc_tslices_list = [i for i in tslices_list if '%' in i]

    pos_tslices_list = []
    for i in tslices_list:
        if (i not in neg_tslices_list) & (i not in perc_tslices_list):
            pos_tslices_list.append(i)
    
    tslice_grouped_list = [perc_tslices_list, pos_tslices_list, neg_tslices_list]
    perf_measures = ['roc_auc', 'average_precision']
    fontsize=18
    for x in tslice_grouped_list:
        for perf_measure in perf_measures:
            f, axs = plt.subplots(figsize=[10,8])
            model_colors=['r', 'b', 'g']
            for p, model in enumerate(perf_df.model.unique()):
                inds = (perf_df.model==model) & (perf_df.tslice.isin(x))
                cur_df = perf_df.loc[inds, :].copy()

                y = cur_df.loc[cur_df.percentile==50, perf_measure].values
                y_err  = np.zeros((2, len(x)))
                y_err[0,:] = y - cur_df.loc[cur_df.percentile==5, perf_measure].values
                y_err[1,:] = cur_df.loc[cur_df.percentile==95, perf_measure].values - y

                axs.errorbar(x=x, y=y, yerr=y_err, label=model, fmt='.-', linewidth=3, color=model_colors[p])
                axs.plot(x, y, '.', markersize=20, color=model_colors[p])

            if x==neg_tslices_list:
                axs.invert_xaxis()
                axs.set_xlabel('Hours Until Deterioration', fontsize=fontsize)
                fig_aka='perf_%s_neg_hours.pdf'%perf_measure
            elif x==perc_tslices_list:
                axs.set_xlabel('% Data Recorded', fontsize=fontsize)
                fig_aka='perf_%s_perc_hours.pdf'%perf_measure
            elif x==pos_tslices_list:
                axs.set_xlabel('Hours From Admission', fontsize=fontsize)
                fig_aka='perf_%s_pos_hours.pdf'%perf_measure
                
            if perf_measure == 'roc_auc':
                axs.set_ylim([0.48, 1])
            elif perf_measure == 'average_precision':
                axs.set_ylim([0, 0.6])

            axs.set_ylabel(perf_measure, fontsize=fontsize)
            axs.legend(fontsize=fontsize-2, loc='upper left')
            axs.tick_params(labelsize=fontsize)
            fig_name = os.path.join(args.output_dir, fig_aka)
            f.savefig(fig_name)
            print('Saved results to %s'%fig_name)