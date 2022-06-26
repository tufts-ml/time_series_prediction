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
                             roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score)
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
    parser.add_argument('--include_medications', default='True', type=str,
                       help='temporary flag to add/not add medictaions')
    
    
    args = parser.parse_args()
    
    models = ['logistic_regression', 'random_forest', 'mlp']
    clf_models_dict = dict.fromkeys(models)
    for model in models:
        clf_model_file = os.path.join(args.clf_models_dir, model, '%s_trained_model.joblib'%model)
        clf_model = load(clf_model_file)
        clf_models_dict[model] = clf_model
    clf_models_dict['mews'] = pd.read_csv(os.path.join(args.clf_models_dir, 'mews', 'mews_best_threshold.csv' ))
    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.clf_train_test_split_dir, 'y_test.csv.gz'))
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
    _,_,_,_,demographics_df, demographics_data_dict,_,_,_,_ = get_preprocessed_data(args.preproc_data_dir)
    
    prctile_vals = [5, 50, 95]
    random_seed_list = args.random_seed_list.split(' ')
    perf_df = pd.DataFrame()
    for p, tslice in enumerate(tslices_list):
        tslice_folder = tslice_folders + tslice
        collapsed_tslice_folder = collapsed_tslice_folders + tslice
        # get test set collapsed labs and vitals
        collapsed_vitals_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedVitalsPerSequence.csv.gz'))
        collapsed_labs_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedLabsPerSequence.csv.gz'))
        mews_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'MewsScoresPerSequence.csv.gz'))
        outcomes_df = pd.read_csv(os.path.join(tslice_folder,
                                               'clinical_deterioration_outcomes_filtered_%s_hours.csv.gz'%tslice))
        collapsed_vitals_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedVitalsPerSequence.json'))
        collapsed_labs_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedLabsPerSequence.json'))
        mews_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_MewsScoresPerSequence.json'))
        test_vitals_df = pd.merge(collapsed_vitals_df, y_test_ids_df, on=id_cols)
        test_labs_df = pd.merge(collapsed_labs_df, y_test_ids_df, on=id_cols)
        test_mews_df = pd.merge(mews_df, y_test_ids_df, on=id_cols)
        
        if args.include_medications=='True':
            collapsed_medications_df = pd.read_csv(os.path.join(collapsed_tslice_folder, 'CollapsedMedicationsPerSequence.csv.gz'))
            collapsed_medications_data_dict = load_data_dict_json(os.path.join(collapsed_tslice_folder, 'Spec_CollapsedMedicationsPerSequence.json'))
            test_medications_df = pd.merge(collapsed_medications_df, y_test_ids_df, on=id_cols)
            
            # merge them
            test_collapsed_features_df = pd.merge(pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner'),
                                                  test_medications_df,on=id_cols, how='inner')
            data_dicts_list = [collapsed_vitals_data_dict, 
                               collapsed_labs_data_dict,
                               collapsed_medications_data_dict,
                               demographics_data_dict]
            
        else :
            test_collapsed_features_df = pd.merge(test_vitals_df, test_labs_df, 
                                                           on=id_cols, how='inner')
            
            data_dicts_list = [collapsed_vitals_data_dict,
                               collapsed_labs_data_dict,
                               demographics_data_dict]
        
        test_features_df = pd.merge(test_collapsed_features_df, demographics_df, on=id_cols)
        if p==0:
            test_features_dict = merge_data_dicts(data_dicts_list)

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
            precision_score_np = np.zeros(len(random_seed_list))
            recall_score_np = np.zeros(len(random_seed_list))
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
                precision_score_np[k] = precision_score(curr_y_test, y_pred)
                recall_score_np[k] = recall_score(curr_y_test, y_pred)
            
            print('Median AUROC : %.2f'%np.median(roc_auc_np))
            print('Median average precision : %.3f'%np.median(avg_precision_np))
            print('Median precision score : %.3f'%np.median(precision_score_np))
            print('Median recall score : %.3f'%np.median(recall_score_np))
            print('Median balanced accuracy score : %.3f'%np.median(balanced_accuracy_np))

            for prctile in prctile_vals:
                row_dict = dict()
                row_dict['model'] = model
                row_dict['percentile'] = prctile
                row_dict['tslice'] = tslice
                row_dict['roc_auc'] = np.percentile(roc_auc_np, prctile)
                row_dict['balanced_accuracy'] = np.percentile(balanced_accuracy_np, prctile)
                row_dict['log_loss'] = np.percentile(log_loss_np, prctile)
                row_dict['average_precision'] = np.percentile(avg_precision_np, prctile)
                row_dict['precision_score'] = np.percentile(precision_score_np, prctile)
                row_dict['recall_score'] = np.percentile(recall_score_np, prctile)

                perf_df = perf_df.append(row_dict, ignore_index=True)        
                
    perf_csv = os.path.join(args.output_dir, 'lr_rf_mlp_mews_pertslice_performance.csv')
    print('Saving lr, rf, mlp, mews per-tslice performance to %s'%perf_csv)
    perf_df.to_csv(perf_csv, index=False)
  
