'''
Evaluate probability of clinical deterioration over time with a single trained classifier on test patients with 
'''
import os
import numpy as np
import pandas as pd
from joblib import dump, load
import sys
from matplotlib.pyplot import cm
sys.path.append(os.path.join(os.path.abspath('../'), 'src'))

DEFAULT_PROJECT_REPO = os.path.sep.join(__file__.split(os.path.sep)[:-2])
PROJECT_REPO_DIR = os.path.abspath(
    os.environ.get('PROJECT_REPO_DIR', DEFAULT_PROJECT_REPO))

sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_REPO_DIR, 'src', 'rnn'))

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
from impute_missing_values import get_time_since_last_observed_features
from dataset_loader import TidySequentialDataCSVLoader
from RNNBinaryClassifier import RNNBinaryClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shallow_clf_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--shallow_clf_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--rnn_models_dir', default=None, type=str,
                        help='Directory where classifier models are saved')
    parser.add_argument('--rnn_train_test_split_dir', default=None, type=str,
                        help='Directory where the train-test split data for the classifier is saved')
    parser.add_argument('--preproc_data_dir', type=str,
                        help='folder where the preprocessed data is stored')
    parser.add_argument('--outcome_column_name', default='clinical_deterioration_outcome', type=str,
                       help='name of outcome column in test dataframe')
    parser.add_argument('--output_dir', default='', type=str,
                       help='dir to save plots')
    
    args = parser.parse_args()
    
    
    ## get the test patient id's
    # get the test set's csv and dict
    y_test_df = pd.read_csv(os.path.join(args.shallow_clf_train_test_split_dir, 'y_test.csv'))
    y_test_dict_file = os.path.join(args.shallow_clf_train_test_split_dir, 'y_dict.json')
    
    y_test_dict = load_data_dict_json(y_test_dict_file)
    id_cols = parse_id_cols(y_test_dict)
    
#     rnn_train_test_split_dir=args.rnn_train_test_split_dir.replace(' ', '')
    ## get the data dict of sequence features with mask features
    x_test_dict_file = os.path.join(args.rnn_train_test_split_dir,'x_dict.json')
    x_test_dict = load_data_dict_json(x_test_dict_file)
    feature_cols_with_mask_features = parse_feature_cols(x_test_dict)
    x_train_df =  pd.read_csv(os.path.join(args.rnn_train_test_split_dir,'x_train.csv'))
    
    # load shallow models
    shallow_models = ['logistic_regression', 'random_forest']
    clf_models_dict = dict.fromkeys(shallow_models)
    for model in shallow_models:
        clf_model_file = os.path.join(args.shallow_clf_models_dir, model, '%s_trained_model.joblib'%model)
        clf_model = load(clf_model_file)
        clf_models_dict[model] = clf_model
    
    # load rnn
    rnn = RNNBinaryClassifier(module__rnn_type='LSTM',
                              module__n_layers=2,
                              module__n_hiddens=32,
                             module__n_inputs=len(feature_cols_with_mask_features))
    rnn.initialize()
    best_model_prefix = 'hiddens=32-layers=2-lr=0.005-dropout=0.3-weight_decay=1e-06'
    rnn.load_params(f_params=os.path.join(args.rnn_models_dir,
                                          best_model_prefix+'params.pt'),
                    f_optimizer=os.path.join(args.rnn_models_dir,
                                             best_model_prefix+'optimizer.pt'),
                    f_history=os.path.join(args.rnn_models_dir,
                                           best_model_prefix+'history.json'))

    clf_models_dict['rnn'] = rnn
    
    models = shallow_models + ['rnn']
    
    # get id's  of 3 patients with clinical deterioration and 3 different stay lengths  
    chosen_thresh = 192
    chosen_stay_ids_df = y_test_df[(y_test_df[args.outcome_column_name]==1)][id_cols].copy().reset_index(drop=True)

    chosen_stay_ids_df = chosen_stay_ids_df.drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    # for each patient get their vitals, labs, demographics
    labs_df, labs_data_dict, vitals_df, vitals_data_dict, \
        demographics_df, demographics_data_dict, outcomes_df, outcomes_data_dict = get_preprocessed_data(args.preproc_data_dir)

    vitals = parse_feature_cols(vitals_data_dict)
    labs = parse_feature_cols(labs_data_dict)
    
    chosen_stay_labs_df = pd.merge(labs_df, chosen_stay_ids_df, on=id_cols, how='inner')
    chosen_stay_vitals_df = pd.merge(vitals_df, chosen_stay_ids_df, on=id_cols, how='inner')
    chosen_stay_highfreq_df = pd.merge(chosen_stay_labs_df, chosen_stay_vitals_df, on = id_cols + ['hours_since_admission', 
                                                                                                'timestamp'], how='outer')
    
    
    highfreq_features_dict = merge_data_dicts([labs_data_dict, vitals_data_dict])
    highfreq_features_dict['fields'] = highfreq_features_dict['schema']['fields']
    
    # choose a subject
    chosen_short_stay_subj_list = ['14343967', '18115638', '18826316', '17245153', 
                             '17557700', '11212084', '11163358', '17684794', 
                             '12751842', '11528888', '1379931', '17745211', 
                             '12862019', '14201044', '14917356', '17682462', 
                             '1339889', '17995864', '15787542', '13007083', 
                             '18239690', '11692208', '19352552', '19438165', 
                             '14858518']
    chosen_long_stay_subj_list = ['12702290', '19160387', '19806342', '19222313', '17863017']
    
    chosen_stay_subj_list = chosen_short_stay_subj_list + chosen_long_stay_subj_list
#     chosen_stay_subj_list = chosen_long_stay_subj_list[0:2]
    for idx in chosen_stay_subj_list:
#         try:
        chosen_stay_subj = chosen_stay_ids_df[chosen_stay_ids_df.hospital_admission_id == int(idx)]   
        print('chosen patient stay %s'%(idx))
        # get all the features for the chosen subject
        chosen_stay_subj_highfreq_features_df = pd.merge(chosen_stay_highfreq_df, chosen_stay_subj)
        chosen_stay_subj_vitals_df = pd.merge(chosen_stay_vitals_df, chosen_stay_subj)
        chosen_stay_outcomes_df = pd.merge(outcomes_df, chosen_stay_subj)
        
        time_col = parse_time_col(highfreq_features_dict['schema'])
        
        # collapse features in slices of first 16 hrs, first 24 hours, ... full stay 
        eval_times = np.arange(24, chosen_stay_subj_highfreq_features_df[time_col].max(), 8)
        if max(eval_times) < chosen_stay_subj_highfreq_features_df[time_col].max():
            eval_times = np.insert(eval_times, len(eval_times), 
                                   chosen_stay_subj_highfreq_features_df[time_col].max())
        collapse_range_features = "std hours_since_measured present slope median min max"
        range_pairs = "[('50%','100%'), ('0%','100%'), ('T-16h','T-0h'), ('T-24h','T-0h')]"

        # create data dict of collapsed features and merge it with demographics data dict to create data dict of all features
        collapsed_features_dict = update_data_dict_collapse(highfreq_features_dict, 
                                                            collapse_range_features, range_pairs)
        collapsed_features_data_dict = merge_data_dicts([collapsed_features_dict, demographics_data_dict]) 
        sequence_features_data_dict = merge_data_dicts([highfreq_features_dict, demographics_data_dict])
        
        eval_times_collapsed_features_df_list = []
        eval_times_sequence_features_df_list = []
        for eval_t in eval_times:
            t_inds = chosen_stay_subj_highfreq_features_df[time_col]<=eval_t
            curr_highfreq_features_df = chosen_stay_subj_highfreq_features_df.loc[t_inds, :].copy().reset_index(drop=True)
            curr_highfreq_features_df.sort_values(by=time_col, inplace=True)
            tstops_df = chosen_stay_subj.copy()
            tstops_df.loc[:, 'tstop'] = eval_t

            # collapse features for chosen patient-stay
            curr_collapsed_df = collapse_np(curr_highfreq_features_df, highfreq_features_dict, collapse_range_features, range_pairs, tstops_df)
            curr_collapsed_features_df = pd.merge(curr_collapsed_df, demographics_df, on=id_cols, how='inner')
            curr_sequence_features_df = pd.merge(curr_highfreq_features_df, demographics_df, on=id_cols, how='inner')
            
            eval_times_collapsed_features_df_list.append(curr_collapsed_features_df)
            eval_times_sequence_features_df_list.append(curr_sequence_features_df)
        
        
        proba_deterioration_over_time = np.zeros((len(clf_models_dict), len(eval_times_collapsed_features_df_list)))
        total_missing_features_over_time = np.zeros(len(eval_times_collapsed_features_df_list))
        for q, collapsed_features_df in enumerate(eval_times_collapsed_features_df_list):
           
            for p, model in enumerate(models):
                if model == 'rnn':
                    curr_sequence_features_df = eval_times_sequence_features_df_list[q]
                    
                    sequence_feature_cols = parse_feature_cols(sequence_features_data_dict['schema'])
                    
                    # get the mask features
                    print('Adding missing values mask as features...')
                    for feature_col in sequence_feature_cols:
                        curr_sequence_features_df.loc[:, 'mask_'+feature_col] = (~curr_sequence_features_df[feature_col].isna())*1.0

                    print('Adding time since last missing value is observed as features...')
                    curr_sequence_features_df = get_time_since_last_observed_features(curr_sequence_features_df, 
                                                                                      id_cols, time_col, sequence_feature_cols)
                    
                    
                    # impute missing values in the test features
                    curr_sequence_features_df = curr_sequence_features_df.groupby(id_cols).apply(lambda x: x.fillna(method='pad')).copy()

                    for feature_col in sequence_feature_cols:
                        curr_sequence_features_df[feature_col].fillna(curr_sequence_features_df[feature_col].mean(), inplace=True)  
                    
                    for feature_col in sequence_feature_cols:
                        curr_sequence_features_df[feature_col].fillna(x_train_df[feature_col].mean(), inplace=True) 
                    # load test data with TidySequentialDataLoader
                    test_vitals = TidySequentialDataCSVLoader(
                        x_csv_path=curr_sequence_features_df,
                        y_csv_path=chosen_stay_outcomes_df,
                        x_col_names=feature_cols_with_mask_features,
                        idx_col_names=id_cols,
                        y_col_name=args.outcome_column_name,
                        y_label_type='per_sequence'
                    )    

                    # predict on test data
                    x_test, y_test = test_vitals.get_batch_data(batch_id=0)


                    per_feature_scaling = np.load(os.path.join(args.rnn_models_dir, 'per_feature_scaling.npy'))
                    for f in range(x_test.shape[2]):
                        x_test[:,:,f] = x_test[:,:,f]/per_feature_scaling[f]
                    
                    mask_feature_cols = [i for i in feature_cols_with_mask_features if 'mask' in i]
                    
                    total_missing_features_over_time[q]=(curr_sequence_features_df[mask_feature_cols]==0).sum().sum()
                else:
                    collapsed_feature_cols = parse_feature_cols(collapsed_features_data_dict['schema'])
                    x_test = collapsed_features_df[collapsed_feature_cols].values.astype(np.float32)
                proba_deterioration_over_time[p,q] = clf_models_dict[model].predict_proba(x_test)[0][1]
        
                
        ## plot vitals collected over time
        plot_vitals_dict = {'heart_rate':'bpm','o2_sat':'%',
                  'blood_glucose_concentration':'/vol', 'body_temperature':'$^\circ$C', 
                  'systolic_blood_pressure':'mmHg', 'diastolic_blood_pressure':'mmHg'
                }
        plot_vitals = list(plot_vitals_dict.keys())
        f, axs = plt.subplots(len(plot_vitals)+2,1,figsize=(8,14), sharex=True)
        plot_subj = chosen_stay_subj_vitals_df[chosen_stay_subj_vitals_df.hospital_admission_id==int(idx)]
        color=cm.rainbow(np.linspace(0,0.8,len(plot_vitals)))
        fontsize=12
        for p,vital in enumerate(plot_vitals):

            x = plot_subj[time_col]
            y = plot_subj[vital]
            mask = np.isfinite(y)
            axs[p].plot(x[mask], y[mask], c=color[p,:], marker = '*')
            axs[p].set_title(vital, fontsize=fontsize)
            axs[p].set_xlabel('hours since admission', fontsize=fontsize)
            axs[p].set_ylabel(plot_vitals_dict[vital], fontsize=fontsize)
#             axs[p].set_xticklabels([''])
            axs[p].set_xlabel('')        
        
        axs[p+1].plot(eval_times, total_missing_features_over_time)
        axs[p+1].set_xlabel('') 
        axs[p+1].set_title('Total missing features (summed over time)', fontsize=fontsize)
        axs[p+1].set_ylim([0, 3000])
        ## plot probability of deterioration over time
        markers = ['r*-', 'bo-', 'k.-']
        for k, model in enumerate(models):
            axs[-1].plot(eval_times, proba_deterioration_over_time[k,:], markers[k], markersize=5, label=model)
        axs[-1].set_ylim([0, 1.2])
        axs[-1].set_xlim([0, max(x)+1])
        axs[-1].set_xlabel('Hours from Admission', fontsize=fontsize)
        axs[-1].set_title('Probability of Deterioration', fontsize=fontsize)
        axs[-1].tick_params(labelsize=fontsize)
        axs[-1].legend(loc='upper left', fontsize=fontsize-2, borderaxespad=0)
        
#         plt.legend()
        plt.subplots_adjust(hspace=0.4)
        if chosen_stay_subj_highfreq_features_df[time_col].max() > chosen_thresh:
            fig_file = os.path.join(args.output_dir, 'long_stays_predict_proba', '%s_proba_deterioration.pdf'%(int(chosen_stay_subj.hospital_admission_id)))
        else:
            fig_file = os.path.join(args.output_dir, 'short_stays_predict_proba', '%s_proba_deterioration.pdf'%(int(chosen_stay_subj.hospital_admission_id)))
        print('Performance fig saved to : \n%s'%fig_file)
        f.savefig(fig_file, bbox_inches='tight')
        plt.close()