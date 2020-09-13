'''
Old Script of assessing classifier models that were trained and evaluated individually on multiple patient-stay-slices. (Deprecated)
'''
import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_performance_dir', default=None, type=str,
        help='Directory where classifeir performance csvs are saved')

    args = parser.parse_args()
    models = ['random_forest', 'logistic_regression', 'mews']
    #models = ['random_forest']
    # get performance vals for classifiers at tstep=2,4,6,10,14,-1 etc. -1 is full history     
    final_perf_df_list=list()
    for model in models:
        model_tstep_folders = glob.glob(os.path.join(args.clf_performance_dir, model, '*'))
        for tstep_folder in model_tstep_folders:
            #perf_csv = os.path.join(tstep_folder, 'performance_df.csv')
            perf_csvs = glob.glob(os.path.join(tstep_folder, 'performance_df_random_seed*.csv'))
            for perf_csv in perf_csvs:
                if os.path.exists(perf_csv):
                    perf_df = pd.read_csv(perf_csv)
                    perf_df.loc[:,'model']=model
                    #perf_df.loc[:,'tstep']=re.findall('\-?\d+',os.path.basename(tstep_folder))[0]
                    perf_df.loc[:,'tstep']=os.path.basename(tstep_folder).replace('TSTEP=', '')
                    perf_df.loc[:,'random_seed']=re.findall('\-?\d+',os.path.basename(perf_csv))[0]
                    perf_df.drop(columns='confusion_html',inplace=True)
                    final_perf_df_cols = perf_df.columns
                    final_perf_df_list.append(perf_df.to_numpy())

    final_perf_df = pd.DataFrame(np.vstack(final_perf_df_list), columns = final_perf_df_cols)
    #final_perf_df['tstep'] = final_perf_df['tstep'].astype(float)
    
    # create perf metrics for random guessing
    final_perf_df_random_model = final_perf_df[final_perf_df.model=='random_forest'].copy()
    final_perf_df_random_model.loc[:, 'model'] = 'chance'
    final_perf_df_random_model.loc[:, 'average_precision'] = final_perf_df_random_model.loc[:,'frac_labels_positive']
    final_perf_df_random_model.loc[:, 'AUROC'] = 0.5
    final_perf_df_random_model.loc[:, 'f1_score'] = 0.5
    final_perf_df_random_model.loc[:, 'balanced_accuracy'] = 0.5
    final_perf_df_random_model.loc[:, 'accuracy'] = 0.5
    final_perf_df_random_model.loc[:, 'cross_entropy_base2'] = np.nan
    final_perf_df = final_perf_df.append(final_perf_df_random_model)

    # manipulation to make negative tsteps be plotted latter in the graph in increasing order
    drop_inds = final_perf_df.tstep=='full'
    final_perf_df = final_perf_df.loc[~drop_inds, :]
    final_perf_df.loc[:,'tstep'] = pd.to_numeric(final_perf_df['tstep'], errors='coerce')
    '''
    full_history_temp_val = 2*final_perf_df['tstep'].max()+20
    final_perf_df.loc[final_perf_df['tstep'].isna(), 'tstep'] = full_history_temp_val
    neg_tstep_inds = final_perf_df['tstep']<0
    neg_tstep_vals = final_perf_df.loc[neg_tstep_inds, 'tstep']
    neg_tstep_temp_vals = final_perf_df.loc[neg_tstep_inds, 
            'tstep'] + full_history_temp_val
    final_perf_df.loc[neg_tstep_inds, 'tstep'] = neg_tstep_temp_vals
    
    # create a dict to map real and temp tstep vals
    neg_tstep_vals_dict = dict(zip(neg_tstep_temp_vals.to_list(), neg_tstep_vals.to_list()))
    neg_tstep_vals_dict[full_history_temp_val] = 'all'
    '''

    pos_final_perf_df = final_perf_df[final_perf_df.tstep>0].copy()
    neg_final_perf_df = final_perf_df[final_perf_df.tstep<0].copy()

    # plot selected metrics
    for segment, perf_df in [('first', pos_final_perf_df), ('last', neg_final_perf_df)]:
        plot_metrics = [ 'balanced_accuracy', 'f1_score', 'average_precision', 'AUROC']
        for f_idx, plot_metric in enumerate(plot_metrics):
            f, axs = plt.subplots()
            #axs_list = [ax for sublist in axs for ax in sublist]
            sns.pointplot(x='tstep', y=plot_metric, hue = 'model', data=perf_df[perf_df.split_name=='test'],ax=axs)
            ticklabels = [item.get_text() for item in axs.get_xticklabels()]
            ticklabels_new = [ticklabel.replace('-', '') for ticklabel in ticklabels]
            axs.set_xticklabels(ticklabels_new)
            if segment=='first':
                axs.set_xlabel('Hours from admission')
            else:
                axs.set_xlabel('Hours until deterioration')
            if plot_metric == 'AUROC':
                axs.set_ylim([0.45, 0.95])
            elif plot_metric == 'average_precision':
                axs.set_ylim([0, 0.3])
            plt.suptitle('Prediction of Clinical Deterioration From Collapsed Features')
            results_fig = os.path.join(args.clf_performance_dir,  '{segment}_tstep_performance_{plot_metric}.pdf'.format(segment=segment, plot_metric=plot_metric))
            print('Saving performance results fig to %s'%results_fig)
            f.savefig(results_fig)

    # get the train test count plits for each tstep
    fig, axs = plt.subplots()
    sns.barplot(x='tstep', y='n_examples', hue='split_name', data=pos_final_perf_df[pos_final_perf_df.model=='random_forest'], ax=axs)
    plt.suptitle('Train-Test Splits Per Slice')
    ticklabels = [item.get_text() for item in axs.get_xticklabels()]
    print('Saved train-test split counts to %s'%(os.path.join(args.clf_performance_dir, 'tstep_examples_count_distribution.pdf')))
    fig.savefig(os.path.join(args.clf_performance_dir, 'tstep_examples_count_distribution.pdf'))
