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
    models = ['logistic_regression', 'random_forest']

    # get performance vals for classifiers at tstep=2,4,6,10,14,-1 etc. -1 is full history     
    final_perf_df_list=list()
    for model in models:
        model_tstep_folders = glob.glob(os.path.join(args.clf_performance_dir, model, '*'))
        for tstep_folder in model_tstep_folders:
            perf_csv = os.path.join(tstep_folder, 'performance_df.csv')
            if os.path.exists(perf_csv):
                 perf_df = pd.read_csv(perf_csv)
                 perf_df.loc[:,'model']=model
                 perf_df.loc[:,'tstep']=re.findall('\-?\d+',os.path.basename(tstep_folder))[0]
                 perf_df.drop(columns='confusion_html',inplace=True)
                 final_perf_df_cols = perf_df.columns
                 final_perf_df_list.append(perf_df.to_numpy())

    final_perf_df = pd.DataFrame(np.vstack(final_perf_df_list), columns = final_perf_df_cols)
    final_perf_df['tstep'] = final_perf_df['tstep'].astype(float) 
    # plot selected metrics
    plot_metrics = ['cross_entropy_base2', 'accuracy', 'balanced_accuracy', 'f1_score', 'average_precision', 'AUROC']
    f, axs = plt.subplots(3,2, figsize=(18,15))
    axs_list = [ax for sublist in axs for ax in sublist]
    split_marker_dict = {'train':'x', 'test':'.'}
    split_linestyle_dict = {'train':'--', 'test':'-'}
    model_color_dict={'logistic_regression': 'b', 'random_forest':'r'}
    for ax_idx, plot_metric in enumerate(plot_metrics):
        for model in final_perf_df.model.unique():
            for split in final_perf_df.split_name.unique():
                metric_model_split_idx = (final_perf_df['split_name']==split) & (final_perf_df['model']==model)
                metric_vals = np.asarray(final_perf_df.loc[metric_model_split_idx,plot_metric])
                tsteps = np.asarray(final_perf_df.loc[metric_model_split_idx,'tstep'])
                
                # move the results for full history later to right of the axis
                tsteps[tsteps==-1]=tsteps.max()+5
                sorted_idx = np.argsort(tsteps)
                plotstyle =  model_color_dict[model] + split_linestyle_dict[split]
                plotlabel = '%s-%s'%(model, split)
                axs_list[ax_idx].plot(tsteps[sorted_idx], metric_vals[sorted_idx], plotstyle, label=plotlabel)
                axs_list[ax_idx].set_title(plot_metric)
                axs_list[ax_idx].set_xlabel('hours_from_admission')
                axs_list[ax_idx].legend()
                
                # change the xlabel for last timepoint accurately
                axs_list[ax_idx].set_xticks(tsteps)
                axs_list[ax_idx].set_xticklabels([str(tstep) for tstep in tsteps[:-1]]+['full_history'])
    plt.subplots_adjust(hspace=0.3)
    f.savefig(os.path.join(args.clf_performance_dir, 'tstep_performance.png'))


