'''
plot performance of a all classifiers on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
import argparse
import glob
import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--performance_csv_dir', default=None, type=str,
                        help='Directory where classifier models performance csvs are saved')
    parser.add_argument('--output_dir', default='', type=str,
                       help='Directory to save the plots in')
    
    
    args = parser.parse_args()
    
    # load all performance csv into a single dataframe
    add_dir = "*/v05112022/classifier_per_tslice_performance/"
    
    pchmm_perf_csv = glob.glob(os.path.join(args.performance_csv_dir+add_dir,'semi_supervised_pchmm_performance.csv'),
                               recursive=True)[0] 
    mixmatch_perf_csv = glob.glob(os.path.join(args.performance_csv_dir+add_dir,'MixMatch_performance.csv'), 
                                  recursive=True)[0] 
    grud_perf_csv = glob.glob(os.path.join(args.performance_csv_dir+add_dir,'GRUD_performance.csv'), 
                                  recursive=True)[0] 
    
    
    
    perf_csvs = [pchmm_perf_csv, 
                 mixmatch_perf_csv,
                 grud_perf_csv
                ]
    perf_df = pd.DataFrame()
    for perf_csv in perf_csvs:
        curr_perf_df = pd.read_csv(perf_csv)
        perf_df = perf_df.append(curr_perf_df, ignore_index=True)

    # plot performance as a function of hours from admission, % data observed and hours from deterioration 
    perc_labelled_list = perf_df['perc_labelled'].unique()
    
    exclude_perc_labels = [30]
    perc_labelled_vals = [int(i) for i in perc_labelled_list if i not in exclude_perc_labels]
    
    
    for i in exclude_perc_labels:
        drop_inds = perf_df['perc_labelled']==i
        perf_df = perf_df.loc[~drop_inds].reset_index(drop=True)
    print('Saving plots to : \n%s'%(args.output_dir))

    perf_measures = ['roc_auc', 'average_precision']
    fontsize=18
    suffix = 'first_24_hours'
    
    yticks = np.arange(.1, 1, .05)
    for perf_measure in perf_measures:
        f, axs = plt.subplots(figsize=[10,8])
        model_colors=['r', 'b', 'g', 'k', 'm', 'y', 'c', 'mediumseagreen']
        sns.set_style("whitegrid") # or use "white" if we don't want grid lines
        sns.set_context("notebook", font_scale=1.5)
        for p, model in enumerate(perf_df.model.unique()):
            inds = (perf_df.model==model) 
            cur_df = perf_df.loc[inds, :].copy()
            y = cur_df.loc[cur_df.percentile==50, perf_measure].values
            y_err  = np.zeros((2, len(perc_labelled_vals)))
            try:
                y_err[0,:] = y - cur_df.loc[cur_df.percentile==5, perf_measure].values
            except:
                y_err[1,:] = cur_df.loc[cur_df.percentile==95, perf_measure].values - y
            
            axs.errorbar(x=perc_labelled_vals, y=y, yerr=y_err, label=model, fmt='.-', linewidth=3, color=model_colors[p])
            axs.plot(perc_labelled_vals, y, '.', markersize=20, color=model_colors[p])

        axs.set_xlabel(r'$\%$ labels (trained on)', fontsize=fontsize)
        fig_aka='perf_%s_semi_supervised_%s.pdf'%(perf_measure, suffix)
        axs.set_xticks(perc_labelled_vals)
        axs.set_xticklabels([str(i) for i in perc_labelled_vals])
        axs.set_yticks(yticks)
        axs.set_yticklabels(['%.2f'%i for i in yticks])
        
        if perf_measure == 'roc_auc':
            axs.set_ylim([0.1, 1])
        elif perf_measure == 'average_precision':
            axs.set_ylim([0.1, 1])
        
        axs.set_ylabel('test set %s'%perf_measure, fontsize=fontsize)
        axs.legend(fontsize=fontsize-2, loc='lower right')
        axs.tick_params(labelsize=fontsize)
        fig_name = os.path.join(args.output_dir, fig_aka)
        plt.suptitle('Predicting in-hospital mortality in first 24 hours (MIMIC)', fontsize=fontsize)
#         plt.grid(True)
        f.savefig(fig_name,bbox_inches='tight', pad_inches=0)
#         f.savefig(fig_name)
        print('Saved results to %s'%fig_name)