'''
plot performance of a all classifiers on multiple patient-stay-slices
'''
import os
import numpy as np
import pandas as pd
import argparse
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--performance_csv_dir', default=None, type=str,
                        help='Directory where classifier models performance csvs are saved')
    parser.add_argument('--output_dir', default='', type=str,
                       help='Directory to save the plots in')
    
    
    args = parser.parse_args()
    
    # load all performance csv into a single dataframe
    perf_csvs = glob.glob(os.path.join(args.performance_csv_dir,'**','*perlambda_performance.csv'), recursive=True) 
    perf_df = pd.DataFrame()
    for perf_csv in perf_csvs:
        curr_perf_df = pd.read_csv(perf_csv)
        perf_df = perf_df.append(curr_perf_df, ignore_index=True)
    # plot performance as a function of hours from admission, % data observed and hours from deterioration 
    lambda_list = perf_df['lambda'].unique()
    lambda_vals = [int(i) for i in lambda_list]
    
    print('Saving plots to : \n%s'%(args.output_dir))

    perf_measures = ['roc_auc', 'average_precision']
    fontsize=18
    suffix = 'first_24_hours'
    
    for perf_measure in perf_measures:
        f, axs = plt.subplots(figsize=[10,8])
        model_colors=['r', 'b', 'g', 'k', 'm']
        for p, model in enumerate(perf_df.model.unique()):
            inds = (perf_df.model==model) 
            cur_df = perf_df.loc[inds, :].copy()
            y = cur_df.loc[cur_df.percentile==50, perf_measure].values
            y_err  = np.zeros((2, len(lambda_vals)))
            y_err[0,:] = y - cur_df.loc[cur_df.percentile==5, perf_measure].values
            y_err[1,:] = cur_df.loc[cur_df.percentile==95, perf_measure].values - y

            axs.errorbar(x=lambda_vals, y=y, yerr=y_err, label=model, fmt='.-', linewidth=3, color=model_colors[p])
            axs.plot(lambda_vals, y, '.', markersize=20, color=model_colors[p])

        axs.set_xlabel('Lambda', fontsize=fontsize)
        fig_aka='perf_%s_per_lambda_%s.pdf'%(perf_measure, suffix)
        axs.set_xticks(lambda_vals)
        axs.set_xticklabels([str(i) for i in lambda_vals])

        if perf_measure == 'roc_auc':
            axs.set_ylim([0.48, 1])
        elif perf_measure == 'average_precision':
            axs.set_ylim([0, 1])
        
        plt.xscale('log')
        axs.set_ylabel(perf_measure, fontsize=fontsize)
        axs.legend(fontsize=fontsize-2, loc='upper left')
        axs.tick_params(labelsize=fontsize)
        fig_name = os.path.join(args.output_dir, fig_aka)
        f.savefig(fig_name)
        print('Saved results to %s'%fig_name)