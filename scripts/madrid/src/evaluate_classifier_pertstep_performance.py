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
                    perf_df.loc[:,'tstep']=re.findall('\-?\d+',os.path.basename(tstep_folder))[0]
                    perf_df.loc[:,'random_seed']=re.findall('\-?\d+',os.path.basename(perf_csv))[0]
                    perf_df.drop(columns='confusion_html',inplace=True)
                    final_perf_df_cols = perf_df.columns
                    final_perf_df_list.append(perf_df.to_numpy())

    final_perf_df = pd.DataFrame(np.vstack(final_perf_df_list), columns = final_perf_df_cols)
    final_perf_df['tstep'] = final_perf_df['tstep'].astype(float) 
    
    # change tstep of full history from -1 to some temp value
    full_history_temp_tstep = np.asarray(final_perf_df['tstep'].max())+5
    final_perf_df.loc[final_perf_df['tstep']==-1, 'tstep']=full_history_temp_tstep
    
    # plot selected metrics
    plot_metrics = [ 'balanced_accuracy', 'f1_score', 'average_precision', 'AUROC']
    for f_idx, plot_metric in enumerate(plot_metrics):
        f, axs = plt.subplots()
    #axs_list = [ax for sublist in axs for ax in sublist]
    #for ax_idx, plot_metric in enumerate(plot_metrics):
        sns.pointplot(x='tstep', y=plot_metric, hue = 'model', data=final_perf_df[final_perf_df.split_name=='test'],ax=axs)
        ticklabels = [item.get_text() for item in axs.get_xticklabels()]
        ticklabels_new = [ticklabel.replace(str(full_history_temp_tstep), 'full_history') for ticklabel in ticklabels]
        axs.set_xticklabels(ticklabels_new)
        axs.set_xlabel('Hours from admission')
        plt.suptitle('Transfer to ICU Prediction From Collapsed Features')
        f.savefig(os.path.join(args.clf_performance_dir, 'tstep_performance_{plot_metric}.pdf'.format(plot_metric=plot_metric)))
