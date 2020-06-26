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
    parser.add_argument('--missingness_dir', default=None, type=str,
        help='Directory where missingness csvs are saved')

    args = parser.parse_args()

    # get performance vals for classifiers at tstep=2,4,6,10,14,-1 etc. -1 is full history     
    final_perf_df_list=list()
    tstep_folders = glob.glob(os.path.join(args.missingness_dir, '*'))
    for tstep_folder in tstep_folders:
        #perf_csv = os.path.join(tstep_folder, 'performance_df.csv')
        perf_csvs = glob.glob(os.path.join(tstep_folder, 'is_available*.csv'))
        for perf_csv in perf_csvs:
            if os.path.exists(perf_csv):
                perf_df = pd.read_csv(perf_csv)
                perf_df.loc[:,'tstep']=re.findall('\-?\d+',os.path.basename(tstep_folder))[0]
                final_perf_df_cols = perf_df.columns
                final_perf_df_list.append(perf_df.to_numpy())

    final_perf_df = pd.DataFrame(np.vstack(final_perf_df_list), columns = final_perf_df_cols)
    final_perf_df['tstep'] = final_perf_df['tstep'].astype(float) 
    
    from IPython import embed; embed()
    # change tstep of full history from -1 to some temp value
    full_history_temp_tstep = np.asarray(final_perf_df['tstep'].max())+5
    final_perf_df.loc[final_perf_df['tstep']==-1, 'tstep']=full_history_temp_tstep
    final_perf_df.sort_values(by='tstep', inplace=True)
    f, axs = plt.subplots(2,2,sharex=True, figsize=(18,12))
    axs_list = [ax for sublist in axs for ax in sublist]
    feature_cols = list(final_perf_df.columns)[:-1]
    for axs_idx, feature in enumerate(feature_cols):
        ax = axs_list[axs_idx]
        final_perf_df[['tstep', feature]].plot(x='tstep', y=feature, ax=ax, kind='bar')
        ticklabels = [item.get_text() for item in ax.get_xticklabels()]
        ticklabels_new = [ticklabel.replace(str(full_history_temp_tstep), 'full_history') for ticklabel in ticklabels]
        ax.set_xticklabels(ticklabels_new)
        ax.set_xlabel('Hours from admission')
        ax.set_title(feature)

    plt.suptitle('Availability Per Feature Across Stays')
    plt.subplots_adjust(hspace=0.3)
    print('saving results to %s'%(os.path.join(args.missingness_dir, 'missingness_pertstep.png')))
    f.savefig(os.path.join(args.missingness_dir, 'missingness_pertstep.png'))
