import os
import pandas as pd

class remote_run(object):
    def __init__(self, remote_path, host):
        self.remote_path = remote_path
        self.host = host

    def __call__(self):
        try:
            return self.host.sync_obj_2_local(self.remote_path, name='results')
        except:
            self.host.sync_2_local('/tmp/scratch/results.csv', os.path.join(self.remote_path, 'results.csv'), False)
            return dict(results=dict(table=pd.read_csv('/tmp/scratch/results.csv')))

def run_remote(host, name, dataset, model, download_dir=None, pull=True, n_trials=1, remote_path=None,
               **kwargs):
    if remote_path is None:
        remote_path = os.path.join(host.results_dir, name)
    args = dict(dataset=dataset, model=model, n_trials=n_trials, download_dir=host.datapath, **kwargs)
    host.run_remote("mkdir -p " + remote_path + "\n", silent=False)
    host.sync_obj_2_remote(args, remote_path, name='config', silent=False)
    script = host.create_remote_script(remote_path, pull=pull)
    host.run_remote(script, silent=False)
    return remote_run(remote_path, host)
