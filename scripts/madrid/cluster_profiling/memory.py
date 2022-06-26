'''
Usage : python -u memory.py --query_pids 6699,6702,7119
'''

import argparse
import pandas as pd
import os
import subprocess
import psutil

global GPU_UUID_MAP
GPU_UUID_MAP = dict()

def get_current_cpu_mem_usage(field='rss', process=None):
    ''' Return the memory usage in MB of provided process
    '''
    if process is None:
        process = psutil.Process(os.getpid())
    mem = getattr(process.memory_info(), field)
    mem_MiB = mem / float(2 ** 20)
    return mem_MiB

def sanitize_single_line_output_from_list_gpu(line):
    """ Helper to parse output of `nvidia-smi --list-gpus`
    
    Args
    ----
    line : string
        One line from `nvidia-smi --list-gpus`
        
    Returns    
    -------
    ldict : dict
        One field for each of name, num, and uuid
    
    Examples
    --------
    >>> s = "GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-4b3bcbe7-8762-7baf-cd29-c1c51268360d)"
    >>> ldict = sanitize_single_line_output_from_list_gpu(s)
    >>> ldict['num']
    '0'
    >>> ldict['name']
    'Tesla-P100-PCIE-16GB'
    >>> ldict['uuid']
    'GPU-4b3bcbe7-8762-7baf-cd29-c1c51268360d'
    """
    num, name, uuid = map(str.strip, line.split(":"))
    num = num.replace("GPU ", "")
    name = name.replace(" (UUID", "").replace(" ", "-")
    uuid = uuid.replace(")", "")
    return dict(num=num, name=name, uuid=uuid)

def lookup_gpu_num_by_uuid(uuid):
    ''' Helper method to lookup a gpu's integer id by its UUID string
    
    Args
    ----
    uuid : string
    
    Returns
    -------
    int_id : int
        Integer ID (matching index of PCI_BUS_ID sorting used by nvidia-smi)
    '''
    global GPU_UUID_MAP
    try:
        return GPU_UUID_MAP[uuid]
    except KeyError:
        result = subprocess.check_output(['nvidia-smi', '--list-gpus'])
        # Expected format of 'result' is a multi-line string:
        #GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-4b3bcbe7-8762-7baf-cd29-c1c51268360d)
        #GPU 1: Tesla P100-PCIE-16GB (UUID: GPU-f9acf3b8-b5fa-31c5-ecce-81add0ee6a3e)
        #GPU 2: Tesla V100-PCIE-32GB (UUID: GPU-89d98666-7ceb-ccde-136f-28e562834116)

        # Convert into a dictionary mapping a UUID to a plain GPU integer id
        row_list = [
            sanitize_single_line_output_from_list_gpu(line)
            for line in result.decode('utf-8').strip().split('\n')]
        GPU_UUID_MAP = dict(zip([d['uuid'] for d in row_list], [d['num'] for d in row_list]))
        return GPU_UUID_MAP.get(uuid, None)
    
def get_current_cpu_and_gpu_mem_usage_df(query_pids=None):
    """ Get a dataframe of the current cpu and gpu memory usage.

    Will assess usage for either a provided list of processes,
    or the current process and any child process.

    Works only on Unix systems (Linux/MacOS). Not tested on Windows!

    Args
    ----
    query_pids : list or None
        If provided, each entry in list should be an integer process id

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """

    # Determine list of process ids to track called "keep_pids"
    if query_pids is None:
        keep_pids = []
        current_process = psutil.Process(os.getpid())
        keep_pids.append(current_process.pid)
        for child_process in current_process.children(recursive=True):
            keep_pids.append(child_process.pid)
    elif isinstance(query_pids, list):
        keep_pids = list(map(int, query_pids))
    else:
        keep_pids = list(map(int, query_pids.split(',')))
        current_process = psutil.Process(os.getpid()).pid
        keep_pids.append(current_process)

    # Obtain dataframe of memory usage for each process
    cpu_row_list = list()
    for pid in keep_pids:
        proc = psutil.Process(pid)
        mem_val = get_current_cpu_mem_usage(process=proc)
        row_dict = dict()
        row_dict['pid'] = pid
        row_dict['cpu_mem'] = mem_val
        cpu_row_list.append(row_dict)
    cpu_usage_by_pid_df = pd.DataFrame(cpu_row_list)

    # Request dataframe of GPU memory usage from NVIDIA command-line tools
    '''
    result = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-compute-apps=pid,gpu_name,gpu_uuid,process_name,used_memory',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    keys = ['pid', 'gpu_name', 'gpu_uuid', 'process_name', 'used_memory']
    row_list = [
        dict(zip(keys, map(str.strip, x.split(',')))) for x in result.decode('utf-8').strip().split('\n')]

    all_usage_df = pd.DataFrame(row_list)
    all_usage_df['pid'] = all_usage_df['pid'].astype(int)
    all_usage_df['gpu_mem'] = all_usage_df['used_memory'].astype(float)
    gpu_usage_by_pid_df = all_usage_df[all_usage_df['pid'].isin(keep_pids)].copy()
    gpu_usage_by_pid_df['gpu_id'] = [int(lookup_gpu_num_by_uuid(v)) for v in gpu_usage_by_pid_df['gpu_uuid'].values]
    del gpu_usage_by_pid_df['gpu_uuid']
    del gpu_usage_by_pid_df['used_memory']
    # Combine GPU and CPU into one df
    row_dict = dict()
    row_dict['cpu_mem'] = cpu_usage_by_pid_df['cpu_mem'].sum()
    total = 0.0
    for gpu_id in map(int, GPU_UUID_MAP.values()):
        mem_val = gpu_usage_by_pid_df.query("gpu_id == %d" % gpu_id)['gpu_mem'].sum()
        row_dict['gpu_%d_mem' % gpu_id] = mem_val
        total += mem_val
    row_dict['gpu_total_mem'] = total
    agg_df = pd.DataFrame([row_dict])
    return agg_df, cpu_usage_by_pid_df, gpu_usage_by_pid_df
    '''
    return cpu_usage_by_pid_df
    

if __name__ == '__main__':
    # Good habit for any code using CUDA
    # Makes sure CUDA_VISIBLE_DEVICES ids align with nvidia-smi
    # since nvidia-smi sorts by pci_bus location of the GPU
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

    import numpy as np
    import time
    try:
        import tensorflow as tf
        HAS_TF = True
    except Exception:
        HAS_TF = False

    if HAS_TF:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("Configured GPUs with memory growth")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    pd.set_option('display.precision', 3)
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_np_arrays', default=0, type=int)
    parser.add_argument('--include_tf_cpu_arrays', default=0, type=int)
    parser.add_argument('--include_tf_gpu_arrays', default=0, type=int)
    parser.add_argument('--query_pids', default=None, type=str)
    args = parser.parse_args()

    max_step = 6
    all_arrays = list() # make arrays persist so wont be garbage collected

    for step in range(max_step):
        print("--- Begin step %d" % step)

        if step > 0 and step < max_step//2:
            if args.include_np_arrays:
                A = np.random.randn(1000000, 64).astype(np.float32)
                print("Allocated numpy float64 arr of shape (%d,%d) and size %.2f MB" % (A.shape[0], A.shape[1], A.nbytes / (2**20)))
                all_arrays.append(A)

            if HAS_TF:
                if args.include_tf_cpu_arrays:
                    with tf.device("/device:cpu:0"):
                        tfA = tf.multiply(A, 1.0)
                        print("Allocated tf float64 arr of shape (%d,%d) and size %.2f MB on device %s" % (
                            tfA.shape[0], tfA.shape[1], A.nbytes / (2**20), tfA.device))
                        all_arrays.append(tfA)

                if args.include_tf_gpu_arrays:
                    for localid, gpu_num in enumerate(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')):
                        with tf.device('/device:gpu:%s' % localid):
                            tfB = tf.multiply(A, 1.0)
                            tfB += 1 # force gpu to do some work
                            print("Allocated tf float64 arr of shape (%d,%d) and size %.2f MB on device %s" % (
                                tfB.shape[0], tfB.shape[1], A.nbytes / (2**20), tfB.device))
                            all_arrays.append(tfB)

        if step == 1:
            n_per_step = len(all_arrays) # Num arrays allocated each step

        if step > max_step // 2:
            for _ in range(n_per_step):
                arr = all_arrays.pop()
                print("Deleting most recent arr of type %s" % type(arr))
                del arr

        # Wait a bit for garbage collection etc
        time.sleep(0.5)

        print("Reporting current memory")
        cpu_pid_df= get_current_cpu_and_gpu_mem_usage_df(query_pids=args.query_pids)
        print(cpu_pid_df)

        print("--- End step %d" % step)
