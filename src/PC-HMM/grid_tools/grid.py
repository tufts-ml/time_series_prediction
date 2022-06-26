import os
import numexpr as ne  # Has nice cpuinfo utils


def detect_jobid_and_taskid_from_environ():
    """ Determine job/task ids for current grid job (if any).

    Returns
    -------
    is_grid : bool
        True if current os environment indicates we're a grid task
        False otherwise
    jobid : string
        Identifier for current job on grid system
        If not a grid job, will be 'none'.
    taskid : string
        Identifier for current task on grid system
        If not a grid job, will be 'none'.
    """
    is_grid = False
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        jobid_str = os.environ["SLURM_ARRAY_JOB_ID"]
        taskid_str = os.environ["SLURM_ARRAY_TASK_ID"]
        is_grid = True
    elif "LSB_JOBINDEX" in os.environ:
        jobid_str = os.environ["LSB_JOBID"]
        taskid_str = os.environ["LSB_JOBINDEX"]
        is_grid = True

    elif "SGE_TASK_ID" in os.environ:
        jobid_str = os.environ["JOB_ID"]
        taskid_str = os.environ["SGE_TASK_ID"]
        is_grid = True
    elif "XHOST_TASK_ID" in os.environ:
        jobid_str = "none"
        taskid_str = os.environ["XHOST_TASK_ID"]
    else:
        jobid_str = "none"
        taskid_str = "none"
    return is_grid, jobid_str, taskid_str


def create_info_file_and_stdout_and_stderr_files_for_grid_task(
    output_path=None,
    log_dir=None,
    stdout_log_file_template="$jobid.$taskid.out",
    stderr_log_file_template="$jobid.$taskid.err",
    verbose=True,
):
    """ Create symlinks to stdout and stderr inside provided output dir.

    Args
    ----
    output_path : string, valid file path
        Provides full system path to current task's output directory
    log_dir : string, valid file path
        If None, defaults to env var XHOST_LOG_DIR

    Post Condition
    --------------
    The directory specified by 'output_path' contains 4 files:
    * machine_info.txt
    * job_info.csv
    * grid_log.out
        Symlink to grid system's stdout file for current task.
    * grid_log.err
        Symlink to grid system's stderr file for current task.
    """
    is_grid, jobid_str, taskid_str = detect_jobid_and_taskid_from_environ()
    if output_path is None:
        if "output_path" in os.environ:
            output_path = os.path.abspath(os.environ["output_path"])
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(os.path.join(output_path))

    with open(os.path.join(output_path, "jobinfo.csv"), "w") as f:
        f.write(",".join([str(is_grid), str(jobid_str), str(taskid_str)]) + "\n")

    info_list = list()
    info_list.append("JOB_ID  = %s" % jobid_str)
    info_list.append("TASK_ID = %s" % taskid_str)
    uname_list = os.uname()
    info_list.append("hostname = %s" % uname_list[1])
    info_list.append(" ".join(map(str, uname_list)))
    try:
        cpu_list = ne.cpuinfo.cpuinfo.info
        if isinstance(cpu_list, list):
            info_list.append("n_cpus = %d" % len(cpu_list))
            for cpu_info in cpu_list[:4]:
                info_list.append(
                    "%s MHz  %s" % (cpu_info["cpu MHz"], cpu_info["model name"])
                )
    except Exception as e:
        print((str(e)))
        pass
    info_list.append("")
    with open(os.path.join(output_path, "machine_info.txt"), "w") as f:
        print("---------- START machine_info")
        for line in info_list:
            print(line)
            f.write(line + "\n")
        print("---------- STOP  machine_info")

    if not is_grid:
        return is_grid

    if log_dir is None:
        if "XHOST_LOG_DIR" in os.environ:
            log_dir = os.path.abspath(os.environ["XHOST_LOG_DIR"])
    if log_dir is not None:
        if not os.path.exists(log_dir):
            print(("WARNING: provided log dir does not exist\n%s" % log_dir))

    output_path = os.path.abspath(output_path).rstrip(os.path.sep)
    stdout_fpath = (
        os.path.join(log_dir, stdout_log_file_template)
        .replace("$jobid", jobid_str)
        .replace("$taskid", taskid_str)
    )
    if not os.path.exists(stdout_fpath):
        raise ValueError("stdout log not found %s" % stdout_fpath)

    stderr_fpath = (
        os.path.join(log_dir, stderr_log_file_template)
        .replace("$jobid", jobid_str)
        .replace("$taskid", taskid_str)
    )
    if not os.path.exists(stderr_fpath):
        raise ValueError("stderr log not found %s" % stderr_fpath)

    if verbose:
        print("---------- SETUP symlinks at grid_log.out and grid_log.err")
        print(("stdout: %s" % stdout_fpath))
        print(("stderr: %s" % stderr_fpath))
    try:
        os.symlink(stdout_fpath, os.path.join(output_path, "grid_log.out"))
        os.symlink(stderr_fpath, os.path.join(output_path, "grid_log.err"))
    except Exception as e:
        print((str(e)))
        pass
    if verbose:
        print("----------")

    return is_grid


# Lets us run this as a script from setup_train.sh
if __name__ == "__main__":
    try:
        create_info_file_and_stdout_and_stderr_files_for_grid_task()
    except:
        pass
