#!/bin/bash

if [[ $XHOST == 'list' || $XHOST == 'dry' ]]; then
    if [[ -z $target_names ]]; then
        echo $output_path
    else
        echo $target_names $output_path
    fi
elif [[ $XHOST == 'grid' ]]; then
    launcher_exe=`python $PCPYROOT/grid_tools/detect_grid_executable.py`
    tmp_script_path=`python $PCPYROOT/grid_tools/make_launcher_script.py`
    CMD="$launcher_exe < $tmp_script_path"
    #echo $CMD
    eval $CMD
elif [[ $XHOST == 'local' ]]; then
    echo $output_path
    bash $XHOST_BASH_EXE
    exit 0
elif [[ $XHOST == 'local_alltasks' ]]; then
    echo $output_path
    for XHOST_TASK_ID in `seq $XHOST_FIRSTTASK $XHOST_NTASKS`
    do
        echo ">>> task $XHOST_TASK_ID"
        export XHOST_TASK_ID=$XHOST_TASK_ID
        bash $XHOST_BASH_EXE
    done
    unset XHOST_TASK_ID
else
    if [[ -z $XHOST ]]; then 
        echo "USER DID NOT DEFINE XHOST"
    else
        echo "UNRECOGNIZED VALUE FOR XHOST: $XHOST"
    fi
    echo "SUPPORTED OPTIONS:"
    echo "XHOST=list  : list output_path for all tasks, then exit"
    echo "XHOST=local : run first task on current local machine"
    echo "XHOST=grid  : run all tasks on available grid engine (SLURM/SGE/LSF)"
    exit 1
fi
