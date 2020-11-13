#!/bin/bash

if [[ $XHOST == 'grid' ]]; then
    # Avoid race conditions on NFS file access
    # by sleeping a little while (60 sec or less)
    sleep $[ ( $RANDOM % 10 )  + 1 ]s
fi

if [[ -z $mod_name ]]; then
    mod_name=vae__np;
fi

echo "---------- START setup_train.sh"

if [[ -z $XHOST_NUM_THREADS  ]]; then
    export OMP_NUM_THREADS=4
	export MKL_NUM_THREADS=4
else
	export OMP_NUM_THREADS=$XHOST_NUM_THREADS
	export MKL_NUM_THREADS=$XHOST_NUM_THREADS
fi


echo "SHMMROOT=$SHMMROOT"
echo "PCPYROOT=$PCPYROOT"
echo "PYTHONPATH=$PYTHONPATH"
echo "MATLABPATH=$MATLABPATH"

end_of_mod_name=`python -c "print '$mod_name'[-2:]"`

if [[ -z $TFPYTHONEXE ]]; then
    echo "case 1"
    export TFPYTHONEXE=`which python`
fi
echo "python executable:"
echo $TFPYTHONEXE
echo "output_path:"
echo "$output_path"
echo "---------- STOP  setup_train.sh"

# Run the jobinfo stdout code (moved here from rungrid to be model agnostic)
eval $TFPYTHONEXE $PCPYROOT/grid_tools/grid.py
