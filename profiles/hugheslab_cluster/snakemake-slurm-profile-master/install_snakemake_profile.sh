#!/usr/bin/env bash
# Copies the slurm profile to $HOME/.config/snakemake, substituting in the install directory
# of this pipeline in the appropriate places.
# After installation, you can run snakemake --profile slurm in order to run your snakemake file
# with all the necessary commandline options to submit through slurm

PROFILE_DIR=$HOME/.config/snakemake/slurm

mkdir -p $PROFILE_DIR
cp ../cluster.json ../slurm-status.py ../slurm-submit.py $PROFILE_DIR
chmod u+x $PROFILE_DIR/slurm-status.py $PROFILE_DIR/slurm-submit.py
sed "s@PROFILE_DIR@$PROFILE_DIR@g" ../config.yaml > $PROFILE_DIR/config.yaml 
