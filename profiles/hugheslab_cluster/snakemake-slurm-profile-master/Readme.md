# Slurm Snakemake Profile

This is a snakemake profile that you can use to have any snakemake file submit jobs on Sherlock using slurm.

## Usage

First, download this repo and install the profile by running 
```
bash install_snakemake_profile.sh
```
To use the profile, just run
```
snakemake --profile slurm
```
You can add any additional arguments to snakemake just like normal.

## Notes on Snakemake Cluster Mode
- In cluster mode, quitting snakemake will not cancel jobs that have already been submitted. Use `scancel` to clean up any remaining snakemake jobs before restarting snakemake. 
- The snakemake log is decidedly less helpful in saying why a rule crashed in cluster mode. See the Accessing Logs section below for tips.

## Customizing job resource requests

If you have a snakemake rule that needs more time, memory, or cores than the defaults listed in `cluster.json`, you can add parameters to your snakemake rule to specify the following:
- CPUs: Set the `threads` attribute as normal in snakemake
- Memory, time, partition, priority: Set values for `cluster_memory`, `cluster_time`, `cluster_partition`, or `cluster_priority` under the `params` section of the rule. These correspond to the `--mem`, `-t`, `-p`, and `--qos` arguments to `sbatch`.
For example, to make a job with 8 cores, 12G of memory and 4 hours of time do:
```
rule my_job:
    input: ...
    output: ...
    threads: 8
    params:
        cluster_memory = "12G",
        cluster_time = "4:00:00"
    shell: ...
```

## Accessing logs

Log records containing the outputs of all submitted slurm jobs are saved in your snakemake working directory as follows:
- Standard out: `.slurm-logs/{rule_name}/snake.{wildcard_values}.{jobid}.out`
- Standard error: `.slurm-logs/{rule_name}/snake.{wildcard_values}.{jobid}.err`

Usually, I end up wanting to look at logs from a specific jobid that I know crashed. In this case it is tedious to type out the rule name and wildcard values. Instead, for example if job `54085154` crashed I will run:
```
less .slurm-logs/*/*54085154.out
less .slurm-logs/*/*54085154.err
```

### Credits
The submission scripts were adapted from: https://github.com/Snakemake-Profiles/slurm/