#!/usr/bin/env python3

# Handles slurm submission of cluster jobs. Will follow the cluster.json settings by default,
# but individual jobs can specify cluster_partition, cluster_time, cluster_memory, or cluster_priority
# under their params directive in order to overwrite the sbatch parameters. 

# Notes:
#    - All slurm command logs are written to the folder .slurm-logs/{job_name} under the working directory
#    - The jobscript will echo the sbatch command to the snakemake log

import os
import shutil
import subprocess
import sys

from snakemake.utils import read_job_properties


workingdir = os.getcwd()

jobscript = sys.argv[1]
job_properties = read_job_properties(jobscript)


submission_params = {
    "workingdir": workingdir,
    "jobscript": jobscript,
    "cores": job_properties["threads"]
}

submission_param_names = ["partition", "time", "memory", "priority"]
for p in submission_param_names:
    if "params" in job_properties and "cluster_" + p in job_properties["params"]:
        submission_params[p] = job_properties["params"]["cluster_" + p]
    else:
        submission_params[p] = job_properties["cluster"][p]

def file_escape(string):
    return string.replace("/", "_").replace(" ", "_")

if job_properties["type"] == "single":
    submission_params["job_name"] = "snake." + job_properties["rule"] 
    if len(job_properties["wildcards"]) > 0:
        submission_params["job_name"] += "." + ".".join([key + "=" + file_escape(value) for key,value in job_properties["wildcards"].items()])
    submission_params["log_dir"] = os.path.join(workingdir, ".slurm-logs", job_properties["rule"])
elif job_properties["type"] == "group":
    submission_params["job_name"] = "snake." + job_properties["groupid"]
    submission_params["log_dir"] = os.path.join(workingdir, ".slurm-logs", job_properties["groupid"])
else:
    print("Error: slurm-submit.py doesn't support job type {} yet!".format(job_properties["type"]))
    sys.exit(1)

# Make the slurm-logs directory in case it doesn't exist already
# (required for sbatch job submissions to run properly with logging)
os.makedirs(submission_params["log_dir"], exist_ok=True)

submit_string = "sbatch --job-name={job_name} -p {partition} -c {cores}  -t {time} --mem {memory} --qos {priority} --parsable -o {log_dir}/{job_name}.%j.out -e {log_dir}/{job_name}.%j.err {jobscript}".format(**submission_params)

result = subprocess.run(submit_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

sys.stdout.write(result.stdout.decode())

if len(result.stderr) > 0:
    sys.stderr.write(result.stderr.decode())




