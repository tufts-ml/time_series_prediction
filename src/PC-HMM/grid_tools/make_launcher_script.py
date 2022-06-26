#!/usr/bin/env python

import os
import distutils.spawn
import tempfile

DEFAULT_KEYS = [
    "XHOST_MACHINE_NAME",
    "XHOST_LOG_DIR",
    "XHOST_FIRSTTASK",
    "XHOST_NTASKS",
    "XHOST_MEM_MB",
    "XHOST_SWP_MB",
    "XHOST_TIME_HR",
    "XHOST_GPUS",
    "XHOST_NUM_THREADS",
]


def set_default_environment():
    if "XHOST_LOG_DIR" not in os.environ:
        raise ValueError("NEED TO DEFINE XHOST_LOG_DIR")
    if "XHOST_NTASKS" not in os.environ:
        os.environ["XHOST_NTASKS"] = "1"
    if "XHOST_FIRSTTASK" not in os.environ:
        os.environ["XHOST_FIRSTTASK"] = "1"
    if "XHOST_MEM_MB" not in os.environ:
        os.environ["XHOST_MEM_MB"] = "2500"
    if "XHOST_SWP_MB" not in os.environ:
        os.environ["XHOST_SWP_MB"] = "25000"
    if "XHOST_MACHINE_NAME" not in os.environ:
        os.environ["XHOST_MACHINE_NAME"] = "*"
    if "XHOST_TIME_HR" not in os.environ:
        os.environ["XHOST_TIME_HR"] = "240"
    if "XHOST_GPUS" not in os.environ:
        os.environ["XHOST_GPUS"] = "0"
    if "XHOST_NUM_THREADS" not in os.environ:
        os.environ["XHOST_NUM_THREADS"] = "1"


def detect_template_ext_for_current_system():
    if distutils.spawn.find_executable("sacct"):
        return "slurm"
    elif distutils.spawn.find_executable("bjobs"):
        return "lsf"
    elif distutils.spawn.find_executable("qstat"):
        return "sge"
    raise ValueError("Unknown grid system")


def make_launcher_script_file():
    ext_str = detect_template_ext_for_current_system()
    with open(
        os.path.expandvars("$PCPYROOT/grid_tools/templates") + "/template." + ext_str,
        "r",
    ) as f:
        lines = f.readlines()

    launcher_f = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="launcher_for_%s_" % os.environ["USER"],
        suffix="." + ext_str,
        delete=False,
    )
    for line in lines:
        gpu = "XHOST_GPUS" in line
        threads = "XHOST_NUM_THREADS" in line
        for key in DEFAULT_KEYS:
            line = line.replace("$" + key, os.environ[key])
        line = line.replace(
            "$XHOST_BASH_EXE", os.path.abspath(os.environ["XHOST_BASH_EXE"])
        )
        if (not gpu or os.environ["XHOST_GPUS"] != "0") and (
            not threads or os.environ["XHOST_GPUS"] == "0"
        ):
            launcher_f.write(line)
    launcher_f.close()

    return os.path.abspath(launcher_f.name)


if __name__ == "__main__":
    set_default_environment()
    print(make_launcher_script_file())
