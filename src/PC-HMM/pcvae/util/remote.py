import os, uuid, subprocess
import numpy as np
import pickle


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Export a variable for a run
def export(name, value):
    if value is None:
        return ""
    try:
        if round(value) == value:
            value = int(value)
    except:
        pass
    return "export " + str(name) + "=" + str(value) + "\n"


def execute(cmd, silent=True, raisefail=True, retries=3):
    if not silent:
        print(cmd)

    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

    output = ""
    for stdout_line in iter(popen.stdout.readline, ""):
        if not silent:
            print(stdout_line, end=" ")
        output += stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code and raisefail:
        if retries <= 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        if not silent:
            print("Command failed: " + cmd)
            print("Retries left: %d" % (retries - 1))
        return execute(cmd, silent, raisefail, retries - 1)
    return output

class remote_host(object):
    def __init__(self, config={}, local=False, filename=None, retries=3, **kwargs):
        if type(config) is str:
            config, filename = {}, config

        defaults = dict(
            env=None,
            XHOST="local",
            PYTHONPATH=None,
            pythonexe=None,
            user="",
            host="",
            XHOST_MACHINE_NAME=None,
        )
        known_configs = [
                            "XHOST",
                            "XHOST_RESULTS_DIR",
                            "XHOST_LOG_DIR",
                            "XHOST_MACHINE_NAME",
                        ] + [d for d in defaults.keys()]

        defaults.update(config)
        defaults.update(kwargs)
        config = defaults
        if filename is not None:
            config.update(self.parse_config(filename))

        self.user = config["user"]
        self.host = config["host"]
        self.addr = self.user + "@" + self.host
        self.datapath = config["datapath"]
        self.env = config["env"]
        self.pythonexe = config["TFPYTHONEXE"]
        self.xhost = config["XHOST"]
        self.xhost_machine = config["XHOST_MACHINE_NAME"]
        self.root = config["PCPYROOT"] if "PCPYROOT" in config else config["PCVAEROOT"]
        self.results_dir = config["XHOST_RESULTS_DIR"]
        self.log_dir = config["XHOST_LOG_DIR"]
        self.pythonpath = config["PYTHONPATH"]
        self.local = local
        self.scratchdir = (
            config["XHOST_SCRATCH_DIR"] if "XHOST_SCRATCH_DIR" in config else "~"
        )

        self.other_config = {
            c: v
            for c, v in config.items()
            if c not in known_configs
        }
        self.retries = retries
        makedirs("/tmp/scratch/")

    def parse_config(self, filename):
        config = {}
        with open(filename, "r") as reader:
            for line in reader:
                if len(line.strip()) > 0 and line.strip()[0] != "#":
                    var = line.split()
                    config[var[0]] = var[1]
        return config

    def run_remote(self, script, silent=False, background=False):
        makedirs("/tmp/scratch/")
        with open("/tmp/scratch/rscript.sh", "w") as text_file:
            text_file.write(script)

        if self.local:
            cmd = "bash /tmp/scratch/rscript.sh"
            if background:
                cmd += " &"
        else:
            cmd = "ssh " + self.addr
            cmd += " 'bash -s' < /tmp/scratch/rscript.sh"
        return execute(cmd, silent=silent, retries=self.retries)

    def sync_2_remote(self, file, remote_path, R=False, silent=True):
        cmd = "rsync --copy-links " + ("-R " if R else "")
        cmd += file + " "
        cmd += self.addr + ":" if not self.local else ""
        cmd += remote_path
        print(cmd)
        return execute(cmd, silent=silent, retries=self.retries)

    def sync_2_local(self, file, remote_path, R=True, silent=True):
        cmd = "rsync --copy-links " + ("-R " if R else "")
        cmd += self.addr + ":" if not self.local else ""
        cmd += remote_path + " " + file
        return execute(cmd, silent=silent, retries=self.retries)

    def sync_obj_2_remote(self, obj, remote_path, name='pick', R=False, silent=False):
        with open('/tmp/scratch/%s.pkl' % name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return self.sync_2_remote('/tmp/scratch/%s.pkl' % name, remote_path, R, silent)

    def sync_obj_2_local(self, remote_path, name='pick', R=False, silent=True):
        self.sync_2_local('/tmp/scratch/%s.pkl' % name, os.path.join(remote_path, '%s.pkl' % name), R, silent)
        with open('/tmp/scratch/%s.pkl' % name, 'rb') as f:
            return pickle.load(f)

    def get_env_script(self, remote_path, pull=True, force_local=False):
        txt = "cd " + self.root + "\n"
        if pull and not self.local:
            txt += "git pull\n"

        txt += "unset PYTHONPATH\nunset TFPYTHONEXE\n"

        txt += export("XHOST", self.xhost)
        if self.xhost_machine:
            machine = "local" if force_local else self.xhost_machine
            txt += export("XHOST_MACHINE_NAME", machine)
        txt += export("XHOST_RESULTS_DIR", remote_path)
        txt += export("XHOST_LOG_DIR", self.log_dir)

        for c, v in self.other_config.items():
            txt += export(c, v)

        if self.env:
            txt += "conda activate " + self.env + "\n"
        txt += export("TFPYTHONEXE", self.pythonexe)
        txt += export("PCPYROOT", self.root)
        txt += export("PYTHONPATH", "$PYTHONPATH:$PCPYROOT")
        if self.pythonpath:
            txt += export("PYTHONPATH", "$PYTHONPATH:" + self.pythonpath)

        return txt

    def create_remote_script(self, remote_path, pull=True):
        txt = self.get_env_script(remote_path, pull=pull)

        train_script = "$PCPYROOT/grid_tools/train.sh"
        txt += export("XHOST_BASH_EXE", train_script)
        txt += export("dataset_path", self.datapath)

        output_path = remote_path + "/"
        txt += export("output_path", output_path)
        txt += (
            "bash $PCPYROOT/grid_tools/launch_job_on_host_via_env.sh || { exit 0; }\n"
        )
        txt += "#" + (78 * "-") + "#\n"
        return txt
