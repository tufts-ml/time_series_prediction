import distutils.spawn

def detect_template_ext_for_current_system():
    if distutils.spawn.find_executable("sacct"):
        return "slurm"
    elif distutils.spawn.find_executable("bjobs"):
        return "lsf"
    elif distutils.spawn.find_executable("qstat"):
        return "sge"
    raise ValueError("Unknown grid system")

if __name__ == "__main__":
    ext_str = detect_template_ext_for_current_system()

    if ext_str == "sge":
        print("qsub")
    elif ext_str == "lsf":
        print("bsub")
    else:
        print("sbatch")
