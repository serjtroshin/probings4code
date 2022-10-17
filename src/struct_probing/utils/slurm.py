import logging
import socket
import subprocess
import sys
from pathlib import Path

logging.basicConfig()

def get_slurm_str(save_dir, cpus=2, gpus=0, nodes=1, time="5-00:00:00", constraint="", name="", task="", other=""):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    name = name.replace("+", "").replace(" ", "_")
    task = task.replace(" ", "_")
    o_dir = f"slurm-{name}-{task}.%j.out"
    e_dir = f"slurm-{name}-{task}.%j.err"
    # Path(o_dir).mkdir(exist_ok=True, parents=True)
    logging.info(f"output will be saved at: {save_dir}/slurm-*")

    slurm_str = f"sbatch -c {cpus} --gpus={gpus} --nodes={nodes} -t {time} --constraint={constraint} -o {o_dir} -e {e_dir} {other}"
    return slurm_str

def submit_job(slurm_str: str):
    # return job id
    output = subprocess.run(slurm_str, shell=True, check=True, universal_newlines=True, capture_output=True)
    return output.stdout.split()[-1]

def submit_local(slurm_str: str):
    subprocess.run(slurm_str, shell=True, check=True, stdout=sys.stdout, universal_newlines=True)
    return None

def get_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    port = addr[1]
    s.close()
    return int(port)
