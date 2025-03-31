import os
import torch
import subprocess


def get_world_size():
    try:
        return int(os.environ.get("WORLD_SIZE", 1))
    except ValueError:
        return 1


def get_rank():
    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("SLURM_PROCID", 0)
    return int(rank)


def get_local_rank():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID", 0)
    return int(local_rank)

def get_gpus_per_node():
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert torch.cuda.device_count() == gpus_per_node, f"Torch GPU per node count {torch.cuda.device_count()} != {gpus_per_node} on 'SLURM_GPUS_ON_NODE'"
    return gpus_per_node
def setup_master():
    master_addr = os.environ.get("MASTER_ADDR")
    if master_addr is None:
        try:
            master_addr = subprocess.check_output(
                "scontrol show hostnames $SLURM_JOB_NODELIST", shell=True
            ).decode().split()[0]
        except Exception:
            master_addr = "127.0.0.1"
    master_port = os.environ.get("MASTER_PORT", "12345")
    return master_addr, master_port