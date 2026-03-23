"""
Cluster utilities for PDE Latent Representation demonstrator.

Provides SSH and SLURM utilities for remote execution on NAIC VMs and HPC clusters.
PDE solving is more memory-intensive than typical ML workloads, so defaults are
set accordingly (32G RAM, 4-hour time limit).
"""

import os
import random

# Configuration - users must set these
HOSTNAME = os.getenv("CLUSTER_HOSTNAME", "login.saga.sigma2.no")
USERNAME = os.getenv("CLUSTER_USERNAME", "")
PRIVATE_KEY_PATH = os.getenv("SSH_KEY_PATH", os.path.expanduser("~/.ssh/id_rsa"))


def connect_ssh():
    """Connect to cluster via SSH. Requires paramiko."""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if not USERNAME:
        raise ValueError("Set CLUSTER_USERNAME environment variable or configure utils.py")

    private_key = paramiko.RSAKey.from_private_key_file(PRIVATE_KEY_PATH)
    client.connect(hostname=HOSTNAME, username=USERNAME, pkey=private_key)
    return client


def get_available_nodes(client, ignore_list=None):
    """Fetch available idle nodes from SLURM cluster."""
    if ignore_list is None:
        ignore_list = []

    command = 'sinfo --noheader --Node -o "%N %C" -t idle,mix'
    stdin, stdout, stderr = client.exec_command(command)
    lines = stdout.read().decode().splitlines()

    nodes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            if name not in ignore_list:
                nodes.append(name)

    random.shuffle(nodes)
    return nodes[:4]


def submit_slurm_job(client, node, account, script_args, time_limit="4:00:00",
                      cpus=4, mem="32G", gpu=False):
    """Submit a SLURM batch job for PDE solution generation.

    Args:
        client: Paramiko SSH client connected to the cluster.
        node: Target node name for job placement.
        account: SLURM account string for billing.
        script_args: Command-line arguments passed to cd2d_streamfunc.py.
        time_limit: Maximum wall-clock time (default: 4 hours for PDE solving).
        cpus: Number of CPU cores per task (default: 4).
        mem: Memory allocation (default: 32G for multi-resolution PDE solving).
        gpu: Whether to request a GPU (default: False, PDE solving is CPU-bound).

    Returns:
        String output from sbatch submission.
    """
    gpu_flag = "--gres=gpu:1" if gpu else ""
    cmd = (
        f'sbatch --job-name=pde-latent --account={account} '
        f'--time={time_limit} --nodes=1 --ntasks=1 '
        f'--cpus-per-task={cpus} --mem={mem} {gpu_flag} '
        f'--nodelist={node} '
        f'--wrap="source venv/bin/activate && '
        f'python src/cd2d_streamfunc.py {script_args}"'
    )

    stdin, stdout, stderr = client.exec_command(cmd)
    output = stdout.read().decode()
    error = stderr.read().decode()
    print(f"Job submission: {output.strip()}")
    if error:
        print(f"Errors: {error.strip()}")
    return output
