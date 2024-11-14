import os
import time
import torch
import paramiko
import numpy as np

from utils import console

REMOTE_HOST = "129.173.66.44"
P_REMOTE = "/home/finlay/disklavier/data/outputs"
tag = "[#5f00af]panthr[/#5f00af]:"


def calc_embedding(file_path: str, user: str = "finlay") -> np.ndarray:
    # fs setup
    local_folder = os.path.dirname(file_path)
    remote_file_path = os.path.join(P_REMOTE, os.path.basename(file_path))
    # panther login
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, 22, user)
    sftp = ssh.open_sftp()

    # clear old files
    try:
        sftp.stat(remote_file_path)
        sftp.remove(remote_file_path)
        sftp.remove(remote_file_path[:-4] + ".pt")
        console.log(f"{tag} Existing file '{remote_file_path}' deleted.")
    except FileNotFoundError:
        console.log(
            f"{tag} No existing file found at '{remote_file_path}'. Proceeding with upload."
        )
    # upload
    sftp.put(file_path, remote_file_path)

    # wait for new tensor
    pf_tensor_remote = remote_file_path[:-4] + ".pt"
    while 1:
        try:
            sftp.stat(pf_tensor_remote)
            break
        except FileNotFoundError:
            time.sleep(1)

    pf_tensor_local = os.path.join(local_folder, os.path.basename(pf_tensor_remote))
    sftp.get(pf_tensor_remote, pf_tensor_local)

    sftp.close()
    ssh.close()

    embedding = torch.load(pf_tensor_local, weights_only=True).numpy()

    return embedding / np.linalg.norm(embedding)
