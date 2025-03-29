import os
import time
import torch
import paramiko
import numpy as np

from utils import console

REMOTE_HOST = "129.173.66.44"
PORT = 22
P_REMOTE = "/home/finlay/disklavier/data/outputs/uploads"
tag = "[#5f00af]panthr[/#5f00af]:"


def calc_embedding(
    file_path: str, user: str = "finlay", model: str = "specdiff"
) -> np.ndarray:
    """
    Calculates the embedding of a MIDI file using the Panther model.

    Parameters
    ----------
    file_path : str
        The path to the MIDI file to calculate the embedding of.
    user : str
        The user to connect to the Panther model as.
    model : str
        The model to use for embedding calculation.

    Returns
    -------
    np.ndarray
        The embedding of the MIDI file.
    """
    # fs setup
    local_folder = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path)

    # add model name to the remote filename
    filename, ext = os.path.splitext(base_filename)
    model_filename = f"{filename}_{model}{ext}"
    remote_file_path = os.path.join(P_REMOTE, model_filename)

    # panther login
    console.log(f"{tag} connecting to panther at {user}@{REMOTE_HOST}:{PORT}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, PORT, user)
    sftp = ssh.open_sftp()
    console.log(f"{tag} connection opened")

    # clear old files
    try:
        sftp.stat(remote_file_path)
        sftp.remove(remote_file_path)
        sftp.remove(os.path.splitext(remote_file_path)[0] + ".pt")
        console.log(f"{tag} existing file '{remote_file_path}' deleted.")
    except FileNotFoundError:
        console.log(
            f"{tag} no existing file found at '{remote_file_path}', proceeding..."
        )
    # upload
    sftp.put(file_path, remote_file_path)
    console.log(
        f"{tag} upload complete, waiting for embedding using model '{model}'..."
    )

    # wait for new tensor
    pf_tensor_remote = os.path.splitext(remote_file_path)[0] + ".pt"
    while 1:
        try:
            sftp.stat(pf_tensor_remote)
            break
        except FileNotFoundError:
            time.sleep(0.1)

    pf_tensor_local = os.path.join(local_folder, os.path.basename(pf_tensor_remote))
    sftp.get(pf_tensor_remote, pf_tensor_local)
    console.log(f"{tag} downloaded embedding file '{pf_tensor_local}'")

    sftp.close()
    ssh.close()

    embedding = torch.load(pf_tensor_local, weights_only=True).numpy().reshape(1, -1)
    console.log(f"{tag} loaded embedding {embedding.shape}")

    return embedding
