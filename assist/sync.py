#!/usr/bin/env python3

import subprocess


def sync_folder(remote_user: str, remote_host: str, remote_folder: str, local_folder: str, exclude_patterns=None):
    """
    Synchronize a remote folder with a local folder using rsync.

    :param remote_user: The username on the remote host (e.g. 'yli3100').
    :param remote_host: The remote host address (e.g. 'a100cse.cc.gatech.edu').
    :param remote_folder: The path of the folder on the remote host (e.g. '/home/yli3100/data').
    :param local_folder: The local folder path to sync to (e.g. '/home/user/data').
    :param exclude_patterns: A list of string patterns to exclude from the sync (e.g. ['*.tmp', 'backup*']).
    """

    if exclude_patterns is None:
        exclude_patterns = []

    # Construct the base rsync command
    # -a (archive) preserves permissions & time
    # -v (verbose) helps see what's happening
    # -z (compress) compresses file data during the transfer
    # --progress shows progress during file transfer
    command = ["rsync", "-avz", "--progress", f"{remote_user}@{remote_host}:{remote_folder}", local_folder]

    # Add exclude options to the command
    for pattern in exclude_patterns:
        command += ["--exclude", pattern]

    # Print the final command (helpful for debugging)
    print(f"Running command: {' '.join(command)}")

    # Execute the rsync command
    try:
        subprocess.run(command, check=True)
        print("Sync completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during rsync: {e}")
        raise


def main():
    # Example usage:
    remote_user = "yli3100"
    remote_host = "a100cse.cc.gatech.edu"
    # remote_folder = "/localscratch/yli3100/LLM-Uncertainty/output-sc/"
    # local_folder = "output-sc/"
    # remote_folder = "/localscratch/yli3100/LLM-Uncertainty/datasets/"
    remote_folder = "/net/csefiles/chaozhanglab/yli3100/LLM-Uncertainty/output/bbh/DeepSeek-R1-Distill-Llama-8B/"
    local_folder = "output/bbh/DeepSeek-R1-Distill-Llama-8B/"

    # Patterns to exclude:
    # e.g., exclude folders named "temp" or anything with ".backup" in the name
    exclude_patterns = ["token-preds*/"]

    sync_folder(remote_user, remote_host, remote_folder, local_folder, exclude_patterns)


if __name__ == "__main__":
    main()
