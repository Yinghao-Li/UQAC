import numpy as np
import glob
import json
import os
import os.path as osp
import logging
import h5py
import concurrent.futures
import logging
import os.path as osp
import pandas as pd
from natsort import natsorted
from transformers import AutoTokenizer
from seqlbtoolkit.io import ProgressBar


logger = logging.getLogger(__name__)


def load_attrs(
    file_dir: str,
    file_name: str = "predictions.h5",
    instance_ids: list[str] = None,
    keys: list[str] = None,
    num_workers: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load (or lazy-load) HDF5 prediction files from the specified directory.

    Parameters
    ----------
    file_dir : str
        Path to the directory containing prediction files.
    instance_ids : list of str, optional
        List of instance IDs to load. If None, all subdirectories in `file_dir` are used.
    keys : list of str, optional
        If provided, only load these specific keys/datasets from each .h5 file.
    parallel : bool, default=True
        If True, load predictions in parallel using multiple processes (for speed).
    num_workers : int, default=8
        Number of worker processes when `parallel` is True.
    """
    parallel = True if num_workers != 1 else False

    # If no instance_ids provided, find all subdirectories
    if instance_ids is None:
        instance_ids = [osp.basename(osp.normpath(d)) for d in glob.glob(osp.join(file_dir, "*")) if osp.isdir(d)]

    # Step 1: Collect references (lazy) or store file paths
    predictions: dict[str, h5py.File | str] = {}
    for instance_id in instance_ids:
        instance_dir = osp.join(file_dir, instance_id)
        h5_path = osp.join(instance_dir, file_name)

        if not osp.exists(h5_path):
            logger.error(f"Predictions for instance '{instance_id}' not found in: {instance_dir}")
            continue

        predictions[instance_id] = h5_path

    # If we're not lazy, we need to actually load into memory
    if parallel:
        # Load in parallel using multiple processes
        return _load_h5_parallel(
            predictions=predictions,
            instance_ids=instance_ids,
            keys=keys,
            num_workers=num_workers,
        )
    else:
        # Load in serial (single process)
        predictions_in_memory: dict[str, dict[str, np.ndarray]] = {}

        pbar = ProgressBar(total=len(instance_ids), desc="Load Generated Token Attributes", transient=True)
        with pbar:
            for instance_id in instance_ids:
                h5_path = predictions.get(instance_id)
                if not h5_path or not isinstance(h5_path, str):
                    # Missing or invalid path
                    continue

                # Load from file
                with h5py.File(h5_path, "r") as hf:
                    if keys is None:
                        predictions_in_memory[instance_id] = {k: hf[k][()] for k in hf.keys()}
                    else:
                        predictions_in_memory[instance_id] = {k: hf[k][()] for k in keys if k in hf}

                pbar.update()

        return predictions_in_memory


def _load_single_instance(h5_path: str, keys: list[str] = None) -> dict[str, np.ndarray]:
    """
    A top-level helper function (so it can be pickled) that loads data
    from a single HDF5 file path into memory.
    """
    if not osp.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file path does not exist: {h5_path}")

    with h5py.File(h5_path, "r") as hf:
        if keys is None:
            return {k: hf[k][()] for k in hf.keys()}
        else:
            return {k: hf[k][()] for k in keys if k in hf}


def _load_h5_parallel(
    predictions: dict[str, h5py.File | str],
    instance_ids: list[str],
    keys: list[str] = None,
    num_workers: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Helper function to load HDF5 predictions in parallel using multiple processes.
    """

    # We'll populate the final output here
    output: dict[str, dict[str, np.ndarray]] = {}

    # Create the progress bar
    pbar = ProgressBar(total=len(instance_ids), desc="Load Generated Token Attributes", transient=True)

    with pbar, concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_id = {}
        for instance_id in instance_ids:
            h5_path = predictions.get(instance_id)
            if not isinstance(h5_path, str):
                # Log an error if the path is missing or invalid
                logger.error(f"Invalid .h5 file path for instance '{instance_id}'")
                output[instance_id] = {}
                continue

            future = executor.submit(_load_single_instance, h5_path, keys)
            future_to_id[future] = instance_id

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_id):
            instance_id = future_to_id[future]
            try:
                output[instance_id] = future.result()
            except Exception as e:
                logger.error(f"Failed to load predictions for '{instance_id}': {e}")
                output[instance_id] = {}

            # Update progress bar
            pbar.update()

    return output


def save_attrs(attr_dict: dict):
    """
    Save a dictionary of attributes to an HDF5 file.

    Parameters
    ----------
    attr_dict : dict
        Dictionary of attributes to save.
    """
    # Create the directory if it doesn't exist
    file_path = attr_dict.pop("file_path", None)
    metadata = attr_dict.pop("metadata", {})

    os.makedirs(osp.dirname(file_path), exist_ok=True)

    with h5py.File(file_path, "w") as hf:
        for k, v in attr_dict.items():
            if k == "file_path":
                continue
            hf.create_dataset(k, data=v, compression="gzip")

        for k, v in metadata.items():
            hf.attrs[k] = v

    return None


def load_and_prepare_data(
    tokenizer_path: str,
    result_path: str,
    data_path: str,
    preds_dir: str,
    filter_by_attr_present=True,
):
    """
    Load tokenizer, JSON results, and corresponding Parquet data, then merge them.
    Also retrieve paths of all inferred IDs from `preds_dir`.

    Args:
        tokenizer_path (str): Path to the pre-trained tokenizer.
        result_path (str): Path to the JSON file containing inference results.
        data_path (str): Path to the Parquet file containing dataset.
        preds_dir (str): Path to the directory containing token-level predictions.

    Returns:
        tokenizer: Loaded HuggingFace tokenizer.
        df (pd.DataFrame): Merged DataFrame of data and results, filtered to inferred IDs.
    """
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    logger.info(f"Loading results from {result_path}...")
    with open(result_path, "r") as f:
        results = json.load(f)
    result_df = pd.DataFrame(results)

    logger.info(f"Loading dataset from {data_path}...")
    data_df = pd.read_parquet(data_path)

    logger.info("Merging result and data dataframes...")
    df = pd.merge(data_df, result_df, on="idx", how="inner")

    if filter_by_attr_present:
        logger.info(f"Gathering inferred IDs from {preds_dir}...")
        inferred_dirs = natsorted(glob.glob(osp.join(preds_dir, "*")))
        inferred_ids = [osp.basename(d) for d in inferred_dirs]
        df = df[df["idx"].isin(inferred_ids)]

    return tokenizer, df
