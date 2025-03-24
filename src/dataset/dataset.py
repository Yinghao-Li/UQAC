"""
dataset.py

Author: Yinghao Li
Modified: March 18th, 2025
---------------------------------------
Description: Class definition for dataset loading, subset selection,
and references to the external processing pipeline.
"""

import os.path as osp
import numpy as np
import pandas as pd
from natsort import natsorted
from accelerate.logging import get_logger
from datasets import load_dataset, Dataset

from src.utils.io import load_attrs
from .processor import (
    process_default,
    expand_samples_for_ens_inference,
    flatten_ds,
)

logger = get_logger(__name__)


class CausalLMDataset:
    """
    Main dataset handler that loads local data, merges predictions, merges
    attention chain attributes, optionally subsamples, and references external
    processing methods to tokenize/process the dataset before use.
    """

    def __init__(
        self,
        task: str,
        dataset_dir: str,
        dataset_name: str,
        model_name: str,
        resp_dir: str = None,
        resp_file: str = None,
        tokenizer=None,
        subsample: int | float = 0,
        max_seq_length: int = 128,
        disable_dataset_cache: bool = False,
        disable_seq_length_filter: bool = False,
        dataset_sorted_by_length: bool = False,
        seed: int = 0,
        attn_kwargs: dict = None,
        **kwargs,
    ) -> None:
        """
        Args:
            task: The running mode, e.g. 'train', 'infer-uq', 'infer-ens', etc.
            dataset_dir: Local directory that contains the dataset files.
            dataset_name: Name of the dataset (used to locate files).
            model_name: Name of the model (used to adapt formatting).
            pred_dir: Directory containing prediction results.
            pred_file: Filename of the prediction results.
            tokenizer: Tokenizer to encode text into tokens.
            partition: Either "train" or "test".
            subsample: Either 0 (no subsampling) or an integer/float specifying
                       the number/portion of dataset to keep.
            max_seq_length: Maximum allowable sequence length for filtering.
            disable_dataset_cache: Whether to disable loading from HF cache.
            disable_seq_length_filter: Whether to skip the max_seq_length filter.
            dataset_sorted_by_length: Whether to sort the dataset by length.
            seed: Random seed for subsampling.
            attn_kwargs: Additional arguments for attention chain loading.
            kwargs: Any additional arguments.
        """
        self.task = task

        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.model_name = model_name

        self.pred_dir = resp_dir
        self.pred_file = resp_file

        self.tokenizer = tokenizer
        self.disable_cache = disable_dataset_cache
        self.disable_seq_length_filter = disable_seq_length_filter
        self.max_seq_length = max_seq_length
        self.attn_kwargs = attn_kwargs or {}

        self.subsample = subsample
        self.subset_ids = None
        self.sorted_by_length = dataset_sorted_by_length
        self.seed = seed

        self.ds = None

        # columns to remove after processing
        self.columns_to_remove_after_proc = [
            "instruction",
            "response",
            "category",
            "generated",
            "scores",
            "sims",
            "top5_pred_ids",
            "top5_pred_probs",
            "src_ids",
            "input_ids",
        ]

    def load(self) -> "CausalLMDataset":
        """
        Load the dataset from a local directory, optionally merging predictions
        and attention chain attributes (depending on the task).
        """
        assert self.dataset_dir, "You must provide a local dataset directory to load the dataset."

        ds_path = osp.join(self.dataset_dir, "test", f"{self.dataset_name}.parquet")
        logger.info(f"Loading dataset from: {ds_path}")

        loaded_data = load_dataset("parquet", data_files={"test": ds_path})
        self.ds = loaded_data["test"]

        if self.task in ("infer-uq", "infer-vu", "infer-ens"):
            self._load_pred_results()

        if self.task == "infer-ens":
            self._load_attn_chain_attrs()

        self._subsample_instances()
        return self

    def _load_pred_results(self):
        """Load prediction results from a specified JSON file and merge with ds."""
        if not self.pred_dir or not self.pred_file:
            return

        pred_path = osp.join(self.pred_dir, self.pred_file)
        if not osp.exists(pred_path):
            raise FileNotFoundError(f"Prediction file {pred_path} not found!")

        logger.info(f"Loading prediction results from {pred_path}")
        pred_df = pd.read_json(pred_path)

        pred_ids = natsorted(pred_df["idx"].tolist())
        ds_ids = natsorted(self.ds["idx"])

        # If IDs don't match, we'll do an inner merge on "idx"
        if pred_ids != ds_ids:
            logger.warning(
                f"Prediction IDs do not perfectly match dataset IDs. " f"Only the overlapping subset will be used."
            )

        ds_df = self.ds.to_pandas()
        merged_df = pd.merge(ds_df, pred_df, on="idx", how="inner")
        self.ds = Dataset.from_pandas(merged_df)

    def _load_attn_chain_attrs(self):
        """
        Load attention chain attributes and merge them into the dataset.
        """
        # Gather all attention chain attribute directories
        attrs_folder = self.attn_kwargs.get("attrs_folder", "")
        attrs_file_name = self.attn_kwargs.get("attrs_file_name", "")
        attrs = load_attrs(osp.join(self.pred_dir, attrs_folder), attrs_file_name)
        attrs_df = pd.DataFrame(attrs).T.reset_index().rename(columns={"index": "idx"})

        attn_chain_ids = natsorted(attrs_df["idx"].tolist())
        ds_ids = natsorted(self.ds["idx"])
        if attn_chain_ids != ds_ids:
            logger.warning(f"Attention chain IDs do not match dataset IDs. " f"Using only overlapping subset.")

        ds_df = self.ds.to_pandas()
        df = pd.merge(ds_df, attrs_df, on="idx", how="inner")
        # flatten the top5_pred_ids and top5_pred_probs
        df["top5_tkresp_int"] = df["top5_tkresp_int"].apply(lambda x: np.concatenate(x))
        df["top5_tkresp_prob"] = df["top5_tkresp_prob"].apply(lambda x: np.concatenate(x))

        self.ds = Dataset.from_pandas(df)

        return None

    def _subsample_instances(self):
        """Subsamples or filters the dataset if a subset_ids list or subsample ratio is provided."""
        if not self.ds:
            logger.warning("No dataset loaded to subsample.")
            return

        ds = self.ds
        if self.subset_ids:
            # Subsampling by explicit IDs
            logger.warning("Subsampling the dataset using provided subset_ids.")
            idx_map = {idx: i for i, idx in enumerate(ds["idx"])}
            ids_to_keep = [idx_map[x] for x in self.subset_ids if x in idx_map]
            self.ds = ds.select(ids_to_keep)
        else:
            # Subsampling by ratio or absolute number
            sub = self.subsample
            if sub:
                n_subsample = int(len(ds) * sub) if isinstance(sub, float) else int(sub)
                logger.warning(f"Subsampling the dataset to {n_subsample} instances out of {len(ds)}.")
                self.ds = ds.shuffle(seed=self.seed).select(range(n_subsample))

    def process(
        self,
        tokenizer=None,
        dataset_text_field: str = "text",
        num_mapping_proc=8,
        num_filtering_proc=1,
    ) -> "CausalLMDataset":
        """
        Map/filter the dataset with a relevant processing function. The heavy
        tokenization/formatting logic is offloaded to external functions.
        """
        if not self.ds:
            raise RuntimeError("You must load the dataset before preparing it.")

        assert tokenizer is not None, "A tokenizer must be provided to encode the dataset."

        # Choose the appropriate per-sample processing function
        if self.task == "infer-ens":
            process_func = expand_samples_for_ens_inference
        else:
            process_func = process_default

        load_cache = False if self.disable_cache else None

        # 1. Map (tokenization/formatting)
        ds_before = len(self.ds)
        ds_processed = self.ds.map(
            process_func,
            num_proc=num_mapping_proc,
            load_from_cache_file=load_cache,
            fn_kwargs={
                "tokenizer": tokenizer,
                "dataset_name": self.dataset_name,
                "task": self.task,
                "model_name": self.model_name,
                "attn_kwargs": self.attn_kwargs,
                "dataset_text_field": dataset_text_field,
                "max_seq_length": self.max_seq_length,
            },
        )

        # 2. Sort by sequence length if requested
        if self.sorted_by_length:
            ds_processed = ds_processed.sort("n_tks", reverse=True)

        # 3. Filter out sequences that are too long or that failed processing
        if self.disable_seq_length_filter:
            filter_func = lambda ex: ex["keep_instance"]
        else:
            filter_func = lambda ex: ex["keep_instance"] and ex["n_tks"] < self.max_seq_length

        ds_processed = ds_processed.filter(
            filter_func,
            num_proc=num_filtering_proc,
            load_from_cache_file=load_cache,
        )
        ds_after = len(ds_processed)

        # 4. Remove unneeded columns
        if self.task == "infer-ens":
            # Keep only the 'expanded' column
            columns_to_remove = [c for c in ds_processed.column_names if c != "expanded"]
        else:
            columns_to_remove = [
                cols
                for cols in ds_processed.column_names
                if cols in ["n_tks", "keep_instance"] + self.columns_to_remove_after_proc
            ]
        ds_processed = ds_processed.remove_columns(columns_to_remove)

        # 5. In the "infer-ens" case, flatten expansions
        if self.task == "infer-ens":
            ds_processed = flatten_ds(ds_processed, "expanded")

        self.ds = ds_processed

        if ds_before != ds_after:
            logger.warning(f"Filtered out {ds_before - ds_after} samples that were too long or invalid.")

        return self
