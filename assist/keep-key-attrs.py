import sys
import os.path as osp
import json
import logging

import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.metrics import roc_auc_score

from transformers import set_seed, HfArgumentParser

from seqlbtoolkit.io import set_logging, ProgressBar

# Local module imports
from src.utils.io import load_attrs, save_attrs


# Initialize logger
logger = logging.getLogger(__name__)
set_logging(level="INFO")

output_dir = "./output"
datasets = ["gsm8k", "math", "bbh"]

model_names = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B",
]


def main():
    for dataset in datasets:
        for model_name in model_names:
            logger.info(f"Processing {dataset}/{model_name}...")
            tk_attrs = load_attrs(
                file_dir=osp.join(output_dir, dataset, model_name, "token-preds-simplified"),
                keys=["input_ids", "top5_pred_ids", "top5_pred_probs"],
                num_workers=1,
            )
            if not tk_attrs:
                logger.error(f"No token attributes found for {dataset}/{model_name}")
                continue
            for k, v in tk_attrs.items():
                file_path = osp.join(output_dir, dataset, model_name, "key-attrs", k, "predictions.h5")

                v["file_path"] = file_path
                save_attrs(v)
            logger.info(f"Finished processing {dataset}/{model_name}.")


if __name__ == "__main__":

    # --- Setup Logging & Seed ---
    set_logging(level="INFO")

    # --- Main Execution ---
    main()
