"""
# Author: Yinghao Li
# Modified: December 3rd, 2024
# ---------------------------------------
# Description: Download and pre-process the GSM8K dataset
"""

import os
import sys
import os.path as osp
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args


logger = logging.getLogger(__name__)
accelerator = Accelerator()

DATASET_NAME = "gsm8k"


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="./datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    logger.info(f"Downloading and processing the {DATASET_NAME} dataset...")
    splits = {"train": "main/train-00000-of-00001.parquet", "test": "main/test-00000-of-00001.parquet"}
    training_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])

    training_df.rename(columns={"question": "instruction", "answer": "response"}, inplace=True)
    test_df.rename(columns={"question": "instruction", "answer": "response"}, inplace=True)

    training_df["category"] = "gsm8k"
    test_df["category"] = "gsm8k"

    training_df["idx"] = [f"gsm8k.gsm8k.train.{idx}" for idx in range(len(training_df))]
    test_df["idx"] = [f"gsm8k.gsm8k.test.{idx}" for idx in range(len(test_df))]

    training_output_dir = osp.join(args.output_dir, "train")
    os.makedirs(training_output_dir, exist_ok=True)

    test_output_dir = osp.join(args.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    logger.info(f"Saving the {DATASET_NAME} dataset to {args.output_dir}...")

    training_df.to_parquet(osp.join(training_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Training dataset saved to {training_output_dir}")

    test_df.to_parquet(osp.join(test_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Test dataset saved to {test_output_dir}")

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser((Arguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(yaml_file=osp.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging(level="INFO")
    logging_args(arguments)

    main(arguments)
