"""
# Author: Yinghao Li
# Modified: March 13th, 2025
# ---------------------------------------
# Description: Download and pre-process the MATH dataset
"""

import os
import sys
import os.path as osp
import logging
import glob
import pandas as pd
import wget
import json
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args


logger = logging.getLogger(__name__)
accelerator = Accelerator()

DATASET_NAME = "math"


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="./datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    logger.info(f"Processing the {DATASET_NAME} dataset...")

    # url = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
    # download_path = osp.join(args.output_dir, "tmp", "MATH.tar")
    url = "https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip"
    download_path = osp.join(args.output_dir, "tmp", "MATH.zip")
    if not osp.exists(download_path):
        logger.info(f"Downloading the {DATASET_NAME} dataset to {download_path}...")
        os.makedirs(osp.dirname(download_path), exist_ok=True)
        wget.download(url, download_path)

    unzip_dir = osp.join(args.output_dir, "tmp", "MATH")
    if not osp.exists(unzip_dir):
        logger.info(f"Unzipping the {DATASET_NAME} dataset to {unzip_dir}...")
        if download_path.endswith(".zip"):
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(path=osp.dirname(download_path))
        elif download_path.endswith(".tar"):
            with tarfile.open(download_path, "r") as tar:
                tar.extractall(path=osp.dirname(download_path))

    training_files = glob.glob(osp.join(unzip_dir, "train", "**", "*.json"), recursive=True)
    test_files = glob.glob(osp.join(unzip_dir, "test", "**", "*.json"), recursive=True)

    training_data_list = list()
    for idx, file_path in enumerate(training_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["idx"] = idx
        training_data_list.append(data)

    test_data_list = list()
    for idx, file_path in enumerate(test_files):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["idx"] = idx
        test_data_list.append(data)

    training_df = pd.DataFrame(training_data_list, columns=["problem", "solution", "type"])
    test_df = pd.DataFrame(test_data_list, columns=["problem", "solution", "type"])
    training_df.rename(columns={"problem": "instruction", "solution": "response", "type": "category"}, inplace=True)
    test_df.rename(columns={"problem": "instruction", "solution": "response", "type": "category"}, inplace=True)

    training_df["idx"] = [f"math.{cat}.train.{idx}" for (idx, cat) in enumerate(training_df["category"])]
    test_df["idx"] = [f"math.{cat}.test.{idx}" for (idx, cat) in enumerate(test_df["category"])]

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
