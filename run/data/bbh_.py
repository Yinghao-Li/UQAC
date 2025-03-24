"""
# Author: Yinghao Li
# Modified: February 28th, 2025
# ---------------------------------------
# Description: Download and pre-process the BBH dataset
"""

import os
import re
import random
import sys
import os.path as osp
import git
import logging
import glob
import pandas as pd
import json
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args


logger = logging.getLogger(__name__)

DATASET_NAME = "bbh"


def substitute_answer(text):
    """
    Searches within 'text' for the phrase "the answer is ..." (case-insensitive)
    and replaces it with "the answer is \boxed{...}" (preserving any trailing
    period in the output but excluding it from the boxed portion).
    """
    # Explanation of the regex:
    # 1. (?i)                - case-insensitive matching
    # 2. (the answer is\s+)  - capture the phrase "the answer is" plus whitespace (preserves original case and spacing)
    # 3. (.*?)               - non-greedy capture of the answer
    # 4. (\.|$)              - capture either a trailing period or the end of the string
    pattern = r"(?i)(the answer is\s+)(.*?)(\.|$)"

    def replacement(match):
        # Group 1: the phrase "the answer is" plus whitespace (exactly as in the text)
        leading_phrase = match.group(1)
        # Group 2: the core answer (exclude trailing period from it)
        core_answer = match.group(2).strip()
        # Group 3: either '.' or empty string
        trailing_symbol = match.group(3)

        # Wrap the core answer in \boxed{}
        # If there was a period, we put it back after the boxed answer.
        return f"{leading_phrase}\\boxed{{{core_answer}}}{trailing_symbol}"

    # Perform the substitution on all occurrences
    return re.sub(pattern, replacement, text)


@dataclass
class Arguments:
    output_dir: str = field(
        default="./datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    logger.info(f"Processing the {DATASET_NAME} dataset...")

    repo_url = "https://github.com/suzgunmirac/BIG-Bench-Hard.git"
    clone_dir = osp.join(args.output_dir, "tmp", "bbh")
    if not osp.exists(clone_dir):
        logger.info(f"Cloning the {DATASET_NAME} dataset to {clone_dir}...")
        git.Repo.clone_from(repo_url, clone_dir)

    test_json_files = glob.glob(osp.join(clone_dir, "bbh", "*.json"))

    test_data_list = list()
    for test_json_file in test_json_files:
        with open(test_json_file, "r", encoding="utf-8") as f:
            test_json_data = json.load(f)
        task_name = osp.basename(test_json_file).replace(".json", "")
        instances = test_json_data["examples"]

        with open(osp.join(clone_dir, "cot-prompts", f"{task_name}.txt"), "r") as f:
            prompts = f.readlines()
        prompts = prompts[2:]
        instr = prompts[0].strip()
        instr += " Put your final answer in `\\boxed{ }`."
        examples = substitute_answer("".join(prompts[1:]).strip())

        for idx, inst in enumerate(instances):
            q = inst["input"]
            a = inst["target"]
            instruction = f"{instr}\n\nEXAMPLES\n------\n{examples}\n\nQUESTION\n------\nQ: {q}"
            test_data_list.append(
                {
                    "idx": f"{DATASET_NAME}.{task_name}.test.{idx}",
                    "instruction": instruction,
                    "response": a,
                    "category": task_name,
                }
            )

    test_df = pd.DataFrame(test_data_list)

    test_output_dir = osp.join(args.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    logger.info(f"Saving the {DATASET_NAME} dataset to {args.output_dir}...")

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
