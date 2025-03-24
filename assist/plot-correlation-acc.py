import sys
import numpy as np
import os.path as osp
import logging
import warnings
import pandas as pd
from transformers import set_seed, HfArgumentParser
from seqlbtoolkit.io import set_logging
from scipy import stats

# Local module imports
from src.utils.vis import plot_ece_accuracy_correlation
from src.utils.io import load_and_prepare_data, load_attrs
from src.args import (
    TrainingArguments,
    PipelineArguments,
    DataArguments,
    AttentionProcessingArguments,
    ModelArguments,
    argument_processing,
    FUNC_MAP,
)

warnings.filterwarnings("ignore")


# Initialize logger
logger = logging.getLogger(__name__)
set_logging(level="INFO")

model_names = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    # "DeepSeek-R1-Distill-Llama-8B",
]
dataset_names = ["gsm8k", "math", "bbh"]


def get_model(model_path: str) -> str:
    if osp.isdir(model_path) and osp.exists(osp.join(model_path, "config.json")):
        return model_path

    logger.warning(f"Model path '{model_path}' not found. Attempting to resolve model name...")

    name = osp.basename(model_path)
    if "Llama-3" in name:
        return f"meta-llama/{name}"
    elif "gemma" in name:
        return f"google/{name}"
    elif "Qwen" in name:
        return f"Qwen/{name}"
    elif "DeepSeek" in name:
        return f"deepseek-ai/{name}"
    raise ValueError(f"Unknown model name: {name}")


def load_attn_chain_attrs(
    model_name_or_path: str,
    ds_path: str,
    result_path: str,
    attrs_dir: str,
    eval_answer_correctness_function: callable = None,
):
    """
    Load attention chain attributes and merge them with token-level attributes,
    ensemble predictions, and correctness labels into a single dataframe.

    Args:
        model_name_or_path (str): Path or identifier of the model/tokenizer.
        ds_path (str): Path to the dataset file.
        result_path (str): Path to the predictions result file.
        token_attrs_str (str): Path to directory containing token attribute files.
        attn_dir (str): Path to directory containing attention chain attributes.
        ens_path (str): Path to JSON file containing ensemble scores.
        eval_answer_correctness_function (callable): Function to evaluate correctness
                                                     of generated answers.

    Returns:
        df (pd.DataFrame): Merged DataFrame with all relevant attributes.
        tokenizer: The loaded tokenizer.
    """
    # 1. Load tokenizer and base data
    _, df = load_and_prepare_data(
        tokenizer_path=model_name_or_path,
        result_path=result_path,
        data_path=ds_path,
        preds_dir=attrs_dir,
    )

    attrs = load_attrs(attrs_dir, "attrs.h5")
    attrs_df = pd.DataFrame(attrs).T.reset_index().rename(columns={"index": "idx"})

    attn_df = pd.DataFrame(attrs_df)
    df = pd.merge(df, attn_df, on="idx", how="inner")

    # 3. Evaluate correctness and remove failed instances
    correct_ids, _, failed_ids = eval_answer_correctness_function(df)
    accuracy = len(correct_ids) / (len(df) - len(failed_ids))
    return accuracy


def main(
    data_args: DataArguments,
    attn_args: AttentionProcessingArguments,
):
    """
    Orchestrates the data loading, evaluation, and metric computation,
    then saves the final metrics to a CSV file.

    Args:
        model_args (ModelArguments): Arguments controlling model path/name, etc.
        data_args (DataArguments): Arguments controlling dataset and I/O paths.
        attn_args (AttentionProcessingArguments): Arguments controlling attention
                                                  processing and ensemble paths.
    """
    logger.info("\n")
    logger.info("Starting program...")

    uq_name = "ens_marginal"

    # 1. Load data and attention chain attributes
    accuracy_dict = dict()
    ece_dict = dict()
    for dataset in dataset_names:
        accuracy_list = list()
        ece_list = list()
        for model_name in model_names:
            try:
                model_path = osp.join("../models", model_name)
                accuracy = load_attn_chain_attrs(
                    model_name_or_path=get_model(model_path),
                    ds_path=osp.join(data_args.dataset_dir, "test", f"{dataset}.parquet"),
                    result_path=osp.join("./output", dataset, model_name, data_args.resp_file),
                    attrs_dir=osp.join("./output", dataset, model_name, attn_args.attrs_folder),
                    eval_answer_correctness_function=FUNC_MAP[dataset]["eval_answer_correctness_function"],
                )
                metric_path = osp.join("./output", dataset, model_name, f"metrics.csv")
                metric_df = pd.read_csv(metric_path, index_col=0)

                ece = metric_df.loc[uq_name, "ECE-Mean"]

                accuracy_list.append(accuracy)
                ece_list.append(ece)
            except Exception as e:
                pass

        accuracy_dict[dataset] = accuracy_list
        ece_dict[dataset] = ece_list

    fig_path = "./fig/f3-corr-acc.pdf"
    plot_ece_accuracy_correlation(ece_dict, accuracy_dict, fig_path)

    corrs = list()
    for dataset in dataset_names:
        ece = ece_dict[dataset]
        acc = accuracy_dict[dataset]
        corr, _ = stats.pearsonr(ece, acc)
        corrs.append(corr)

    logger.info(f"Correlation Coefficients: {np.mean(corrs)}")
    logger.info("Program completed.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = HfArgumentParser(
        (PipelineArguments, ModelArguments, DataArguments, TrainingArguments, AttentionProcessingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            pipeline_args,
            model_args,
            data_args,
            training_args,
            attn_args,
        ) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (
            pipeline_args,
            model_args,
            data_args,
            training_args,
            attn_args,
        ) = parser.parse_yaml_file(yaml_file=osp.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        (
            pipeline_args,
            model_args,
            data_args,
            training_args,
            attn_args,
        ) = parser.parse_args_into_dataclasses()

    # --- Setup Logging & Seed ---
    set_logging(level="INFO")
    set_seed(training_args.seed)

    # --- Argument Post-processing ---
    argument_processing(model_args, data_args, training_args, attn_args)

    # --- Main Execution ---
    main(data_args, attn_args)
