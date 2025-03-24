import sys
import random
import os.path as osp
import json
import logging
import warnings
import numpy as np
import pandas as pd
from transformers import set_seed, HfArgumentParser
from seqlbtoolkit.io import set_logging

# Local module imports
from src.utils.vis import plot_length_distribution
from src.utils.io import load_and_prepare_data, load_attrs
from src.args import (
    TrainingArguments,
    PipelineArguments,
    DataArguments,
    AttentionProcessingArguments,
    ModelArguments,
    argument_processing,
)

warnings.filterwarnings("ignore")


# Initialize logger
logger = logging.getLogger(__name__)
set_logging(level="INFO")


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
    ens_path: str,
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
    tokenizer, df = load_and_prepare_data(
        tokenizer_path=model_name_or_path,
        result_path=result_path,
        data_path=ds_path,
        preds_dir=attrs_dir,
    )

    attrs = load_attrs(attrs_dir, "attrs.h5")
    attrs_df = pd.DataFrame(attrs).T.reset_index().rename(columns={"index": "idx"})

    attn_df = pd.DataFrame(attrs_df)
    df = pd.merge(df, attn_df, on="idx", how="inner")

    correct_ids, _, failed_ids = eval_answer_correctness_function(df)
    df["correctness"] = df["idx"].isin(correct_ids)

    if failed_ids:
        logger.warning(f"Failed to evaluate correctness for {len(failed_ids)} instances. Removing them.")
    df = df[~df["idx"].isin(failed_ids)]

    with open(ens_path, "r") as f:
        ens_data = json.load(f)

    inst_ens_probs = []
    for inst_idx, probs in ens_data.items():
        ans_probs = np.asarray(probs["ans_prob"])
        seq_probs = np.asarray(probs["seq_prob"])
        joint = ans_probs * seq_probs
        marginal = joint.sum()
        inst_ens_probs.append({"idx": inst_idx, "ens_marginal": marginal})

    ens_df = pd.DataFrame(inst_ens_probs)
    df = pd.merge(df, ens_df, on="idx", how="inner")

    return df, tokenizer


def get_plots(
    df: pd.DataFrame,
    max_num_instances: int = 2000,
    dataset_name: str = "test",
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Given a DataFrame with evaluation values and labels, compute AUC and ECE
    in multiple passes for a balanced subset of correct/incorrect instances.

    Args:
        df (pd.DataFrame): DataFrame with evaluation metrics and labels.
        max_num_instances (int): Maximum number of correct/incorrect instances to sample.
        n_passes (int): Number of sampling passes for repeated evaluation.
        seed (int): Random seed to ensure reproducibility.

    Returns:
        metric_aggr_df (pd.DataFrame): Aggregated DataFrame of mean and std
                                       for AUC and ECE across the passes.
    """
    correct_df = df[df["label"] == 1]
    incorrect_df = df[df["label"] == 0]

    # Balanced sampling
    n_inst = min(max_num_instances, len(correct_df), len(incorrect_df))
    correct_resp_lengths = random.sample([len(tks) for tks in correct_df["input_ids"]], n_inst)
    incorrect_resp_lengths = random.sample([len(tks) for tks in incorrect_df["input_ids"]], n_inst)
    correct_attn_lengths = random.sample([len(src) for src in correct_df["src_ids"]], n_inst)
    incorrect_attn_lengths = random.sample([len(src) for src in incorrect_df["src_ids"]], n_inst)

    fig_path = osp.join("lengths", f"{dataset_name}-{model_name}-length.pdf")
    plot_length_distribution(
        l_resp_correct=correct_resp_lengths,
        l_resp_incorrect=incorrect_resp_lengths,
        l_attn_correct=correct_attn_lengths,
        l_attn_incorrect=incorrect_attn_lengths,
        figure_path=fig_path,
    )

    fig_path = osp.join("lengths-nolegend", f"{dataset_name}-{model_name}-length.pdf")
    plot_length_distribution(
        l_resp_correct=correct_resp_lengths,
        l_resp_incorrect=incorrect_resp_lengths,
        l_attn_correct=correct_attn_lengths,
        l_attn_incorrect=incorrect_attn_lengths,
        figure_path=fig_path,
        disable_legend=True,
    )

    return None


def main(
    model_args: ModelArguments,
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
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")

    # 1. Load data and attention chain attributes
    df = load_attn_chain_attrs(
        model_name_or_path=get_model(model_args.model_name_or_path),
        ds_path=osp.join(data_args.dataset_dir, "test", f"{data_args.dataset_name}.parquet"),
        result_path=osp.join(data_args.resp_dir, data_args.resp_file),
        attrs_dir=osp.join(data_args.resp_dir, attn_args.attrs_folder),
        ens_path=osp.join(
            data_args.resp_dir,
            attn_args.ensembles_pred_folder,
            f"{attn_args.top_k_similarity}-{attn_args.similarity_threshold:.1f}.json",
        ),
        eval_answer_correctness_function=data_args.attn_kwargs["eval_answer_correctness_function"],
    )

    model_name = osp.basename(model_args.model_name_or_path)
    get_plots(df, dataset_name=data_args.dataset_name, model_name=model_name)

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
    main(model_args, data_args, attn_args)
