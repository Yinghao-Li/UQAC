import sys
import os.path as osp
import json
import logging
import warnings
import numpy as np
import pandas as pd
from transformers import set_seed, HfArgumentParser
from seqlbtoolkit.io import set_logging, ProgressBar

# Local module imports
from src.utils.vis import plot_calibration_data_distr_multi_runs
from src.utils.support import thresholded_topk_1d_numpy, find_column_indices
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

    # 5. Load ensemble predictions
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


def get_eval_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract evaluation values from the merged DataFrame by computing probabilities
    and entropies for relevant tokens.

    Args:
        df (pd.DataFrame): The merged DataFrame containing token attributes,
                           attention, etc.
        tokenizer: The tokenizer used to decode tokens.
        extract_answer_span_func (callable): A function that, given the tokenizer
                                             and a text response, returns the span
                                             of the answer tokens.

    Returns:
        result_df (pd.DataFrame): A new DataFrame with computed evaluation metrics.
    """
    result_list = []
    pbar = ProgressBar(total=len(df), desc="Extracting Evaluation Values", transient=True)

    with pbar:
        for (
            idx,
            tkresp_int,
            tkattn_srcidx,
            tkans_srcidx,
            sims,
            top5_tkresp_int,
            top5_tkresp_prob,
            correctness,
        ) in zip(
            df.idx,
            df.tkresp_int,
            df.tkattn_srcidx,
            df.tkans_srcidx,
            df.tkattn_to_tkans_sims,
            df.top5_tkresp_int,
            df.top5_tkresp_prob,
            df.correctness,
        ):
            # 1. Probability for the exact tokens in the response
            resp_tks_rank_in_top5 = find_column_indices(top5_tkresp_int[:-1, :], tkresp_int[1:])
            resp_probs = top5_tkresp_prob[np.arange(len(resp_tks_rank_in_top5)), resp_tks_rank_in_top5]

            # 2. Probability for the tokens that were attended to
            attended_tks = tkresp_int[tkattn_srcidx + 1]
            attended_tks_rank_in_top5 = find_column_indices(top5_tkresp_int[tkattn_srcidx, :], attended_tks)
            attended_probs = top5_tkresp_prob[tkattn_srcidx, attended_tks_rank_in_top5]

            # 3. Probability for tokens in the extracted answer span
            ans_tks = tkresp_int[tkans_srcidx + 1]
            ans_tks_rank_in_top5 = find_column_indices(top5_tkresp_int[tkans_srcidx, :], ans_tks)
            ans_probs = top5_tkresp_prob[tkans_srcidx, ans_tks_rank_in_top5]
            ans_top5_probs = top5_tkresp_prob[tkans_srcidx, :]

            # 4. Probability for top similarities
            sims_src_ids = np.sort(
                tkattn_srcidx[thresholded_topk_1d_numpy(sims[: -len(tkans_srcidx)], k=10, threshold=0.0)]
            )
            sims_tks = tkresp_int[sims_src_ids + 1]
            sims_tks_rank_in_top5 = find_column_indices(top5_tkresp_int[sims_src_ids, :], sims_tks)
            sims_probs = np.r_[top5_tkresp_prob[sims_src_ids, sims_tks_rank_in_top5], ans_probs]

            # 5. Collate metrics
            eval_dict = {
                "idx": idx,
                "label": 1 if correctness else 0,
                "ans_probs_prod": np.prod(ans_probs),
                "ans_probs_mean": np.mean(ans_probs),
                "attn_probs_prod": np.prod(attended_probs),
                "attn_probs_mean": np.mean(attended_probs),
                "sims_probs_prod": np.prod(sims_probs),
                "sims_probs_mean": np.mean(sims_probs),
                "pred_probs_prod": np.prod(resp_probs),
                "pred_probs_mean": np.mean(resp_probs),
                "predictive_entropy": -np.sum(-np.sum(top5_tkresp_prob * np.log(top5_tkresp_prob), axis=-1)),
                "length_norm_entropy": 1 - np.mean(-np.sum(top5_tkresp_prob * np.log(top5_tkresp_prob), axis=-1)),
                "ans_entropy": -np.sum(-np.sum(ans_top5_probs * np.log(ans_top5_probs), axis=-1)),
                "ans_length_norm_entropy": 1 - np.mean(-np.sum(ans_top5_probs * np.log(ans_top5_probs), axis=-1)),
            }
            result_list.append(eval_dict)
            pbar.update()

    # Merge with ensemble marginal
    result_df = pd.DataFrame(result_list)
    ens_df = df[["idx", "ens_marginal"]]
    result_df = pd.merge(result_df, ens_df, on="idx", how="inner")

    return result_df


def get_plots(
    df: pd.DataFrame,
    probs_col_name="ens_marginal",
    max_num_instances: int = 500,
    n_passes: int = 5,
    seed: int = 0,
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

    correct_probs_list = list()
    incorrect_probs_list = list()

    for i in range(n_passes):
        correct_sample = correct_df.sample(n_inst, random_state=seed + i)
        incorrect_sample = incorrect_df.sample(n_inst, random_state=seed + i)
        correct_probs = correct_sample[probs_col_name].values.tolist()
        incorrect_probs = incorrect_sample[probs_col_name].values.tolist()

        correct_probs_list.append(correct_probs)
        incorrect_probs_list.append(incorrect_probs)

    fig_path = osp.join("plots", probs_col_name, f"{dataset_name}-{model_name}-calibration.pdf")
    plot_calibration_data_distr_multi_runs(correct_probs_list, incorrect_probs_list, figure_path=fig_path)

    fig_path = osp.join("plots-nolegend", probs_col_name, f"{dataset_name}-{model_name}-calibration.pdf")
    plot_calibration_data_distr_multi_runs(
        correct_probs_list, incorrect_probs_list, figure_path=fig_path, disable_legend=True
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
    df, tokenizer = load_attn_chain_attrs(
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

    # 2. Extract evaluation values
    result_df = get_eval_values(df=df)

    model_name = osp.basename(model_args.model_name_or_path)
    for col_name in result_df.columns:
        if col_name in ["idx", "label"]:
            continue
        get_plots(
            result_df,
            probs_col_name=col_name,
            max_num_instances=500,
            n_passes=5,
            seed=0,
            dataset_name=data_args.dataset_name,
            model_name=model_name,
        )

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
