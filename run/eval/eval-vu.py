import os.path as osp
import sys
import numpy as np
import logging
import pandas as pd

from transformers import set_seed, HfArgumentParser
from sklearn.metrics import roc_auc_score
from seqlbtoolkit.io import set_logging, ProgressBar

from src.eval import expected_calibration_error, get_boxed_content
from src.utils.io import load_and_prepare_data
from src.args import (
    TrainingArguments,
    PipelineArguments,
    DataArguments,
    AttentionProcessingArguments,
    ModelArguments,
    argument_processing,
)


logger = logging.getLogger(__name__)
set_logging(level="INFO")


def load_attn_chain_attrs(
    model_name_or_path: str,
    ds_path: str,
    result_path: str,
    token_attrs_str: str,
    attn_dir: str,
    ens_path: str,
    eval_answer_correctness_function: callable = None,
):
    tokenizer, df = load_and_prepare_data(
        tokenizer_path=model_name_or_path,
        result_path=result_path,
        data_path=ds_path,
        preds_dir=token_attrs_str,
        filter_by_attr_present=False,
    )

    correct_ids, _, failed_ids = eval_answer_correctness_function(df)
    df["correctness"] = df["idx"].isin(correct_ids)

    if failed_ids:
        logger.warning(f"Failed to evaluate correctness for {len(failed_ids)} instances. Will remove them.")
    df = df[~df["idx"].isin(failed_ids)]

    vu_result_path = result_path.replace(".json", "_vu.json")
    vu_df = pd.read_json(vu_result_path, orient="records")
    vu_df.rename(columns={"idx": "idx", "generated": "vu"}, inplace=True)

    df = pd.merge(df, vu_df, on="idx", how="inner")

    return df, tokenizer


def get_eval_values(df: pd.DataFrame):
    result_list = list()
    pbar = ProgressBar(total=len(df), desc="Extracting Evaluation Values", transient=True)
    with pbar:
        for idx, vu, correctness in zip(df.idx, df.vu, df.correctness):

            try:
                vu_value = get_boxed_content(vu)
                vu_value = float(vu_value) / 10
            except (ValueError, TypeError):
                continue

            if not 0 <= vu_value <= 1:
                continue
            eval_dict = {
                "idx": idx,
                "label": 1 if correctness else 0,
                "vu": vu_value,
            }
            result_list.append(eval_dict)
            pbar.update()

    result_df = pd.DataFrame(result_list)

    return result_df


def get_metrics(df: pd.DataFrame, max_num_instances: int = 500, n_passes: int = 5, seed: int = 0):
    correct_df = df[df["label"] == 1]
    incorrect_df = df[df["label"] == 0]

    n_inst = min(max_num_instances, len(correct_df), len(incorrect_df))
    metric_keys = correct_df.columns[2:]
    metric_dict = {k: {"AUC": list(), "ECE": list()} for k in metric_keys}

    for i in range(n_passes):
        logger.info(f"Pass {i + 1}/{n_passes}")
        correct_df_ = correct_df.sample(n_inst, random_state=seed + i)
        incorrect_df_ = incorrect_df.sample(n_inst, random_state=seed + i)
        lbs = correct_df_["label"].tolist() + incorrect_df_["label"].tolist()
        keys = correct_df_.columns[2:]

        for k in keys:
            auc = roc_auc_score(lbs, correct_df_[k].tolist() + incorrect_df_[k].tolist())
            ece = expected_calibration_error(correct_df_[k].tolist(), incorrect_df_[k].tolist(), n_bins=20)
            metric_dict[k]["AUC"].append(auc)
            metric_dict[k]["ECE"].append(ece)

    metric_aggr_dict = dict()
    for k in metric_keys:
        auc_mean = np.mean(metric_dict[k]["AUC"])
        auc_std = np.std(metric_dict[k]["AUC"])
        ece_mean = np.mean(metric_dict[k]["ECE"])
        ece_std = np.std(metric_dict[k]["ECE"])
        metric_aggr_dict[k] = {
            "AUC-Mean": auc_mean,
            "AUC-Std": auc_std,
            "ECE-Mean": ece_mean,
            "ECE-Std": ece_std,
        }
    metric_aggr_df = pd.DataFrame(metric_aggr_dict).T

    return metric_aggr_df


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    attn_args: AttentionProcessingArguments,
):
    df, tokenizer = load_attn_chain_attrs(
        model_name_or_path=model_args.model_name_or_path,
        ds_path=osp.join(data_args.dataset_dir, "test", f"{data_args.dataset_name}.parquet"),
        result_path=osp.join(data_args.resp_dir, data_args.resp_file),
        token_attrs_str=osp.join(data_args.resp_dir, attn_args.attrs_folder),
        attn_dir=osp.join(data_args.resp_dir, attn_args.attrs_folder),
        ens_path=osp.join(
            data_args.resp_dir,
            attn_args.ensembles_pred_folder,
            f"{attn_args.top_k_similarity}-{attn_args.similarity_threshold:.1f}.json",
        ),
        eval_answer_correctness_function=data_args.attn_kwargs["eval_answer_correctness_function"],
    )
    result_df = get_eval_values(df=df)
    metric_df = get_metrics(result_df, max_num_instances=500, n_passes=5, seed=0)
    save_path = osp.join(data_args.resp_dir, "metrics-vu.csv")
    logger.info(f"Saving metrics to: {save_path}")
    metric_df.to_csv(save_path, index=True)
    logger.info("Program completed.")

    return None


if __name__ == "__main__":
    # --- set up arguments ---
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

    set_logging(level="INFO")
    set_seed(training_args.seed)

    argument_processing(model_args, data_args, training_args, attn_args)
    main(model_args, data_args, attn_args)
