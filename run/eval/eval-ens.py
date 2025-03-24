import os.path as osp
import glob
import json
import sys
import numpy as np
import logging
import random
import pandas as pd
from natsort import natsorted

from transformers import set_seed, HfArgumentParser
from sklearn.metrics import roc_auc_score
from seqlbtoolkit.io import set_logging

from src.eval import expected_calibration_error
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
    token_attrs_str: str,
    ens_dir: str,
    eval_answer_correctness_function: callable = None,
):
    _, df = load_and_prepare_data(
        tokenizer_path=model_name_or_path,
        result_path=result_path,
        data_path=ds_path,
        preds_dir=token_attrs_str,
    )

    correct_ids, _, failed_ids = eval_answer_correctness_function(df)
    df = df[~df["idx"].isin(failed_ids)]
    df["correctness"] = df["idx"].isin(correct_ids)
    df["label"] = df["correctness"].astype(int)

    ens_files = natsorted(glob.glob(osp.join(ens_dir, "*.json")))
    for ens_path in ens_files:
        basename = osp.basename(ens_path).replace(".json", "")

        with open(ens_path, "r") as f:
            ens_data = json.load(f)

        inst_ens_probs = list()
        for inst_idx, probs in ens_data.items():
            ans_probs = np.asarray(probs["ans_prob"])
            seq_probs = np.asarray(probs["seq_prob"])

            joint = ans_probs * seq_probs
            marginal = joint.sum()
            inst_ens_probs.append({"idx": inst_idx, f"ens-{basename}": marginal})

        ens_df = pd.DataFrame(inst_ens_probs)
        df = pd.merge(df, ens_df, on="idx", how="inner")

    df = df.filter(items=["idx", "label"] + [c for c in df.columns if "ens-" in c], axis=1)
    return df


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
    df = load_attn_chain_attrs(
        model_name_or_path=get_model(model_args.model_name_or_path),
        ds_path=osp.join(data_args.dataset_dir, "test", f"{data_args.dataset_name}.parquet"),
        result_path=osp.join(data_args.resp_dir, data_args.resp_file),
        token_attrs_str=osp.join(data_args.resp_dir, attn_args.attrs_folder),
        ens_dir=osp.join(data_args.resp_dir, attn_args.ensembles_pred_folder),
        eval_answer_correctness_function=data_args.attn_kwargs["eval_answer_correctness_function"],
    )
    metric_df = get_metrics(df, max_num_instances=500, n_passes=5, seed=0)

    save_path = osp.join(data_args.resp_dir, "metrics-ens.csv")
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
