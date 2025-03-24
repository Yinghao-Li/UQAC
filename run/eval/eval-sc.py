import os.path as osp
import glob
import torch
import sys
import numpy as np
import logging
import pandas as pd

from transformers import set_seed, HfArgumentParser, AutoTokenizer
from sklearn.metrics import roc_auc_score
from seqlbtoolkit.io import set_logging, ProgressBar

from src.eval.gsm8k import extract_gsm8k_target
from src.eval.math import get_boxed_content, is_math_express_equiv
from src.eval import expected_calibration_error
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


def load_results(preds_dir: str, data_path: str):
    pred_files = glob.glob(osp.join(preds_dir, "*.pt"))
    preds_dict = dict()
    logger.info(f"Loading predictions from: {pred_files}")
    pbar = ProgressBar(total=len(pred_files), desc="Loading predictions")
    with pbar:
        for pred_file in pred_files:
            data = torch.load(pred_file)
            for item in data:
                idx = item["idx"]
                if idx not in preds_dict:
                    preds_dict[idx] = list()
                logprobs = item["logprobs"]

                tks = [list(tk.keys())[0] for tk in logprobs]
                probs = np.exp([list(tk.values())[0] for tk in logprobs])

                preds_dict[idx].append({"tks": tks, "probs": probs})
            pbar.update()

    data_df = pd.read_parquet(data_path)
    data_dict = {idx: gen for idx, gen in zip(data_df.idx, data_df.response)}

    return preds_dict, data_dict


def get_answer(preds_dict, tokenizer, extract_answer_span_func):
    n_failed = 0
    ans_dict = dict()
    pbar = ProgressBar(total=len(preds_dict), desc="Extracting answers")
    with pbar:
        for idx, preds in preds_dict.items():
            choices = dict()
            for pred in preds:
                tks = pred["tks"]
                probs = pred["probs"]

                resp = tokenizer.decode(tks, skip_special_tokens=False)

                try:
                    answer_span = extract_answer_span_func(tokenizer, resp)
                except (TypeError, ValueError):
                    continue

                ans = tokenizer.decode(tks[answer_span[0][0] : answer_span[-1][-1]], skip_special_tokens=False)
                prob = np.prod(probs[answer_span[0][0] : answer_span[-1][-1]])
                choices[ans] = choices.get(ans, 0) + prob / len(preds)

            pbar.update()

            if not choices:
                n_failed += 1
                continue
            best_ans = max(choices, key=choices.get)
            ans_dict[idx] = (best_ans, choices[best_ans])

    logger.info(f"Failed to extract answer for {n_failed} instances.")

    return ans_dict


def get_correctness(data_dict, ans_dict, dataset_name):
    df_list = list()
    if dataset_name == "gsm8k":
        for idx, (ans, prob) in ans_dict.items():
            if idx not in data_dict:
                continue
            ref_seq = data_dict[idx]
            try:
                ref = extract_gsm8k_target(ref_seq)
                ref = float(ref)
                ans = float(ans)
            except Exception:
                continue
            is_correct = int(ref == ans)
            df_list.append({"idx": idx, "label": is_correct, "prob": prob})

    elif dataset_name == "math":
        for idx, (ans, prob) in ans_dict.items():
            if idx not in data_dict:
                continue
            ref_seq = data_dict[idx]
            try:
                ref = get_boxed_content(ref_seq)
            except Exception:
                continue
            is_correct = is_math_express_equiv(ans, ref)
            df_list.append({"idx": idx, "label": is_correct, "prob": prob})

    elif dataset_name == "bbh":
        for idx, (ans, prob) in ans_dict.items():
            if idx not in data_dict:
                continue
            ref = data_dict[idx].lower()
            is_correct = ans.lower() == ref
            df_list.append({"idx": idx, "label": is_correct, "prob": prob})

    df = pd.DataFrame(df_list)
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
):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    preds_dict, data_dict = load_results(
        preds_dir=data_args.resp_dir,
        data_path=osp.join(data_args.dataset_dir, "test", f"{data_args.dataset_name}.parquet"),
    )
    ans_dict = get_answer(
        preds_dict,
        tokenizer,
        extract_answer_span_func=data_args.attn_kwargs["extract_answer_span_function"],
    )
    result_df = get_correctness(data_dict, ans_dict, dataset_name=data_args.dataset_name)
    metric_df = get_metrics(result_df, max_num_instances=500, n_passes=5, seed=0)

    save_path = osp.join(data_args.resp_dir, "metrics.csv")
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
    main(model_args, data_args)
