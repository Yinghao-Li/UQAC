import sys
import numpy as np
import os.path as osp
import logging
import warnings
import pandas as pd
from seqlbtoolkit.io import set_logging
from scipy import stats

# Local module imports
from src.utils.vis import plot_ece_auc_correlation

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


def main():
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
    auc_dict = dict()
    ece_dict = dict()
    for dataset in dataset_names:
        auc_list = list()
        ece_list = list()
        for model_name in model_names:
            try:
                metric_path = osp.join("./output", dataset, model_name, f"metrics.csv")
                metric_df = pd.read_csv(metric_path, index_col=0)

                ece = metric_df.loc[uq_name, "ECE-Mean"]
                auc = metric_df.loc[uq_name, "AUC-Mean"]

                ece_list.append(ece)
                auc_list.append(auc)

            except Exception as e:
                pass

        auc_dict[dataset] = auc_list
        ece_dict[dataset] = ece_list

    fig_path = "./fig/f3-corr-auc.pdf"
    plot_ece_auc_correlation(ece_dict, auc_dict, fig_path, disable_legend=True)

    corrs = list()
    for dataset in dataset_names:
        ece = ece_dict[dataset]
        auc = auc_dict[dataset]
        corr, _ = stats.pearsonr(ece, auc)
        corrs.append(corr)

    logger.info(f"Correlation Coefficients: {np.mean(corrs)}")
    logger.info("Program completed.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # --- Setup Logging & Seed ---
    set_logging(level="INFO")

    # --- Main Execution ---
    main()
