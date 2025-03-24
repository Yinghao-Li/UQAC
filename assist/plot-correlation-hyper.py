import sys
import numpy as np
import os.path as osp
import logging
import warnings
import pandas as pd
from seqlbtoolkit.io import set_logging
from scipy import stats

# Local module imports
from src.utils.vis import plot_hyper_correlation

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
    dataset = "gsm8k"

    auc_list = list()

    ece_list = list()
    for model_name in model_names:
        try:
            metric_path = osp.join("./output", dataset, model_name, f"metrics-ens.csv")
            metric_df = pd.read_csv(metric_path, index_col=0)
            metric_df = metric_df[~metric_df.index.astype(str).str.contains("0.5")]
            ece_list.append(metric_df["ECE-Mean"])
            auc_list.append(metric_df["AUC-Mean"])

        except Exception as e:
            pass

    fig_path = "./fig/f3-corr-hyper.pdf"
    plot_hyper_correlation(ece_list, auc_list, fig_path, disable_legend=True)

    corrs = list()
    for ece, auc in zip(ece_list, auc_list):
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
