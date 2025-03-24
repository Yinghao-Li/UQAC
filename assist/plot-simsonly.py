import sys
import numpy as np
import os.path as osp
import logging
import warnings
import pandas as pd
from seqlbtoolkit.io import set_logging
from scipy import stats

# Local module imports
from src.utils.vis import plot_simsonly

warnings.filterwarnings("ignore")


# Initialize logger
logger = logging.getLogger(__name__)
set_logging(level="INFO")

model_names = [
    "Llama-3.2-1B-Instruct",
    "gemma-2-2b-it",
    "Qwen2.5-1.5B-Instruct",
]
dataset_names = ["gsm8k", "math", "bbh"]
dataset_mapping = {
    "gsm8k": "GSM8k",
    "math": "MATH",
    "bbh": "BBH",
}


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

    metric_filename = "metrics.csv"
    simsonly_filename = "metrics-simsonly.csv"

    model_dict = dict()
    for model_name in model_names:
        data_dict = dict()
        for dataset in dataset_names:
            metric_df = pd.read_csv(osp.join("output", dataset, model_name, metric_filename), index_col=0)
            simsonly_df = pd.read_csv(osp.join("output", dataset, model_name, simsonly_filename), index_col=0)

            uqac_auc = metric_df.loc["ens_marginal", "AUC-Mean"]
            uqac_ece = metric_df.loc["ens_marginal", "ECE-Mean"]
            simsonly_auc = simsonly_df.loc["sims_probs_prod", "AUC-Mean"]
            simsonly_ece = simsonly_df.loc["sims_probs_prod", "ECE-Mean"]

            data_dict[f"{dataset_mapping[dataset]}-UQAC"] = {"AUC": uqac_auc, "ECE": uqac_ece}
            data_dict[f"{dataset_mapping[dataset]}-w/o attn"] = {"AUC": simsonly_auc, "ECE": simsonly_ece}

        data_df = pd.DataFrame(data_dict).T

        model_dict[model_name] = data_df

    df_mean = (model_dict[model_names[0]] + model_dict[model_names[1]] + model_dict[model_names[2]]) / 3
    model_dict["Average"] = df_mean
    # Prepare a list to hold DataFrames whose columns have been relabeled with a MultiIndex
    df_list = []

    for model, df in model_dict.items():
        # Create a MultiIndex for the columns: (model, original_column_name)
        df.columns = pd.MultiIndex.from_product([[model], df.columns])
        df_list.append(df)

    # Concatenate along the columns axis
    combined_df = pd.concat(df_list, axis=1)
    combined_df *= 100  # Convert to percentage

    # Convert to LaTeX
    latex_str = combined_df.to_latex(
        multicolumn=True,
        multirow=True,
        float_format="%.2f",  # optional, sets floating-point precision
    )
    print(latex_str)

    logger.info("Program completed.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # --- Setup Logging & Seed ---
    set_logging(level="INFO")

    # --- Main Execution ---
    main()
