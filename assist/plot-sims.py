import numpy as np
import os.path as osp
import logging
import warnings
import pandas as pd
from seqlbtoolkit.io import set_logging

# Local module imports
from src.utils.vis import plot_sims

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
    "DeepSeek-R1-Distill-Llama-8B",
]
model_name_map = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-2-2b-it": "gemma2-2b",
    "gemma-2-9b-it": "gemma2-9b",
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-8B",
}
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

    dataset_resp_sims = list()
    dataset_attn_sims = list()

    for dataset in dataset_names:
        dfs = None
        for model_name in model_names:
            try:
                sims_path = osp.join("./output", dataset, model_name, f"inves-sims.csv")
                df = pd.read_csv(sims_path)
                df.index = [model_name_map[model_name]]

                if dfs is None:
                    dfs = df
                else:
                    dfs = pd.concat([dfs, df], axis=0)

            except Exception as e:
                pass

        dataset_resp_sims.append(np.mean(dfs.resp_sim))
        dataset_attn_sims.append(np.mean(dfs.attn_sim))

        fig_path = f"./fig/f5-sims-{dataset}.pdf"
        plot_sims(dfs, fig_path)

    logger.info(f"Average response similarity {np.mean(dataset_resp_sims):.4f}")
    logger.info(f"Average attention similarity {np.mean(dataset_attn_sims):.4f}")

    logger.info("Program completed.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # --- Setup Logging & Seed ---
    set_logging(level="INFO")

    # --- Main Execution ---
    main()
