import os.path as osp
import pandas as pd
import logging
import os

from seqlbtoolkit.io import set_logging


logger = logging.getLogger(__name__)
set_logging(level="INFO")

output_dir = "./output"
sc_output_dir = "./output-sc"
metric_file_name = "metrics.csv"
vu_file_name = "metrics-vu.csv"
dataset = "bbh"
latex_out_dir = "./tables/"

dataset_name_mapping = {
    "gsm8k": "GSM8k",
    "math": "MATH",
    "bbh": "BBH",
}

# model_names = [
#     "Llama-3.2-1B-Instruct",
#     "Llama-3.2-3B-Instruct",
#     "Meta-Llama-3.1-8B-Instruct",
#     "gemma-2-2b-it",
#     "gemma-2-9b-it",
#     "Qwen2.5-1.5B-Instruct",
#     "Qwen2.5-3B-Instruct",
#     "Qwen2.5-7B-Instruct",
#     "DeepSeek-R1-Distill-Llama-8B",
# ]
model_names = ["DeepSeek-R1-Distill-Llama-8B"]


def csv_to_latex_table(df, label="tab:results", caption="Results table", decimal_places=3):
    """
    Reads a CSV file and converts it to a LaTeX table of the form:
    Method & AUC (mean ± std) & ECE (mean ± std).

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
    label : str
        Label for the LaTeX table (for referencing in LaTeX).
    caption : str
        Caption for the LaTeX table.
    decimal_places : int
        Number of decimal places for mean and std.
    """

    latex_str = "\\begin{table}[ht]\n"
    latex_str += "    \\centering\n"
    latex_str += f"    \\caption{{{caption}}}\n"
    latex_str += f"    \\label{{{label}}}\n"
    latex_str += "    \\begin{tabular}{llcc}\n"
    latex_str += "    \\toprule\n"
    latex_str += "    & & AUROC & ECE \\\\ \n"
    latex_str += "    \\midrule\n"

    # Construct each row of the table
    for idx, (method, row) in enumerate(df.iterrows()):
        auc_mean = f"{row['AUC-Mean']:.{decimal_places}f}"
        auc_std = f"{row['AUC-Std']:.{decimal_places}f}"
        ece_mean = f"{row['ECE-Mean']:.{decimal_places}f}"
        ece_std = f"{row['ECE-Std']:.{decimal_places}f}"

        if idx == 0:
            latex_str += "    \\multirow{4}{*}{Token Probability}\n"
        if idx == 4:
            latex_str += "    \\multirow{4}{*}{Token Entropy}\n"
        if idx == 8:
            latex_str += "    \\multirow{2}{*}{Multi-Round Prompting}\n"
        if idx == 10:
            latex_str += "    \\multirow{5}{*}{\\ours}\n"

        if idx in (6, 7):
            latex_str += f"    & {method} & \\num{{{auc_mean}}} $\\pm$ \\num{{{auc_std}}} & - \\\\ \n"
        else:
            latex_str += f"    & {method} & \\num{{{auc_mean}}} $\\pm$ \\num{{{auc_std}}} & \\num{{{ece_mean}}} $\\pm$ \\num{{{ece_std}}} \\\\ \n"

        if idx in (3, 7, 9):
            latex_str += "    \\midrule\n"

    latex_str += "    \\bottomrule\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "\\end{table}\n"

    return latex_str


def main():

    for idx, model_name in enumerate(model_names):
        path = osp.join(output_dir, dataset, model_name, metric_file_name)
        if not osp.exists(path):
            logger.warning(f"Skipping {model_name} as file does not exist: {path}")
            continue

        df = pd.read_csv(path, index_col=0)

        path = osp.join(output_dir, dataset, model_name, vu_file_name)
        df_vu = pd.read_csv(path, index_col=0)
        df = pd.concat([df, df_vu], axis=0)

        path = osp.join(sc_output_dir, dataset, model_name, metric_file_name)
        df_sc = pd.read_csv(path, index_col=0)
        df = pd.concat([df, df_sc], axis=0)
        df *= 100

        desired_order = [
            "ans_probs_mean",
            "pred_probs_mean",
            "ans_probs_prod",
            "pred_probs_prod",
            "length_norm_entropy",
            "ans_length_norm_entropy",
            "predictive_entropy",
            "ans_entropy",
            "vu",
            "prob",
            "attn_probs_mean",
            "sims_probs_mean",
            "attn_probs_prod",
            "sims_probs_prod",
            "ens_marginal",
        ]
        df = df.reindex(desired_order)

        df = df.rename(
            index={
                "ans_probs_prod": r"$P_{\gM}(\xans|\xcot, \xinstr)$",
                "ans_probs_mean": r"$\widebar{P}_{\gM}(\xans)$",
                "attn_probs_prod": r"$\widetilde{P}_{\gM, \text{attn}}$",
                "attn_probs_mean": r"$\widebar{P}_{\gM, \text{attn}}$",
                "sims_probs_prod": r"$\widetilde{P}_{\gM, \text{sim}}$",
                "sims_probs_mean": r"$\widebar{P}_{\gM, \text{sim}}$",
                "pred_probs_prod": r"$P_{\gM}(\xresp|\xinstr)$",
                "pred_probs_mean": r"$\widebar{P}_{\gM}(\xresp)$",
                "predictive_entropy": r"$\gH$",
                "length_norm_entropy": r"$\widebar{\gH}$",
                "ans_entropy": r"$\gH(\xans)$",
                "ans_length_norm_entropy": r"$\widebar{\gH}(\xans)$",
                "ens_marginal": r"$\widetilde{P}_{\gM}$",
                "vu": "Verbalized Uncertainty",
                "prob": "Self-Consistency",
            }
        )

        latex = csv_to_latex_table(
            df,
            label=f"tabapp:result-details-{dataset}-{model_name.lower()}",
            caption=f"{model_name} results on the {dataset_name_mapping[dataset]} dataset.",
            decimal_places=2,
        )
        output_path = osp.join(latex_out_dir, f"t8-result-{dataset}-{idx}-{model_name.lower()}.tex")
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)


if __name__ == "__main__":
    main()
