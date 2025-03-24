import os.path as osp
import pandas as pd
import logging

from seqlbtoolkit.io import set_logging


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
    latex_str += "    \\begin{tabular}{lcc}\n"
    latex_str += "    \\toprule\n"
    latex_str += "    Method & AUC & ECE \\\\ \n"
    latex_str += "    \\midrule\n"

    # Construct each row of the table
    for method, row in df.iterrows():
        auc_mean = f"{row['AUC-Mean']:.{decimal_places}f}"
        auc_std = f"{row['AUC-Std']:.{decimal_places}f}"
        ece_mean = f"{row['ECE-Mean']:.{decimal_places}f}"
        ece_std = f"{row['ECE-Std']:.{decimal_places}f}"

        latex_str += f"    {method} & \\num{{{auc_mean}}}$\\pm$\\num{{{auc_std}}} & \\num{{{ece_mean}}}$\\pm$\\num{{{ece_std}}} \\\\ \n"

    latex_str += "    \\bottomrule\n"
    latex_str += "    \\end{tabular}\n"
    latex_str += "\\end{table}\n"

    return latex_str


logger = logging.getLogger(__name__)
set_logging(level="INFO")

output_dir = "./output"
metric_file_name = "metrics-vu.csv"
dataset = "bbh"

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

dfs = list()
for model_name in model_names:
    path = osp.join(output_dir, dataset, model_name, metric_file_name)
    dfs.append(pd.read_csv(path, index_col=0))

df_mean = sum(dfs) / len(dfs)
df_mean *= 100

# Now df_mean has the mean of each cell.
print("Combined Mean DataFrame:")
print(df_mean)

latex = csv_to_latex_table(df_mean, label="tab:results", caption="Results table", decimal_places=1)
with open("table.tex", "w") as f:
    f.write(latex)
