import os
import os.path as osp
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
import spacy
from spacy import displacy
from spacy.tokens import Span

from seqlbtoolkit.data import respan
from seqlbtoolkit.io import ProgressBar

from .support import tokenize

logger = logging.getLogger(__name__)


def print_token_probs(
    txt_tks_list: list,
    tk_pred_probs: list,
    highlight_ids: list = None,
    token_field_width: int = 30,
    printing_func=print,
):
    """
    Print each token along with its predicted probability in aligned columns.
    Special characters like newlines and tabs are displayed as the literal sequences '\n' and '\t'.

    Parameters
    ----------
    txt_tks_list : list
        A list of text tokens.
    tk_pred_probs : list
        A list of predicted probabilities corresponding to tokens after they appear.
        Typically this should have one fewer element than txt_tks_list.
    highlight_ids : list, optional
        Indices of tokens that should be highlighted. Default is None, meaning no highlights.
    """
    assert isinstance(txt_tks_list, list), "txt_tks_list must be a list."

    if highlight_ids is None:
        highlight_ids = []

    # Prepend a 0.0 probability to align with tokens
    aligned_probs = [0.0] + tk_pred_probs

    # Format specifications
    # Adjust these field widths as needed.
    token_field_width = 30
    prob_field_width = 10
    prob_format = f"{{:>{prob_field_width}.4f}}"  # Right-align probability with 4 decimal places

    for i, (tk, prob) in enumerate(zip(txt_tks_list, aligned_probs)):
        # Replace special characters with their literal escape sequences
        display_token = tk.replace("\n", "<new-line>").replace("\t", "<tab>").replace("\r", "<carriage-return>")

        # Highlight tokens if needed
        if i in highlight_ids:
            display_token = f"{display_token}"

        # Print token and probability in aligned columns
        # Left-align the token within the specified width
        # Follow with a space, then the right-aligned probability
        printing_func(f"{display_token:<{token_field_width}}{prob_format.format(prob)}")

    return None


def render_text_info(
    nlp: spacy.language.Language,
    tokenizer,
    text: str,
    info_spans: dict[tuple[int, int], str] | list[tuple[int, int]],
    jupyter: bool = None,
):
    pred_tks = np.asarray(tokenize(tokenizer, text))

    nlp = spacy.blank("en")
    doc = nlp(text)
    spacy_tks = [tk.text for tk in doc]

    spacy_tk_spans = respan(pred_tks, spacy_tks, info_spans)

    if isinstance(spacy_tk_spans, dict):
        doc.spans["sc"] = [Span(doc, start, end, str(value)) for (start, end), value in spacy_tk_spans.items()]
    elif isinstance(spacy_tk_spans, list):
        doc.spans["sc"] = [Span(doc, start, end, "") for start, end in spacy_tk_spans]
    else:
        raise ValueError("info_spans must be a dictionary or list.")

    rts = displacy.render(doc, style="span", jupyter=jupyter)

    return rts


def generate_html_visualizations(token_attributes, tokenizer, save_dir: str):
    """
    Generate HTML visualizations of the attention/probabilities and save them.

    Args:
        results_dict (dict): A dictionary mapping IDs to probability calculation results.
        df (pd.DataFrame): Merged DataFrame containing references to predicted text.
        tokenizer: The HuggingFace tokenizer used to tokenize data.
        save_root (str): Root directory where HTML files will be saved.

    Returns:
        None
    """
    logger.info(f"Generating HTML visualizations in {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    nlp = spacy.blank("en")

    pbar = ProgressBar(transient=True, total=len(token_attributes), desc="Saving instances")
    with pbar:
        for inst_idx, tk_ids, _, top_probs, src_tk_ids, attn_scores, tk_sims in token_attributes:

            attended_tk_ids = np.asarray(src_tk_ids) + 1
            pred_text = tokenizer.decode(tk_ids, skip_special_tokens=False)
            attended_tk_probs = top_probs[src_tk_ids, 0]

            attended_tk_spans = {(i, i + 1): int(p * 100) for (i, p) in zip(attended_tk_ids, attended_tk_probs)}
            prob_markup = render_text_info(
                nlp=nlp, tokenizer=tokenizer, text=pred_text, info_spans=attended_tk_spans, jupyter=False
            )

            attended_tk_spans = {(i, i + 1): int(p * 100) for (i, p) in zip(attended_tk_ids, attn_scores)}
            scores_markup = render_text_info(
                nlp=nlp, tokenizer=tokenizer, text=pred_text, info_spans=attended_tk_spans, jupyter=False
            )

            attended_tk_spans = {(i, i + 1): int(s * 100) for (i, s) in zip(attended_tk_ids, tk_sims)}
            sims_markup = render_text_info(
                nlp=nlp, tokenizer=tokenizer, text=pred_text, info_spans=attended_tk_spans, jupyter=False
            )

            html_markup = (
                f"<h2>Probabilities</h2>{prob_markup}\n\n\n"
                f"<h2>Attention Scores</h2>{scores_markup}\n\n\n"
                f"<h2>Similarities</h2>{sims_markup}"
            )

            # Save HTML
            with open(osp.join(save_dir, f"{inst_idx}.html"), "w") as f:
                f.write(html_markup)

            pbar.update()


def plot_calibration_data_distr(correct_probs: list, incorrect_probs: list, n_bins: int = 20):
    """
    Manually compute a calibration curve with uniform binning, skipping empty bins.
    Then plot the resulting curve alongside bars for each non-empty bin,
    with each bar centered on that bin's mean predicted value, and
    a dynamically determined width that prevents overlapping.
    Finally, enforce that the y-axis range for the curve plot and the bar plot are the same.
    """
    # 1. Prepare data
    y_true = np.array([1] * len(correct_probs) + [0] * len(incorrect_probs))
    y_prob = np.array(correct_probs + incorrect_probs)
    total_samples = len(y_prob)

    # 2. Sort out bin edges: n_bins uniform bins over [0,1]
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Arrays to store computations per bin
    bin_counts = np.zeros(n_bins, dtype=int)  # how many samples in each bin
    sum_of_probs = np.zeros(n_bins, dtype=float)  # sum of predicted probabilities
    sum_of_positives = np.zeros(n_bins, dtype=int)  # how many positives in each bin

    # 3. Assign each sample to a bin (digitize returns [1..n_bins], subtract 1 -> [0..n_bins-1])
    indices = np.digitize(y_prob, bin_edges) - 1

    # 4. Accumulate stats per bin
    for i in range(total_samples):
        bin_idx = indices[i]
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bin_counts[bin_idx] += 1
        sum_of_probs[bin_idx] += y_prob[i]
        sum_of_positives[bin_idx] += y_true[i]

    # 5. Compute fraction_of_positives and mean_predicted_value for non-empty bins only
    fraction_of_positives_list = []
    mean_predicted_value_list = []
    bin_portions_list = []

    for i in range(n_bins):
        if bin_counts[i] > 0:
            fraction_of_positives_list.append(sum_of_positives[i] / bin_counts[i])
            mean_predicted_value_list.append(sum_of_probs[i] / bin_counts[i])
            bin_portions_list.append(bin_counts[i] / total_samples)

    fraction_of_positives = np.array(fraction_of_positives_list)
    mean_predicted_value = np.array(mean_predicted_value_list)
    bin_portions = np.array(bin_portions_list)

    # 6. Create figure and main axis (for the calibration curve)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Calibration Plot and Data Distribution")

    # 7. Perfectly calibrated line
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    # 8. Plot the calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="blue", label="Calibration Curve", zorder=5)

    # 9. Create a twin axis for the bar plot
    ax2 = ax.twinx()

    # 10. Dynamically compute each bar's width to avoid overlap
    if len(mean_predicted_value) <= 1:
        # If only one bin or none are non-empty, pick a default bar width
        bar_widths = [0.1 for _ in mean_predicted_value]
    else:
        bar_widths = []
        for i in range(len(mean_predicted_value)):
            if i == 0:
                gap_left = mean_predicted_value[1] - mean_predicted_value[0]
                gap_right = gap_left
            elif i == len(mean_predicted_value) - 1:
                gap_left = mean_predicted_value[i] - mean_predicted_value[i - 1]
                gap_right = gap_left
            else:
                gap_left = mean_predicted_value[i] - mean_predicted_value[i - 1]
                gap_right = mean_predicted_value[i + 1] - mean_predicted_value[i]

            width = 0.8 * (gap_left + gap_right) / 2
            width = max(width, 0.01)  # a small fallback if gaps are extremely small
            bar_widths.append(width)

    # 11. Plot bars for each non-empty bin
    for i in range(len(fraction_of_positives)):
        x_center = mean_predicted_value[i]
        width = bar_widths[i]
        ax2.bar(
            x_center,
            bin_portions[i],
            width=width,
            alpha=0.3,
            color="gray",
            edgecolor="k",
            align="center",
            label="Prediction Distribution" if i == 0 else None,
        )

    # 12. Set axis labels
    ax.set_xlabel("Mean Predicted Value")
    ax.set_ylabel("Fraction of Positives", color="blue")
    ax2.set_ylabel("Portion of Samples", color="gray")

    # Match tick label colors
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="gray")

    # 13. Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # 14. Option A: Align y-axis labels automatically (Matplotlib >= 3.4)
    fig.align_ylabels([ax, ax2])

    # 15. Enforce the same y-range on both axes
    # First, get the min/max from each axis
    y_min_curve, y_max_curve = ax.get_ylim()
    y_min_bars, y_max_bars = ax2.get_ylim()

    # Compute the combined extremes
    shared_y_min = min(y_min_curve, y_min_bars)
    shared_y_max = max(y_max_curve, y_max_bars)

    # Apply them to both axes
    ax.set_ylim(shared_y_min, shared_y_max)
    ax2.set_ylim(shared_y_min, shared_y_max)

    # 16. Show grid on main axis
    ax.grid(True)
    plt.show()


def plot_calibration_data_distr_multi_runs(
    correct_probs_list,
    incorrect_probs_list,
    n_bins: int = 20,
    figure_path: str = None,
    disable_legend: bool = False,
):
    """
    Plots a calibration curve (mean ± std across multiple runs) on the primary y-axis,
    and on a twin y-axis shows bin-proportions as thin bars (plus a polygon with hatching)
    that connects the bar tops (removing 'white gaps' horizontally, without solid fill).
    """

    plt.rcParams["font.family"] = "Times New Roman"

    # Number of runs
    n_runs = len(correct_probs_list)
    assert n_runs == len(incorrect_probs_list), "correct_probs_list and incorrect_probs_list must have the same length."

    # -- 1. Prepare storage for per-run results (n_runs x n_bins) --
    fraction_of_positives_all = np.full((n_runs, n_bins), np.nan)
    mean_predicted_value_all = np.full((n_runs, n_bins), np.nan)
    bin_portions_all = np.full((n_runs, n_bins), np.nan)

    # Fixed bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # -- 2. Per-run calculations --
    for r in range(n_runs):
        correct_probs = correct_probs_list[r]
        incorrect_probs = incorrect_probs_list[r]

        y_true = np.array([1] * len(correct_probs) + [0] * len(incorrect_probs))
        y_prob = np.array(correct_probs + incorrect_probs)
        total_samples = len(y_prob)

        bin_counts = np.zeros(n_bins, dtype=int)
        sum_of_probs = np.zeros(n_bins, dtype=float)
        sum_of_positives = np.zeros(n_bins, dtype=int)

        # Digitize
        indices = np.digitize(y_prob, bin_edges) - 1
        for i in range(total_samples):
            bin_idx = indices[i]
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            bin_counts[bin_idx] += 1
            sum_of_probs[bin_idx] += y_prob[i]
            sum_of_positives[bin_idx] += y_true[i]

        # Bin stats
        for i in range(n_bins):
            if bin_counts[i] > 0:
                fraction_of_positives_all[r, i] = sum_of_positives[i] / bin_counts[i]
                mean_predicted_value_all[r, i] = sum_of_probs[i] / bin_counts[i]
                bin_portions_all[r, i] = bin_counts[i] / total_samples

    # -- 3. Mean ± std across runs --
    fraction_of_positives_mean = np.nanmean(fraction_of_positives_all, axis=0)
    fraction_of_positives_std = np.nanstd(fraction_of_positives_all, axis=0)

    mean_predicted_value_mean = np.nanmean(mean_predicted_value_all, axis=0)
    mean_predicted_value_std = np.nanstd(mean_predicted_value_all, axis=0)

    bin_portions_mean = np.nanmean(bin_portions_all, axis=0)

    # -- 4. Keep only valid bins (non-empty in at least one run) --
    valid_bins = ~np.isnan(fraction_of_positives_mean)
    fraction_of_positives_mean = fraction_of_positives_mean[valid_bins]
    fraction_of_positives_std = fraction_of_positives_std[valid_bins]
    mean_predicted_value_mean = mean_predicted_value_mean[valid_bins]
    mean_predicted_value_std = mean_predicted_value_std[valid_bins]
    bin_portions_mean = bin_portions_mean[valid_bins]

    # -- 5. Setup figure --
    fig, ax = plt.subplots(figsize=(8, 6))

    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28
    marker_size = 6

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k:")

    # Calibration curve
    ax.plot(
        mean_predicted_value_mean,
        fraction_of_positives_mean,
        "o-",
        label="Calibration Plot",
        color="blue",
        markersize=marker_size,
        zorder=100,
    )
    # Fill ± std
    ax.fill_between(
        mean_predicted_value_mean,
        fraction_of_positives_mean - fraction_of_positives_std,
        fraction_of_positives_mean + fraction_of_positives_std,
        alpha=0.2,
        color="blue",
        zorder=50,
    )

    ax2 = ax.twinx()

    x_vals = mean_predicted_value_mean
    y_vals = bin_portions_mean
    bar_width = 0.03  # Thin bars

    verts = []
    n = len(x_vals)
    if n > 0:
        # Start at bottom-left corner
        verts.append((x_vals[0] - bar_width / 2, 0))
        # Go up to top-left of bar 0
        verts.append((x_vals[0] - bar_width / 2, y_vals[0]))
        # For each bar i from 0..n-1
        for i in range(n):
            # top-right corner of bar i
            verts.append((x_vals[i] + bar_width / 2, y_vals[i]))
            # if not the last bar, diagonal to next bar's top-left
            if i < n - 1:
                verts.append((x_vals[i + 1] - bar_width / 2, y_vals[i + 1]))
        # Finally down to bottom-right of last bar
        verts.append((x_vals[-1] + bar_width / 2, 0))

    # Construct the polygon
    poly = Polygon(
        verts,
        closed=True,
        facecolor="gray",  # solid fill
        edgecolor="gray",  # polygon outline
        linewidth=1,
        zorder=1,
        alpha=0.6,
        label="Probability Proportion",
    )
    ax2.add_patch(poly)

    # -- 7. Force axes to [0,1] --
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # -- 8. Remove spines, remove "0.0" ticks --
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    main_yticks = ax.get_yticks()
    ax.set_yticks([yt for yt in main_yticks if yt != 0.0])
    twin_yticks = ax2.get_yticks()
    ax2.set_yticks([yt for yt in twin_yticks if yt != 0.0])

    # Labels, ticks, legends
    ax.set_xlabel("Estimated Answer Probability", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize, labelcolor="blue")
    ax2.tick_params(axis="y", labelsize=tick_fontsize, labelcolor="gray")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if not disable_legend:
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=legend_fontsize)

    fig.align_ylabels([ax, ax2])
    ax.grid(True)

    # Tight layout + optional save
    fig.tight_layout()
    os.makedirs(osp.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    return None


def plot_length_distribution(
    l_resp_correct,
    l_resp_incorrect,
    l_attn_correct,
    l_attn_incorrect,
    disable_legend=False,
    figure_path=None,
):
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28

    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_hist(lengths, label, color, fill=False, linestyle="-"):
        counts, bins = np.histogram(lengths, bins=50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plot the outline of the histogram
        if linestyle is not None:
            ax.plot(bin_centers, counts, label=label, color=color, linewidth=2, linestyle=linestyle)

        # Optionally fill the area under the histogram
        if fill:
            ax.fill_between(bin_centers, counts, color=color, alpha=0.3, label=label)

    plot_hist(l_attn_correct, label=r"$L_{\text{attn}}$-Correct", color="blue", fill=False, linestyle="--")
    plot_hist(l_attn_incorrect, label=r"$L_{\text{attn}}$-Incorrect", color="gray", fill=False, linestyle="--")
    plot_hist(l_resp_correct, label=r"$L_{\text{resp}}$-Correct", color="blue", fill=True, linestyle=None)
    plot_hist(l_resp_incorrect, label=r"$L_{\text{resp}}$-Inorrect", color="gray", fill=True, linestyle=None)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a grid
    ax.grid(True, alpha=0.6)

    ax.set_xlabel("Number of Tokens", fontsize=label_fontsize)
    ax.set_ylabel("Count of Samples", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if not disable_legend:
        ax.legend(loc="best", fontsize=legend_fontsize)

    plt.tight_layout()
    os.makedirs(osp.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    return None


def plot_ece_accuracy_correlation(ece_dict, accuracy_dict, figure_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28
    marker_size = 300

    fig, ax = plt.subplots(figsize=(8, 6))

    ece_list = ece_dict["gsm8k"]
    accuracy_list = accuracy_dict["gsm8k"]
    ax.scatter(ece_list, accuracy_list, color="gray", label="GSM8k", marker="o", s=marker_size, alpha=0.4, zorder=10)

    ece_list = ece_dict["math"]
    accuracy_list = accuracy_dict["math"]
    ax.scatter(ece_list, accuracy_list, color="blue", label="MATH", marker="^", s=marker_size, alpha=0.6, zorder=10)

    ece_list = ece_dict["bbh"]
    accuracy_list = accuracy_dict["bbh"]
    ax.scatter(ece_list, accuracy_list, color="black", label="BBH", marker="s", s=marker_size, alpha=0.6, zorder=10)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a grid
    ax.grid(True, alpha=0.4, zorder=0)

    ax.set_xlabel("ECE", fontsize=label_fontsize)
    ax.set_ylabel("Accuracy", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.legend(loc="best", fontsize=legend_fontsize)

    plt.tight_layout()
    os.makedirs(osp.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    return None


def plot_ece_auc_correlation(ece_dict, auc_dict, figure_path=None, disable_legend=False):
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28
    marker_size = 300

    fig, ax = plt.subplots(figsize=(8, 6))

    ece_list = ece_dict["gsm8k"]
    auc_list = auc_dict["gsm8k"]
    ax.scatter(ece_list, auc_list, color="gray", label="GSM8k", marker="o", s=marker_size, alpha=0.4, zorder=10)

    ece_list = ece_dict["math"]
    auc_list = auc_dict["math"]
    ax.scatter(ece_list, auc_list, color="blue", label="MATH", marker="^", s=marker_size, alpha=0.6, zorder=10)

    ece_list = ece_dict["bbh"]
    auc_list = auc_dict["bbh"]
    ax.scatter(ece_list, auc_list, color="black", label="BBH", marker="s", s=marker_size, alpha=0.6, zorder=10)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a grid
    ax.grid(True, alpha=0.4, zorder=0)

    ax.set_xlabel("ECE", fontsize=label_fontsize)
    ax.set_ylabel("AUROC", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if not disable_legend:
        ax.legend(loc="best", fontsize=legend_fontsize)

    plt.tight_layout()
    os.makedirs(osp.dirname(figure_path), exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    return None


def plot_hyper_correlation(ece_list, auc_list, figure_path=None, disable_legend=False):
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    marker_size = 200

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare a set of markers to cycle through
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "X", "D", "h"]

    # Create an array of "shades of blue"
    # For instance, 0.2 to 0.8 in 10 steps (if there are 10 points).
    # Adjust the start/end to get the range of shades you desire.
    n_points = len(ece_list)
    shades_of_blue = cm.Blues(np.linspace(0.3, 1, n_points))

    for i, (ece, auc) in enumerate(zip(ece_list, auc_list)):
        # Pick a color and marker for this point
        color = shades_of_blue[i]
        marker = markers[i % len(markers)]

        # Plot each point with a unique marker and shade
        ax.scatter(ece, auc, color=color, marker=marker, s=marker_size, alpha=0.8, zorder=10)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a grid
    ax.grid(True, alpha=0.4, zorder=0)

    # Labeling
    ax.set_xlabel("ECE", fontsize=label_fontsize)
    ax.set_ylabel("AUROC", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    plt.tight_layout()

    if figure_path is not None:
        os.makedirs(osp.dirname(figure_path), exist_ok=True)
        fig.savefig(figure_path, bbox_inches="tight")

    return None


def plot_sims(df, figure_path=None):
    """
    Plots grouped horizontal bars for 'resp_sim' and 'attn_sim' columns from a given DataFrame.
    The index of the DataFrame is assumed to be the model names.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing 'resp_sim' and 'attn_sim' columns and model names as the index.

    Returns:
    --------
    None
        Displays the plot.
    """

    # Set global font parameters
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate positions for each model on the y-axis
    y_positions = np.arange(len(df))
    bar_height = 0.4

    # Plot the horizontal bars
    ax.barh(
        y_positions - bar_height / 2,
        df["resp_sim"],
        height=bar_height,
        label=r"$\text{sim}(\mathbf{x}_\text{ans},\mathbf{x}_\text{resp})$",
        color="gray",
        alpha=0.5,
    )
    ax.barh(
        y_positions + bar_height / 2,
        df["attn_sim"],
        height=bar_height,
        label=r"$\text{sim}(\mathbf{x}_\text{ans},\mathbf{x}_\text{attn})$",
        color="blue",
        alpha=0.5,
    )

    # Set y-axis ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df.index, fontsize=tick_fontsize)
    # Invert the y-axis so the first label is on top
    ax.invert_yaxis()

    # Set x-axis tick label size
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    # Label the axes
    ax.set_xlabel("Similarity", fontsize=label_fontsize)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Add vertical grid lines (grid lines drawn along the y-axis)
    ax.grid(visible=True, axis="x", which="major", linestyle="-", color="gray", alpha=0.3)

    # Add legend
    # ax.legend(fontsize=legend_fontsize, loc="lower right")
    plt.tight_layout()

    if figure_path is not None:
        os.makedirs(osp.dirname(figure_path), exist_ok=True)
        fig.savefig(figure_path, bbox_inches="tight")

    return None


def plot_simsonly(df, figure_path=None):
    """
    Plots grouped vertical bars for 'AUC' and 'ECE' columns from a given DataFrame.
    The index of the DataFrame is assumed to be the model names.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing 'AUC' and 'ECE' columns and model names as the index.

    figure_path : str or None, optional
        If provided, saves the figure to this file path.

    Returns:
    --------
    None
        Displays (and optionally saves) the plot.
    """

    # Set global font parameters
    plt.rcParams["font.family"] = "Times New Roman"
    label_fontsize = 36
    tick_fontsize = 32
    legend_fontsize = 28

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate positions for each model on the x-axis
    x_positions = np.arange(len(df))
    bar_width = 0.4

    # Plot the vertical bars
    ax.bar(
        x_positions - bar_width / 2,
        df["AUC"],
        width=bar_width,
        label="AUC",
        alpha=0.5,
    )
    ax.bar(
        x_positions + bar_width / 2,
        df["ECE"],
        width=bar_width,
        label="ECE",
        alpha=0.5,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df.index, fontsize=tick_fontsize, rotation=45, ha="right")

    # Label the axes
    ax.set_ylabel("Value", fontsize=label_fontsize)

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optionally add horizontal grid lines
    ax.grid(visible=True, axis="y", which="major", linestyle="-", color="gray", alpha=0.3)

    # Add legend
    ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()

    # Optionally save the figure
    if figure_path is not None:
        os.makedirs(osp.dirname(figure_path), exist_ok=True)
        fig.savefig(figure_path, bbox_inches="tight")

    return None
