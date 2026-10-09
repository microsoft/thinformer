"""Script for plotting runtime vs context length for causal attention.

Example usage:
```bash
python format_single_attention_causal.py -op PATH_TO_OUTPUT_DIR
python format_single_attention_causal.py -op PATH_TO_OUTPUT_DIR --speedup
```
"""

import os
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util_experiments import get_base_parser
from plotting_utils import (
    XLABEL_FONT_SIZE,
    YLABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_ALPHA,
    LINE_WIDTH,
    MARKER_SIZE,
    OUTWARD,
    NEURIPS_WIDTH,
    NEURIPS_HEIGHT,
    get_method_style,
)
from tabulate import tabulate

parser = get_base_parser()
parser.add_argument("--use_triton_testing", action="store_true")
parser.add_argument(
    "--speedup", action="store_true", help="Plot speedup over exact instead of runtime"
)
parser.add_argument(
    "--in_network",
    action="store_true",
    help="time the attention layer in the chatglm2-6b-32k model",
)
args, opt = parser.parse_known_args()
batch_size = args.batch_size
output_path = args.output_path
num_runs = args.num_runs
use_triton_testing = args.use_triton_testing
speedup = args.speedup
mode = args.mode
head_size = args.head_size
dim = args.dim
seed = args.seed
in_network = args.in_network
model_name = args.model_name
causal = bool(args.causal)

figures_dir = os.path.join(output_path, "figures")
os.makedirs(figures_dir, exist_ok=True)

# %%
# load files from output/times/
methods = [
    "exact",
    "hyper-cuda",
    "thinformer_triton",
    "thinformer",
]

df_list = []
for method in methods:
    files = glob(
        os.path.join(
            output_path,
            "times",
            f"times-n{num_runs}-{method}-{mode}-bs{batch_size}-h{head_size}-d{dim}-s{seed}-h*-{in_network}-causal{causal}*.csv",
        )
    )
    if len(files) == 0:
        print(f"No files found for {method} on {mode} with batch size {batch_size}")
        continue
    else:
        print(
            f"Found {len(files)} files for {method} on {mode} with batch size {batch_size}"
        )
    for file in files:
        print(f"Reading from {file}...")
        # read csv file using pandas
        df = pd.read_csv(file, index_col=0)
        grouped = df.groupby(
            ["method", "mode", "seq_len", "h", "in_network"], dropna=False
        )
        attn_quantiles = grouped[["1_attn"]].quantile([0.25, 0.5, 0.75]).unstack()
        df_list.append(attn_quantiles.reset_index())
df = pd.concat(df_list, axis=0).reset_index()
if not in_network:
    df = df[(df["seq_len"] >= 2**15)]  # start from 32k for out-of-network timings
# Calculate speedup if requested
if speedup:
    # Check that exact method exists
    exact_df = df[df["method"] == "exact"]
    if exact_df.empty:
        raise ValueError("--speedup flag requires 'exact' method data to be available")

    # For each non-exact method, compute speedup relative to exact
    speedup_list = []
    for (method_name, h, in_network, seq_len), group in df.groupby(
        ["method", "h", "in_network", "seq_len"], dropna=False
    ):
        if method_name == "exact":
            continue

        # Find corresponding exact row
        exact_row = exact_df[
            (exact_df["seq_len"] == seq_len) & (exact_df["in_network"] == in_network)
        ]

        if exact_row.empty:
            print(
                f"Warning: No exact baseline found for {method_name}, seq_len={seq_len}, in_network={in_network}"
            )
            continue

        # Calculate speedup: exact_time / method_time
        speedup_row = group.copy()
        for quantile in [0.25, 0.5, 0.75]:
            exact_time = exact_row[("1_attn", quantile)].values[0]
            method_time = group[("1_attn", quantile)].values[0]
            speedup_row[("1_attn", quantile)] = exact_time / method_time

        speedup_list.append(speedup_row)

    df = pd.concat(speedup_list, axis=0).reset_index(drop=True)
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
# %%
colors = {}
# Determine y-axis label based on speedup flag
ylabel = "Speed-up over FlashAttention 2" if speedup else "Median Runtime (ms)"

for use_log_scale in [True, False]:
    # Create a plot
    fig = plt.figure(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
    ax = plt.gca()

    # Group by 'method' and 'h' and plot each group.
    # dropna=False so methods with NaN h (exact, hyper-cuda)
    # are still plotted; with default dropna=True they were silently dropped.
    for (method_name, h, in_network), group in df.groupby(
        ["method", "h", "in_network"], dropna=False
    ):
        style = get_method_style(method_name, h, causal)
        label, color, linestyle = style["label"], style["color"], style["linestyle"]

        p = ax.plot(
            group["seq_len"],
            group[("1_attn", 0.5)],
            marker=style["marker"],
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            label=label,
            color=color,
            alpha=LINE_ALPHA,
            ms=MARKER_SIZE,
        )

        if (method_name, h) not in colors:
            colors[(method_name, h)] = p[0].get_color()
            ax.fill_between(
                group["seq_len"],
                group[("1_attn", 0.25)],
                group[("1_attn", 0.75)],
                color=colors[(method_name, h)],
                alpha=0.2,
            )

    # Move spines outward
    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))

    # Set plot labels
    ax.set_xlabel("Sequence length $n$", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONT_SIZE)

    if use_log_scale:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    else:
        # Find the largest power of 2 in the data
        max_seq_len = df["seq_len"].max()
        min_power = int(np.ceil(np.log2(df["seq_len"].min())))
        max_power = int(np.floor(np.log2(max_seq_len)))
        powers = list(range(min_power, max_power + 1))

        # Set x ticks to powers of 2
        ax.set_xticks([2**i for i in powers])
        # Set tick labels, skipping 2k, 4k, 8k, 16k
        labels = []
        for i in powers:
            k_value = 2**i // 1024
            if in_network:
                labels.append(f"{k_value}k")
            else:
                if k_value in [2, 4, 8, 16]:
                    labels.append("")
                else:
                    labels.append(f"{k_value}k")
        ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
        # Shift leftmost non-empty label slightly left so the first two
        # xtick labels don't overlap.
        offset = matplotlib.transforms.ScaledTranslation(-0.05, 0, fig.dpi_scale_trans)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text():
                tick_label.set_transform(tick_label.get_transform() + offset)
                break
        ax.set_ylim(0, max(df[("1_attn", 0.5)]) * 1.1)

    # Set tick font size
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)

    # Add grid to plot (y-axis only)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.grid(True, linestyle=":", alpha=0.2, axis="y", which="minor")

    # Add legend at top left of the figure, ensuring Thinformer (Triton)
    # appears above Thinformer (Original).
    handles, labels = ax.get_legend_handles_labels()
    if "Thinformer (Triton)" in labels and "Thinformer (Original)" in labels:
        triton_idx = labels.index("Thinformer (Triton)")
        original_idx = labels.index("Thinformer (Original)")
        if triton_idx > original_idx:
            handles[triton_idx], handles[original_idx] = (
                handles[original_idx],
                handles[triton_idx],
            )
            labels[triton_idx], labels[original_idx] = (
                labels[original_idx],
                labels[triton_idx],
            )
    ax.legend(
        handles,
        labels,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        # frameon=False,
    )

    # Adjust layout
    plt.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.2)
    
    # Save the figure as PDF
    speedup_suffix = "-speedup" if speedup else ""
    fig_path = os.path.join(
        figures_dir,
        f"single-attention-causal-bs{batch_size}-h{head_size}-d{dim}-s{seed}-{in_network}-{'log' if use_log_scale else 'linear'}{speedup_suffix}.pdf",
    )
    plt.savefig(fig_path, dpi=300)
    print(f"Saved figure to {fig_path}")
    plt.close()
