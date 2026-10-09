"""Script for plotting accuracy vs cumulative time as a 3-panel scatter plot.

All panels are read from a single output directory (``-od/--output_dir``); each
panel plots one longbench-e data_dir (and its mapped press):
1. hotpotqa_e / pyramidkv
2. triviaqa_e / streaming_llm
3. trec_e / snapkv

The figure is written to ``<output_dir>/figures/``.

Example usage:
```bash
python format_accuracy_vs_time_3panel.py -od out-0611 --model meta-llama/Meta-Llama-3.1-8B-Instruct
```
"""

import logging
import os

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_ALPHA,
    LONGBENCH_E_TASK_ALIASES,
    METHOD_ALIASES,
    OUTWARD,
    TICK_FONT_SIZE,
    NEURIPS_HEIGHT,
    NEURIPS_WIDTH,
    get_base_parser,
    get_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Configuration ──────────────────────────────────────────────────────────────
PANELS = [
    {"data_dir": "trec_e"},
    {"data_dir": "triviaqa_e"},
    {"data_dir": "hotpotqa_e"},
]
DATASET = "longbench-e"
ID = "_id"

# Legend mapping: h value → suffix for press-specific labels
H_SUFFIXES = {
    None: "",
    9: "-Express",
}
H_COLORS = {
    None: "gray",
    9: "forestgreen",
}
H_LINESTYLES = {
    None: "--",
    9: "-",
}


# ── Helper functions ───────────────────────────────────────────────────────────
def get_compression_time(row):
    if row["compression_times"] is None:
        return 0
    return np.sum(
        [np.sum(layer_times) for layer_times in row["compression_times"].values() if layer_times is not None]
    )


def get_prefill_context_time(row):
    return np.sum([np.sum(times) for times in row["prefill_context_times"].values()])


def get_prefill_question_time(row):
    return np.sum([np.sum(times) for times in row["prefill_question_times"].values()])


def load_and_prepare(model, output_dir, data_dir):
    df = get_data(model, output_dir, dataset=DATASET, data_dir=data_dir)
    df["mean_cache_size"] = df["cache_sizes_tracking"].apply(lambda x: np.mean([v[0] for v in x.values()]))
    if data_dir == "triviaqa_e":
        max_target = 2**13 + 2 * 32
        df = df[(df["target_size"] <= max_target) | (df["press_name"] == "no_press")]

    df["query_time"] = df["query_times"].apply(lambda x: sum([sum(lt) for lt in x.values()]))
    df["compression_time"] = df.apply(get_compression_time, axis=1)
    df["decoding_time"] = df["query_time"] + df["compression_time"]
    df["prefill_context_time"] = df.apply(get_prefill_context_time, axis=1)
    df["prefill_question_time"] = df.apply(get_prefill_question_time, axis=1)
    df["cumulative_time"] = df["prefill_context_time"] + df["prefill_question_time"] + df["decoding_time"]

    df = (
        df.groupby([ID, "press_name", "target_size", "h"], dropna=False)
        .agg(
            {
                "cumulative_time": "mean",
                "all": "mean",
                "mean_cache_size": "mean",
            }
        )
        .reset_index()
    )
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
parser = get_base_parser()
parser.set_defaults(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
args = parser.parse_args()
model = args.model
output_dir = args.output_dir

# Use NeurIPS page width
fig, axs = plt.subplots(1, len(PANELS), figsize=(NEURIPS_WIDTH, NEURIPS_HEIGHT))

for panel_idx, panel_cfg in enumerate(PANELS):
    ax = axs[panel_idx]
    data_dir = panel_cfg["data_dir"]

    df = load_and_prepare(model, output_dir, data_dir)

    # Infer press_name from data (exclude no_press)
    press_names = [p for p in df["press_name"].unique() if p != "no_press"]
    press_name_display = METHOD_ALIASES.get(press_names[0], press_names[0]) if press_names else ""
    task_display = LONGBENCH_E_TASK_ALIASES.get(data_dir, data_dir)

    panel_legend_handles = []
    groups = list(df.groupby(["press_name", "h"], dropna=False))
    # Sort so that h=None (normal press) comes before h!=None (Express) per press
    groups.sort(key=lambda item: (item[0][0], 0 if pd.isna(item[0][1]) else 1))
    for (pname, h), group in groups:
        h_key = h if pd.notna(h) else None
        color = H_COLORS.get(h_key, "black")
        linestyle = H_LINESTYLES.get(h_key, "-")

        if pname == "no_press":
            continue

        # Aggregate per target_size
        x, y, yerr, cache_sizes, target_sizes = [], [], [], [], []
        for target_size in sorted(group["target_size"].unique()):
            df_t = group[group["target_size"] == target_size]
            x.append(df_t["cumulative_time"].mean())
            y.append(df_t["all"].mean())
            yerr.append(df_t["all"].std(ddof=1) / np.sqrt(len(df_t)))
            cache_sizes.append(df_t["mean_cache_size"].mean())
            target_sizes.append(target_size)

        x = np.array(x)
        y = np.array(y)
        yerr = np.array(yerr)
        cache_sizes = np.array(cache_sizes)
        target_sizes = np.array(target_sizes)

        # Sort by target_size (already sorted, but be explicit)
        sort_idx = np.argsort(target_sizes)
        x, y, yerr = x[sort_idx], y[sort_idx], yerr[sort_idx]
        cache_sizes = cache_sizes[sort_idx]
        target_sizes = target_sizes[sort_idx]

        # Fill between (polygon to avoid criss-crossing)
        poly_x = np.concatenate([x, x[::-1]])
        poly_y = np.concatenate([y + yerr, (y - yerr)[::-1]])
        ax.fill(poly_x, poly_y, color=color, alpha=0.1)

        # Line
        ax.plot(x, y, color=color, alpha=LINE_ALPHA, linewidth=1, linestyle=linestyle, zorder=2)

        # Scatter (equally sized dots)
        ax.scatter(x, y, color=color, marker="o", s=20, alpha=LINE_ALPHA * 0.6)

        # Per-panel legend entry with press-specific label
        h_key = h if pd.notna(h) else None
        suffix = H_SUFFIXES.get(h_key, "")
        legend_label = f"{press_name_display}{suffix}"
        panel_legend_handles.append(
            lines.Line2D(
                [0], [0], color=color, label=legend_label, linewidth=1, linestyle=linestyle, marker="o", markersize=4
            )
        )

    # Axes formatting
    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))
    ax.set_xlabel("Runtime (ms)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    if panel_idx == 0:
        ax.set_ylabel("Accuracy", fontsize=LABEL_FONT_SIZE)
    ax.set_title(f"{task_display}", fontsize=LABEL_FONT_SIZE)
    ax.legend(
        handles=panel_legend_handles,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=1,
    )

plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25, wspace=0.3)

figure_dir = os.path.join(output_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
fig_path = os.path.join(figure_dir, f"scatter_accuracy_vs_time_3panel-{model.replace('/', '--')}.pdf")
plt.savefig(fig_path, bbox_inches="tight", dpi=300)
print(f"Saved to {os.path.abspath(fig_path)}")
