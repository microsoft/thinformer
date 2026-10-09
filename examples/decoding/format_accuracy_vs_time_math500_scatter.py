# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: KVPress (Python 3.13)
#     language: python
#     name: kvpress
# ---

# %%
# Script to aggregate accuracy based on all intermediate cache sizes
# and plot the accuracy vs the cumulative query, compression, and decoding times.
# - Query times are stored in the "query_times" column of the predictions.parquet file as a dictionary
# with keys corresponding to layer indices and values corresponding to the query time in milliseconds 
# for each generated token.
# - Compression times are stored in the "compression_times" column of the predictions.parquet file as a dictionary
# with keys corresponding to layer indices and values corresponding to the compression time in milliseconds 
    # for each generated token.
# - Decoding times is derived from the query and compression times by summing the values for each layer.

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import MultipleLocator
import numpy as np
from tabulate import tabulate

from utils import (
    get_base_parser,
    get_data_math500,
    COLORS,
    MARKERS,
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_ALPHA,
    OUTWARD,
    METHOD_ALIASES,
    NEURIPS_WIDTH,
    NEURIPS_HEIGHT,
)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# %%
parser = get_base_parser()
args = parser.parse_args()
output_dir = args.output_dir
model = args.model
df = get_data_math500(model, output_dir)
ID = 'unique_id'
df['cum_query_time'] = df['query_times'].apply(lambda x: sum([sum(layer_times) for layer_times in x.values()]))
def get_cum_compression_time(row):
    """Wrapper function for backwards compatibility with old dataframes."""
    if row['compression_times'] is not None:
        # Skip None values in the compression_times layers
        return np.sum([np.sum(layer_times) for layer_times in row['compression_times'].values() if layer_times is not None])
    else:
        return 0
df['cum_compression_time'] = df.apply(get_cum_compression_time, axis=1)
def sum_layered_times(x):
    if x is None:
        return 0
    return np.sum([np.sum(layer_times) for layer_times in x.values() if layer_times is not None])
df['cum_prefill_context_time'] = df['prefill_context_times'].apply(sum_layered_times)
df['cum_prefill_question_time'] = df['prefill_question_times'].apply(sum_layered_times)
df['cum_decoding_time'] = (
    df['cum_query_time']
    + df['cum_compression_time']
    + df['cum_prefill_context_time']
    + df['cum_prefill_question_time']
)
# take mean across seeds
df = df.groupby([ID, 'press_name', 'target_size']).agg({
    'cum_query_time': 'mean',
    'cum_compression_time': 'mean',
    'cum_prefill_context_time': 'mean',
    'cum_prefill_question_time': 'mean',
    'cum_decoding_time': 'mean',
    'all': 'mean',
    "prompt_tokens": "mean",
    "completion_tokens": "mean",
    "mean_cache_size": "mean",
}).reset_index()

# %%
# print results as a table
mean_std = lambda x: f"{x.mean():.2f} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f}"
acc = df.groupby(["press_name", 'target_size'], dropna=False)[[
    "cum_query_time",
    "cum_compression_time",
    "cum_prefill_context_time",
    "cum_prefill_question_time",
     "cum_decoding_time",
     "all",
      "completion_tokens",
      "mean_cache_size",
]].agg(mean_std).reset_index()
acc['count'] = df.groupby(["press_name", 'target_size'], dropna=False).size().values
print(tabulate(acc, headers="keys", tablefmt="github"))

# %%
COLUMNS = [
    'cum_query_time',
    'cum_compression_time',
    'cum_decoding_time',
]
X_LABELS = {
    'cum_query_time': 'Cumulative query time (ms)',
    'cum_compression_time': 'Cumulative compression time (ms)',
    'cum_decoding_time': 'Runtime (ms)',
}
LEGEND_ORDER = [
    "express_new_32_32",
    "balancekv_express",
    "decoding_expected_attention",
    "decoding_snapkv",
    "decoding_keydiff",
    "decoding_streaming_llm",
    "decoding_knorm",
]
PRESS_MARKERS = {
    "express_new_32_32": "o",
    "balancekv_express": "o",
    "decoding_expected_attention": "v",
    "decoding_snapkv": "s",
    "decoding_keydiff": "D",
    "decoding_streaming_llm": "P",
    "decoding_knorm": "X",
}
# press names that represent the uncompressed ("Exact") reference point,
# drawn as a star rather than a swept line; mapped to their annotation label
NO_PRESS_LABELS = {
    "no_press": "Exact",
}
for col in COLUMNS:
    fig = plt.figure(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
    # get the mean sequence length for no_press
    max_time = df[df['press_name'] == "no_press"].apply(lambda x: x[col], axis=1).mean()
    press_handles_by_name = {}
    for press_name, group in df.groupby('press_name'):
        if press_name in NO_PRESS_LABELS:
            no_press_acc = group['all'].mean()
            no_press_time = group[col].mean()
            plt.plot(
                no_press_time,
                no_press_acc,
                color=COLORS.get(press_name, None),
                marker='*',
                markersize=10,
                alpha=LINE_ALPHA,
            )
            plt.annotate(
                NO_PRESS_LABELS[press_name],
                xy=(no_press_time, no_press_acc),
                xytext=(8, 0),
                textcoords='offset points',
                color=COLORS.get(press_name, "grey"),
                fontsize=TICK_FONT_SIZE,
                ha="left",
                va="center",
            )
        else:
            x = []
            y = []
            xerr = []
            yerr = []
            target_sizes = []
            # iterate over all unique target_size values
            unique_target_sizes = sorted(group['target_size'].unique())
            for target_size in unique_target_sizes:
                # get all rows for this specific target_size
                df_target = group[group['target_size'] == target_size]
                # check if we have data for all unique IDs
                # if df_target[ID].nunique() != group[ID].nunique():
                #     continue
                # get the mean of the accuracy column
                mean_acc = df_target['all'].mean()
                std_acc = df_target['all'].std(ddof=1) / np.sqrt(len(df_target))
                mean_time = df_target[col].mean()
                std_time = df_target[col].std(ddof=1) / np.sqrt(len(df_target))
                x.append(mean_time)
                y.append(mean_acc)
                xerr.append(std_time)
                yerr.append(std_acc)
                target_sizes.append(target_size)
            # plot acc vs cache size as scatter plot
            color = COLORS.get(press_name, None)
            marker = PRESS_MARKERS.get(press_name, MARKERS.get(press_name, 'o'))
            y = np.array(y)
            yerr = np.array(yerr)

            # Plot line connecting points from same configuration
            line_container = plt.plot(
                x,
                y,
                color=color,
                alpha=LINE_ALPHA * 0.6,
                linewidth=1,
                zorder=2,
            )
            # if press_name wasn't in COLORS, capture the color matplotlib actually
            # assigned to the line so the fill and legend handle match it
            if color is None:
                color = line_container[0].get_color()

            plt.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

            plt.scatter(
                x=x,
                y=y,
                color=color,
                marker=marker,
                alpha=LINE_ALPHA,
                s=10,
            )
            # Label every express_new_32_32 / balancekv_express point with its target size h
            if press_name in ("express_new_32_32", "balancekv_express"):
                for xi, yi, h in zip(x, y, target_sizes):
                    # h=8 overlaps its neighbour when placed above, so put it below
                    below = h == 8
                    plt.annotate(
                        f"$h={h:g}$",
                        xy=(xi, yi),
                        xytext=(0, -3 if below else 3),
                        textcoords='offset points',
                        fontsize=TICK_FONT_SIZE - 2,
                        ha='center',
                        va='top' if below else 'bottom',
                        alpha=0.7,
                    )
            press_handles_by_name[press_name] = lines.Line2D(
                [0],
                [0],
                color=color,
                label=METHOD_ALIASES.get(press_name, press_name),
                linewidth=0,
                marker=marker,
                markersize=np.sqrt(10),
            )
    ax= plt.gca()
    # set x-axis and y-axis ticks
    ax.spines["bottom"].set_position(("outward", OUTWARD))  # Move x-axis outward
    ax.spines["left"].set_position(("outward", OUTWARD))  # Move y-axis outward
    plt.xlabel(X_LABELS.get(col, col), fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.ylabel("Accuracy", fontsize=LABEL_FONT_SIZE)
    ax.grid(True, which="major", linestyle=":", alpha=0.5, axis="y")
    ax.grid(True, which="major", linestyle=":", alpha=0.5, axis="x")
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, axis="y")
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, axis="x")

    # set x-axis limits
    # plt.xlim(0, max_time)
    # in-axes legend (ordered)
    ordered_handles = [press_handles_by_name[name] for name in LEGEND_ORDER if name in press_handles_by_name]
    # append any remaining presses not listed in LEGEND_ORDER
    ordered_handles += [h for n, h in press_handles_by_name.items() if n not in LEGEND_ORDER]
    ax.legend(
        handles=ordered_handles,
        fontsize=LEGEND_FONT_SIZE - 2,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=1.0,
        borderaxespad=0.0,
        frameon=True,
    )
    # fixed margins so the saved figure is exactly figsize
    fig.subplots_adjust(left=0.175, right=0.98, bottom=0.225, top=0.97)

    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig_path = os.path.join(figure_dir, f"scatter_accuracy_vs_{col.replace('cum_', '')}-math500-{model.replace('/', '--')}.pdf")
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
