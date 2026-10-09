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
# Script to aggregate accuracy based on all intermediate cache sizes,
# instead of hard-coded cache sizes.

# %%
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import MultipleLocator
import numpy as np
from tabulate import tabulate
import pandas as pd

from utils import (
    get_base_parser,
    get_data_math500,
    COLORS,
    LINE_STYLES,
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
ID = 'unique_id'
CACHE_SIZE = 'mean_cache_size'
df = get_data_math500(model, output_dir)

# %%
# print results as a table
mean_std = lambda x: f"{x.mean():.2f} ± {x.std(ddof=1) / np.sqrt(len(x)):.2f}"
# NOTE: to ensure backward compatibility, we keep the old column names
acc = df.groupby(["press_name", 'target_size'], dropna=False)[[CACHE_SIZE, "all", "completion_tokens", "prompt_tokens"]].agg(mean_std).reset_index()
acc['count'] = df.groupby(["press_name", 'target_size'], dropna=False).size().values
print(tabulate(acc, headers="keys", tablefmt="github"))

# %%
fig, ax = plt.subplots(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
LEGEND_ORDER = [
    "express_new_32_32",
    "balancekv_express",
    "decoding_expected_attention",
    "decoding_snapkv",
    "decoding_keydiff",
    "decoding_streaming_llm",
    "decoding_knorm",
]

press_handles_by_name = {}
no_press_group = df[df['press_name'] == 'no_press']
if not no_press_group.empty:
    no_press_acc = no_press_group['all'].mean()
    no_press_cache_size = no_press_group[CACHE_SIZE].mean()
    plt.plot(
        no_press_cache_size,
        no_press_acc,
        color=COLORS.get("no_press", None),
        marker='*',
        markersize=10,
        alpha=LINE_ALPHA,
    )
    print("Cache size for no_press: ", no_press_cache_size)
    print("Accuracy for no_press: ", no_press_acc)
else:
    no_press_cache_size = no_press_acc = None
for press_name, group in df.groupby('press_name'):
    if press_name.startswith("no_press"):
        continue
    x = []
    y = []
    xerr = []
    yerr = []
    # iterate over all unique values in real_compression_ratio
    unique_cache_sizes = sorted(group[CACHE_SIZE].unique())
    for cache_size in unique_cache_sizes:
        # get all rows where real_compression_ratio is less than or equal to cache_size
        df_cache_size = group[group[CACHE_SIZE] <= cache_size]
        if df_cache_size[ID].nunique() != group[ID].nunique():
            continue
        
        # 1. group by _id and get the row with the largest real_compression_ratio
        df_cache_size = df_cache_size[df_cache_size[CACHE_SIZE] == df_cache_size.groupby(ID)[CACHE_SIZE].transform('max')]
        # 2. group by _id and get the row with the smallest compression_ratio (if multiple compression ratios lead to the same real_compression_ratio)
        # Note: varying the seed doesn't change the real compression ratio, so we don't need to group by seed
        df_cache_size = df_cache_size[df_cache_size['target_size'] == df_cache_size.groupby(ID)['target_size'].transform('max')]

        # ensure that there is one row per _id and seed
        if len(df_cache_size) != df_cache_size[ID].nunique():
            import pdb; pdb.set_trace()
        # get the mean of the accuracy column
        mean_acc = df_cache_size['all'].mean()
        std_acc = df_cache_size['all'].std(ddof=1) / np.sqrt(len(df_cache_size))
        mean_cache_size = df_cache_size[CACHE_SIZE].mean()                
        std_cache_size = df_cache_size[CACHE_SIZE].std(ddof=1) / np.sqrt(len(df_cache_size))
        x.append(mean_cache_size)
        y.append(mean_acc)
        xerr.append(std_cache_size)
        yerr.append(std_acc)
    # plot acc vs cache size
    # add error bars
    color = COLORS.get(press_name, None)
    y = np.array(y)
    yerr = np.array(yerr)
    container = plt.errorbar(
        x=x,
        y=y,
        # xerr=xerr,
        # yerr=yerr,
        color=color,
        marker=None,
        # capsize=0,
        alpha=LINE_ALPHA,
        linestyle=LINE_STYLES.get(press_name, "-"),
        linewidth=1,
    )
    # if press_name wasn't in COLORS, capture the color matplotlib actually
    # assigned to the line so the fill and legend handle match it
    if color is None:
        color = container.lines[0].get_color()
    plt.fill_between(
        x,
        y - yerr,
        y + yerr,
        color=color,
        alpha=0.1,
    )
    press_handles_by_name[press_name] = lines.Line2D(
        [0],
        [0],
        color=color,
        label=METHOD_ALIASES.get(press_name, press_name),
        linewidth=1,
        marker=None,
        linestyle=LINE_STYLES.get(press_name, "-"),
        alpha=LINE_ALPHA,
    )
    # find the smallest x such that y >= no_press_acc, and annotate it on the plot
    if no_press_acc is not None:
        smallest_lossless_cache_size = None
        for cache_size, acc in sorted(zip(x, y), key=lambda pair: pair[0]):
            if acc >= no_press_acc:
                smallest_lossless_cache_size = cache_size
                break
        print(f"Smallest lossless cache size for press {press_name}: {smallest_lossless_cache_size}")
        print("smallest cache size / no_press cache size: ", smallest_lossless_cache_size / no_press_cache_size if smallest_lossless_cache_size is not None and no_press_cache_size is not None else "N/A")
# set x-axis and y-axis ticks
ax.spines["bottom"].set_position(("outward", OUTWARD))  # Move x-axis outward
ax.spines["left"].set_position(("outward", OUTWARD))  # Move y-axis outward
plt.xlabel("Mean Cache Size", fontsize=LABEL_FONT_SIZE)
plt.xticks(fontsize=TICK_FONT_SIZE)
plt.yticks(fontsize=TICK_FONT_SIZE)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
plt.ylabel("Accuracy", fontsize=LABEL_FONT_SIZE)
ax.grid(True, which="major", linestyle=":", alpha=0.5, axis="y")
ax.grid(True, which="major", linestyle=":", alpha=0.5, axis="x")

# set x-axis lower bound; let matplotlib auto-scale the upper bound
ax.set_xlim(left=0)

# label the no_press point as "Exact"
if no_press_acc is not None:
    ax.annotate(
        "Exact",
        xy=(no_press_cache_size, no_press_acc),
        # placed below the star: above it would run off the top of the page
        xytext=(3, -7),
        textcoords='offset points',
        color=COLORS.get("no_press", "grey"),
        fontsize=TICK_FONT_SIZE,
        ha="right",
        va="top",
    )

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
fig_path = os.path.join(figure_dir, f"accuracy_vs_mean_cache_size-math500-{model.replace('/', '--')}.pdf")
plt.savefig(fig_path, dpi=300)
print(f"Figure saved to {fig_path}")

# %%
