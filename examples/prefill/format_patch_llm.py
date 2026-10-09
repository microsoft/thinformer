"""Script for plotting perplexity vs number of patch layers.

Example usage:
```bash
python format_patch_llm.py -op OUTPUT_PATH
```
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from util_experiments import get_base_parser
import pandas as pd
import json
from plotting_utils import (
    LABEL_FONT_SIZE,
    XLABEL_FONT_SIZE,
    YLABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_ALPHA,
    LINE_WIDTH,
    MARKER_SIZE,
    OUTWARD,
    get_method_style,
)
from tabulate import tabulate


# Default per-method ordering used by plot_perplexity_on_ax. Controls draw and
# legend order; entries missing from the data are skipped. express_triton_2 is
# the current default (produced by slurm/benchmark_patch_llm_array.slurm) and
# a verified bit-identical replacement for express_triton (see get_method_style
# in plotting_utils.py); both are listed so output dirs from either are plotted.
DEFAULT_PLOT_ORDER = [
    ("thinformer_triton", 8),
    ("thinformer_triton", 9),
    ("thinformer_triton", 10),
    ("hyper-cuda", None),
    ("flash", None),
]


def load_perplexity_df(output_path: str) -> tuple[pd.DataFrame, float]:
    """Load and aggregate perplexity CSVs from `<output_path>/acc/`.

    Returns (acc DataFrame with ppls/ppls_sem/ppls_ratio/ppls_pct_increase
    columns, flash_ppl).
    """
    filepaths = glob(os.path.join(output_path, "acc", "acc_*.csv"))
    print(f"Found {len(filepaths)} files")
    df_list = []
    for filepath in filepaths:
        print(f"Reading file: {filepath}")
        df = pd.read_csv(filepath)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    df["attn_method"] = df["args"].apply(lambda x: json.loads(x).get("attn_method"))

    acc = (
        df.groupby(["attn_method", "num_patch_layers", "h"], dropna=False)["ppls"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    acc.columns = ["attn_method", "num_patch_layers", "h", "ppls", "ppls_sem"]

    # Use flash perplexity as the value for 0 patch layers across non-flash methods.
    flash_df = acc[acc["attn_method"] == "flash"]
    assert len(flash_df) == 1, (
        f"Expected only one entry for flash attention but got {len(flash_df)}"
    )
    flash_ppl = flash_df.iloc[0]["ppls"]
    extra_rows = []
    for _, row in acc[
        (acc["attn_method"] != "flash") & (acc["num_patch_layers"] == 2)
    ].iterrows():
        new_row = row.copy()
        new_row["num_patch_layers"] = 0
        new_row["ppls"] = flash_ppl
        new_row["ppls_sem"] = flash_df.iloc[0]["ppls_sem"]
        extra_rows.append(new_row)
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        acc = pd.concat([acc, extra_df], ignore_index=True)
    acc = acc.sort_values(by=["attn_method", "num_patch_layers", "h"])

    acc["ppls_ratio"] = acc["ppls"] / flash_ppl
    acc["ppls_pct_increase"] = 100.0 * (acc["ppls"] - flash_ppl) / flash_ppl

    return acc, flash_ppl


def plot_perplexity_on_ax(
    ax: plt.Axes,
    acc: pd.DataFrame,
    spec: dict,
    plot_order: list = DEFAULT_PLOT_ORDER,
) -> None:
    """Render one perplexity metric onto `ax`.

    `spec` keys: col, ylabel, exact_value, exact_label, optional ylim_top.
    The caller is responsible for the figure (creation, suptitle, savefig).
    """
    for attn_method, h_target in plot_order:
        if h_target is None:
            group = acc[acc["attn_method"] == attn_method]
        else:
            group = acc[
                (acc["attn_method"] == attn_method) & (acc["h"] == h_target)
            ]

        if len(group) == 0:
            continue

        if attn_method == "flash":
            assert len(group) == 1, "Expected only one entry for flash attention"
            flash_color = get_method_style("flash", None)["color"]
            ax.axhline(
                y=spec["exact_value"],
                color=flash_color,
                linestyle="--",
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
            )
            # Floating label sitting just above the right end of the dashed line.
            ax.text(
                28,
                spec["exact_value"],
                spec["exact_label"],
                ha="right",
                va="bottom",
                fontsize=LEGEND_FONT_SIZE,
                color=flash_color,
                alpha=LINE_ALPHA,
            )
        else:
            x = group["num_patch_layers"].values
            y = group[spec["col"]].values

            style = get_method_style(attn_method, h_target)

            ax.plot(
                x,
                y,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=style["label"],
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
                ms=MARKER_SIZE,
            )

    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))

    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.grid(True, linestyle=":", alpha=0.2, axis="y", which="minor")

    ax.set_xticks(np.arange(0, 30, 4))
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_xlim(-1, 29)
    ax.set_xlabel("Number of layers replaced", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel(spec["ylabel"], fontsize=YLABEL_FONT_SIZE)
    if "ylim_top" in spec:
        ax.set_ylim(top=spec["ylim_top"])

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        frameon=False,
    )


def main() -> None:
    """Render the three perplexity metric figures (raw / ratio / pct increase)."""
    parser = get_base_parser()
    args = parser.parse_args()
    output_path = args.output_path

    acc, flash_ppl = load_perplexity_df(output_path)
    print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))

    metric_specs = [
        {
            "col": "ppls",
            "ylabel": "Perplexity",
            "filename": "perplexity.pdf",
            "exact_value": flash_ppl,
            "exact_label": "Exact",
        },
        {
            "col": "ppls_ratio",
            "ylabel": "Relative perplexity",
            "filename": "perplexity_ratio.pdf",
            "exact_value": 1.0,
            "exact_label": "Exact",
            "ylim_top": 1.2,
        },
        {
            "col": "ppls_pct_increase",
            "ylabel": "Perplexity increase (\\%)",
            "filename": "perplexity_pct_increase.pdf",
            "exact_value": 0.0,
            "exact_label": "Exact",
        },
    ]

    figure_dir = os.path.join(output_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for spec in metric_specs:
        fig, ax = plt.subplots(figsize=(3.25, 2.5))
        plot_perplexity_on_ax(ax, acc, spec)
        ax.set_title("Sequence length $n = 32768$", fontsize=LABEL_FONT_SIZE)
        plt.subplots_adjust(right=0.9, top=0.9, bottom=0.15, wspace=0.3)

        fig_path = os.path.join(figure_dir, spec["filename"])
        print(f"Saving figure to {fig_path}")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
