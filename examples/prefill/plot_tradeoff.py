"""Script for plotting the quality/cost trade-off: cost on x, perplexity on y.

Combines the runtimes in `<output_path>/times_all/` with the perplexities in
`<output_path>/acc/`; both are parameterized by the number of layers replaced,
which becomes the parameter traced along each curve rather than an axis.

Example usage:
```bash
python plot_tradeoff.py -op OUTPUT_PATH
```
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from util_experiments import get_base_parser
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
    NEURIPS_WIDTH,
    NEURIPS_HEIGHT,
    get_method_style,
)
from plot_speedup import load_speedup_data, build_speedup_df
from format_patch_llm import DEFAULT_PLOT_ORDER, load_perplexity_df
from tabulate import tabulate


# Sentinel h used as a merge key. Methods without an h (hyper-cuda) store NaN in
# both tables, and NaN != NaN would silently drop them from the join.
NO_H = -1.0


def load_tradeoff_df(output_path: str) -> tuple[pd.DataFrame, float, float]:
    """Join the perplexity and runtime tables on (method, layers, h).

    Returns (merged DataFrame with the perplexity columns of `load_perplexity_df`
    plus `inv_ppls`/`speedup`/`time_ms`, flash_ppl, exact_time_ms). Only layer
    counts present in both tables survive: perplexity is measured every other
    layer, runtime at every layer.
    """
    acc, flash_ppl = load_perplexity_df(output_path)
    speedup_data = load_speedup_data(output_path)
    speedup_df = build_speedup_df(speedup_data["methods_data"])

    if len(speedup_df) == 0:
        return speedup_df, flash_ppl, speedup_data["exact_total"]

    acc = acc.assign(h_key=acc["h"].fillna(NO_H))
    speedup_df = speedup_df.assign(
        h_key=speedup_df["h"].astype(float).fillna(NO_H)
    ).drop(columns=["h"])

    merged = acc.merge(
        speedup_df, on=["attn_method", "num_patch_layers", "h_key"], how="inner"
    ).drop(columns=["h_key"])

    merged["inv_ppls"] = 1.0 / merged["ppls"]
    # Sort so each curve is drawn in order of increasing layers replaced.
    merged = merged.sort_values(by=["attn_method", "h", "num_patch_layers"])

    return merged, flash_ppl, speedup_data["exact_total"]


def plot_tradeoff_on_ax(
    ax: plt.Axes,
    merged: pd.DataFrame,
    spec: dict,
    plot_order: list = DEFAULT_PLOT_ORDER,
) -> None:
    """Render one cost/quality trade-off onto `ax`.

    `spec` keys: xcol, xlabel, ycol, ylabel, exact_value, exact_label,
    exact_label_loc, legend_loc, optional ylim_top.
    The caller is responsible for the figure (creation, title, savefig).
    """
    for attn_method, h_target in plot_order:
        # The exact baseline is named "flash" in the perplexity table and
        # "exact" in the runtime table, so it never survives the join; it is
        # drawn as the reference line below instead.
        if attn_method == "flash":
            continue

        if h_target is None:
            group = merged[merged["attn_method"] == attn_method]
        else:
            group = merged[
                (merged["attn_method"] == attn_method) & (merged["h"] == h_target)
            ]

        if len(group) == 0:
            continue

        style = get_method_style(attn_method, h_target, causal=True)

        ax.plot(
            group[spec["xcol"]].values,
            group[spec["ycol"]].values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            ms=MARKER_SIZE,
        )

    # Reference line at exact attention with a floating label. Unlike the
    # layers-replaced plots, x is no longer a fixed 0-28 range, so the label is
    # anchored in axes fraction along x (and in data coordinates along y).
    flash_color = get_method_style("flash", None)["color"]
    ax.axhline(
        y=spec["exact_value"],
        color=flash_color,
        linestyle="--",
        linewidth=LINE_WIDTH,
        alpha=LINE_ALPHA,
    )
    ha, va = spec["exact_label_loc"]
    ax.text(
        0.98 if ha == "right" else 0.02,
        spec["exact_value"],
        spec["exact_label"],
        transform=ax.get_yaxis_transform(),
        ha=ha,
        va=va,
        fontsize=LEGEND_FONT_SIZE,
        color=flash_color,
        alpha=LINE_ALPHA,
    )

    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))

    # Both axes carry data now, so grid both (the layers-replaced plots grid y only).
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(spec["xlabel"], fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel(spec["ylabel"], fontsize=YLABEL_FONT_SIZE)

    if "ylim_top" in spec:
        ax.set_ylim(top=spec["ylim_top"])
    else:
        # A three-entry legend spans most of these half-width panels, so reserve
        # a band on the legend's side rather than letting it sit over the curves.
        # An explicit ylim_top is assumed to already leave room.
        bottom, top = ax.get_ylim()
        pad = spec.get("legend_headroom", 0.3) * (top - bottom)
        if spec["legend_loc"].startswith("upper"):
            ax.set_ylim(top=top + pad)
        else:
            ax.set_ylim(bottom=bottom - pad)

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc=spec["legend_loc"],
        # frameon=False,
    )


def main() -> None:
    """Render the four cost/quality trade-off figures."""
    parser = get_base_parser()
    args = parser.parse_args()
    output_path = args.output_path

    merged, flash_ppl, exact_time_ms = load_tradeoff_df(output_path)
    if len(merged) == 0:
        print(
            f"No (method, layers, h) combination has both perplexity and "
            f"runtime data in {output_path}; nothing to plot."
        )
        return

    print(
        tabulate(
            merged[
                [
                    "attn_method",
                    "num_patch_layers",
                    "h",
                    "ppls",
                    "ppls_ratio",
                    "inv_ppls",
                    "speedup",
                    "time_ms",
                ]
            ],
            headers="keys",
            tablefmt="github",
            showindex=False,
        )
    )
    print(f"\nExact: ppl = {flash_ppl:.4f}, attention time = {exact_time_ms:.1f} ms")

    # Replacing more layers always costs quality and buys speed, so every curve
    # runs from the exact corner into one corner of the panel: up-and-right for
    # speed-up vs. relative perplexity, mirrored horizontally for runtime (which
    # decreases) and vertically for inverse perplexity (which decreases).
    # legend_loc and exact_label_loc name the corner and the side of the
    # reference line that the curves leave empty.
    metric_specs = [
        {
            "xcol": "speedup",
            "xlabel": "Speed-up over FlashAttention 2",
            "ycol": "ppls_ratio",
            "ylabel": "Relative perplexity",
            "filename": "tradeoff_speedup_vs_perplexity_ratio.pdf",
            "exact_value": 1.0,
            "exact_label": "Exact",
            "ylim_top": 1.2,
            "exact_label_loc": ("right", "bottom"),
            "legend_loc": "upper left",
        },
        {
            "xcol": "time_ms",
            "xlabel": "Runtime (ms)",
            "ycol": "ppls_ratio",
            "ylabel": "Relative perplexity",
            "filename": "tradeoff_runtime_vs_perplexity_ratio.pdf",
            "exact_value": 1.0,
            "exact_label": "Exact",
            "ylim_top": 1.2,
            "exact_label_loc": ("left", "bottom"),
            "legend_loc": "upper right",
        },
        {
            "xcol": "speedup",
            "xlabel": "Speed-up over FlashAttention 2",
            "ycol": "inv_ppls",
            "ylabel": "Inverse perplexity",
            "filename": "tradeoff_speedup_vs_inverse_perplexity.pdf",
            "exact_value": 1.0 / flash_ppl,
            "exact_label": "Exact",
            "exact_label_loc": ("right", "top"),
            "legend_loc": "lower left",
            # The h=9 curve runs furthest right while staying low, so it needs
            # more clearance under it than the default band.
            "legend_headroom": 0.5,
        },
        {
            "xcol": "time_ms",
            "xlabel": "Runtime (ms)",
            "ycol": "inv_ppls",
            "ylabel": "Inverse perplexity",
            "filename": "tradeoff_runtime_vs_inverse_perplexity.pdf",
            "exact_value": 1.0 / flash_ppl,
            "exact_label": "Exact",
            "exact_label_loc": ("left", "top"),
            "legend_loc": "lower right",
        },
    ]

    figure_dir = os.path.join(output_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for spec in metric_specs:
        fig, ax = plt.subplots(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
        plot_tradeoff_on_ax(ax, merged, spec)
        # ax.set_title("Sequence length $n = 32768$", fontsize=LABEL_FONT_SIZE)
        plt.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)

        fig_path = os.path.join(figure_dir, spec["filename"])
        print(f"Saving figure to {fig_path}")
        plt.savefig(fig_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
