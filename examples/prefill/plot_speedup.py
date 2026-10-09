#!/usr/bin/env python3
"""Plot speed-up over exact attention as a function of number of layers replaced.

This script:
1. Reads CSV files containing attention layer runtimes for different methods
2. Computes total attention time with X layers replaced by exact attention
3. Plots speed-up over full exact attention as a line chart

Example usage:
```bash
python plot_speedup.py -op OUTPUT_PATH
```
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob
from util_experiments import get_base_parser
from plotting_utils import (
    LABEL_FONT_SIZE,
    XLABEL_FONT_SIZE,
    YLABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    LINE_ALPHA,
    MARKER_SIZE,
    OUTWARD,
    get_method_style,
)


def load_method_data(
    csv_path: str, seq_len: int = 32768
) -> tuple[pd.Series | None, int | None]:
    """Load data for a specific method and sequence length.

    Returns:
        row: DataFrame row with the data
        h: The h parameter extracted from filename (None if not applicable)

    """
    import re

    df = pd.read_csv(csv_path)
    # Filter for the target sequence length
    row = df[df["seq_len"] == seq_len]
    if len(row) == 0:
        return None, None

    # Extract h from filename (e.g., "h10" -> 10, "h9" -> 9)
    # Note: there are two h's in the filename
    # times_all-n100-thinformer_triton-fwd-bs1-h32-d128-s1-h9-True-causalTrue-thudm-chatglm2-6b-32k.csv
    # The first one is the number of heads and the second one is the h parameter we want.
    h_match = re.search(r"h(\d+)-.*-h(\d+)", csv_path)
    h = int(h_match.group(2)) if h_match else None

    return row.iloc[0], h


def get_attn_columns(row: pd.Series) -> list[str]:
    """Extract all columns matching the pattern 0_attn, 1_attn, etc. from a row."""
    import re

    pattern = re.compile(r"^(\d+)_attn$")
    attn_cols = [col for col in row.index if pattern.match(col)]
    # Sort by layer number
    attn_cols.sort(key=lambda x: int(pattern.match(x).group(1)))
    return attn_cols


def get_attn_std_columns(row: pd.Series) -> list[str]:
    """Extract all columns matching the pattern 0_attn_std, 1_attn_std, etc. from a row."""
    import re

    pattern = re.compile(r"^(\d+)_attn_std$")
    attn_std_cols = [col for col in row.index if pattern.match(col)]
    # Sort by layer number
    attn_std_cols.sort(key=lambda x: int(pattern.match(x).group(1)))
    return attn_std_cols


def compute_hybrid_times(
    method_row: pd.Series, exact_row: pd.Series
) -> tuple[list[float], list[float], list[int]]:
    """Compute total attention time when replacing the last X layers with the method.

    Args:
        method_row: DataFrame row for the approximate method
        exact_row: DataFrame row for exact attention

    Returns:
        total_times: Total attention time (ms) for 0 to 28 layers replaced
        total_time_stds: List of standard deviations for the total times
        num_layers_replaced: List of layer counts (0 to 28)

    """
    method_cols = get_attn_columns(method_row)
    exact_cols = get_attn_columns(exact_row)
    method_std_cols = get_attn_std_columns(method_row)
    exact_std_cols = get_attn_std_columns(exact_row)

    # Ensure both have the same number of layers
    assert len(method_cols) == len(exact_cols), (
        f"Mismatch in number of layers. Expected {len(exact_cols)}, got {len(method_cols)}"
    )
    n_layers = len(method_cols)

    # Get the runtime values and standard deviations
    method_times = [method_row[col] for col in method_cols]
    exact_times = [exact_row[col] for col in exact_cols]
    method_stds = [
        method_row[col] if col in method_row.index else 0 for col in method_std_cols
    ]
    exact_stds = [
        exact_row[col] if col in exact_row.index else 0 for col in exact_std_cols
    ]

    # Total time for full exact attention
    exact_total = sum(exact_times)
    exact_total_std = (sum([s**2 for s in exact_stds])) ** 0.5

    total_times = []
    total_time_stds = []
    num_layers_replaced = list(range(n_layers + 1))  # 0 to 28 layers

    for n_replace in num_layers_replaced:
        if n_replace == 0:
            # No layers replaced - pure exact
            total_time = exact_total
            total_time_std = exact_total_std
        elif n_replace == n_layers:
            # All layers replaced - pure method
            total_time = sum(method_times)
            total_time_std = (sum([s**2 for s in method_stds])) ** 0.5
        else:
            # Replace last n_replace layers with method
            # Use exact times for first (n_layers - n_replace) layers
            # Use method times for last n_replace layers
            total_time = sum(exact_times[: n_layers - n_replace]) + sum(
                method_times[n_layers - n_replace :]
            )
            # Propagate errors (variances add)
            total_time_std = (
                sum([s**2 for s in exact_stds[: n_layers - n_replace]])
                + sum([s**2 for s in method_stds[n_layers - n_replace :]])
            ) ** 0.5

        total_times.append(total_time)
        total_time_stds.append(total_time_std)

    return total_times, total_time_stds, num_layers_replaced


def compute_speedup_vs_layers_replaced(
    method_row: pd.Series, exact_row: pd.Series
) -> tuple[list[float], list[float], list[int]]:
    """Compute speed-up when replacing the last X layers with exact attention.

    Args:
        method_row: DataFrame row for the approximate method
        exact_row: DataFrame row for exact attention

    Returns:
        speedups: List of speed-ups for 0 to 28 layers replaced
        speedup_errors: List of standard errors for speed-ups
        num_layers_replaced: List of layer counts (0 to 28)

    """
    total_times, total_time_stds, num_layers_replaced = compute_hybrid_times(
        method_row, exact_row
    )
    # n_replace = 0 is the pure-exact configuration, i.e. the baseline.
    exact_total = total_times[0]
    exact_total_std = total_time_stds[0]

    speedups = []
    speedup_errors = []

    for total_time, total_time_std in zip(total_times, total_time_stds):
        # Speed-up = exact_time / hybrid_time
        speedup = exact_total / total_time
        speedups.append(speedup)

        # Error propagation for division: σ(a/b) = (a/b) * sqrt((σ_a/a)² + (σ_b/b)²)
        if total_time > 0 and exact_total > 0:
            rel_error = (
                (exact_total_std / exact_total) ** 2
                + (total_time_std / total_time) ** 2
            ) ** 0.5
            speedup_error = speedup * rel_error
        else:
            speedup_error = 0
        speedup_errors.append(speedup_error)

    return speedups, speedup_errors, num_layers_replaced


def build_speedup_df(methods_data: dict) -> pd.DataFrame:
    """Assemble the per-method speed-ups into a tidy DataFrame.

    Mirrors the `acc` table in format_patch_llm.py (attn_method /
    num_patch_layers / h), with speed-up and total attention time (ms) in place
    of perplexity.
    """
    rows = []
    for data in methods_data.values():
        for num_patch_layers, speedup, speedup_sem, time_ms, time_ms_sem in zip(
            data["num_layers"],
            data["speedups"],
            data["speedup_errors"],
            data["times"],
            data["time_errors"],
        ):
            rows.append(
                {
                    "attn_method": data["method_name"],
                    "num_patch_layers": num_patch_layers,
                    "h": data["h"],
                    "speedup": speedup,
                    "speedup_sem": speedup_sem,
                    "time_ms": time_ms,
                    "time_ms_sem": time_ms_sem,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(by=["attn_method", "num_patch_layers", "h"])


def load_speedup_data(output_path: str) -> dict:
    """Load and filter the per-method runtimes from `<output_path>/times_all/`.

    Returns a dict with `num_layers`, `methods_data`, `ordered_methods` (the
    (key, data) pairs in draw order) and `exact_total` (total attention time of
    full exact attention, in ms). Each entry of `methods_data` holds both the
    speed-ups and the absolute hybrid times.
    """
    data_dir = Path(output_path) / "times_all"
    csv_files = glob(str(data_dir / "times_all-*.csv"))

    exact_file = [f for f in csv_files if "exact-fwd" in f][0]
    exact_row, _ = load_method_data(exact_file, seq_len=32768)

    if exact_row is None:
        print("Error: Could not find seq_len=32768 in exact file")
        return {
            "num_layers": [],
            "methods_data": {},
            "ordered_methods": [],
            "exact_total": None,
        }

    methods_data = {}
    num_layers: list[int] = []

    for csv_file in csv_files:
        method_row, h = load_method_data(csv_file, seq_len=32768)
        if method_row is None:
            continue

        method_name = method_row["method"]

        if method_name == "exact":
            continue

        # Only include HyperAttention, Express g=5, g=4.5, g=4.
        # express_triton_2 is a verified bit-identical replacement for
        # express_triton (see get_method_style in plotting_utils.py), so both
        # map to the same method_key/style.
        if method_name == "hyper-cuda":
            method_key = "HyperAttention"
        elif method_name == "thinformer_triton" and h == 10:
            method_key = "thinformer_triton_h10"
        elif method_name == "thinformer_triton" and h == 9:
            method_key = "thinformer_triton_h9"
        elif method_name == "thinformer_triton" and h == 8:
            method_key = "thinformer_triton_h8"
        else:
            continue

        speedups, speedup_errors, num_layers = compute_speedup_vs_layers_replaced(
            method_row, exact_row
        )
        times, time_errors, _ = compute_hybrid_times(method_row, exact_row)
        methods_data[method_key] = {
            "speedups": speedups,
            "speedup_errors": speedup_errors,
            "times": times,
            "time_errors": time_errors,
            "num_layers": num_layers,
            "method_name": method_name,
            "h": h,
        }

    # Define the order: Express g=4, g=4.5, g=5, HyperAttention
    method_order = [
        "thinformer_triton_h8",
        "thinformer_triton_h9",
        "thinformer_triton_h10",
        "HyperAttention",
    ]

    ordered_methods = [
        (key, methods_data[key]) for key in method_order if key in methods_data
    ]

    return {
        "num_layers": num_layers,
        "methods_data": methods_data,
        "ordered_methods": ordered_methods,
        "exact_total": sum(exact_row[col] for col in get_attn_columns(exact_row)),
    }


def plot_speedup_on_ax(ax: plt.Axes, output_path: str) -> dict:
    """Render the speedup bar chart onto `ax`.

    Returns a dict with `num_layers` and `methods_data` for callers that want stats.
    The caller is responsible for the figure (creation, suptitle, savefig).
    """
    result = load_speedup_data(output_path)
    num_layers = result["num_layers"]
    if not num_layers:
        return result

    x_base = np.arange(len(num_layers))

    for method_key, data in result["ordered_methods"]:
        style = get_method_style(data["method_name"], data["h"], causal=True)
        ax.plot(
            x_base,
            data["speedups"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            ms=MARKER_SIZE,
            markevery=2,
        )

    ax.spines["bottom"].set_position(("outward", OUTWARD))
    ax.spines["left"].set_position(("outward", OUTWARD))

    ax.grid(True, linestyle=":", axis="y", alpha=0.5)

    ax.set_xticks(np.arange(0, 30, 4))
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.set_xlim(-1, 29)
    ax.set_xlabel("Number of layers replaced", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Speed-up over\nFlashAttention 2", fontsize=YLABEL_FONT_SIZE)
    # Pad slightly below 1.0 so the markers at y=1 (n=0) aren't clipped.
    ax.set_ylim(bottom=0.95)
    ax.set_yticks(np.arange(1, int(ax.get_ylim()[1]) + 2, 1))
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    # Reference line at the 1× baseline with a floating "Exact" annotation.
    flash_color = get_method_style("flash", None)["color"]
    ax.axhline(
        y=1.0, color=flash_color, linestyle="--", linewidth=LINE_WIDTH, alpha=LINE_ALPHA
    )
    ax.text(
        28,
        1.0,
        "Exact",
        ha="right",
        va="bottom",
        fontsize=LEGEND_FONT_SIZE,
        color=flash_color,
        alpha=LINE_ALPHA,
    )

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        frameon=False,
    )

    return result


def main() -> None:
    """Compute speed-up over exact attention as a function of number of layers replaced
    and plot as a line chart.
    """
    parser = get_base_parser()
    args = parser.parse_args()
    output_path = args.output_path

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    result = plot_speedup_on_ax(ax, output_path)
    methods_data = result["methods_data"]
    num_layers = result["num_layers"]

    if not methods_data:
        return

    ax.set_title("Sequence length $n = 32768$", fontsize=LABEL_FONT_SIZE)
    plt.subplots_adjust(right=0.9, top=0.9, bottom=0.15, wspace=0.3)

    figure_dir = os.path.join(output_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig_path = os.path.join(figure_dir, "speedup.pdf")
    print(f"Saving figure to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Summary Statistics ===")
    print("Sequence length: 32768")
    print(f"Number of layers: {len(num_layers) - 1}")
    print("\nSpeed-up with 28 layers replaced (pure method):")
    for method_name, data in methods_data.items():
        print(f"  {method_name}: {data['speedups'][28]:.2f}x")


if __name__ == "__main__":
    main()
