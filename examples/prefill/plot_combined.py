"""Combined two-panel figure: speedup (left) + relative perplexity (right).

Example usage:
```bash
python plot_combined.py -op OUTPUT_PATH
```
"""

import os
import matplotlib.pyplot as plt
from util_experiments import get_base_parser
from plotting_utils import (
    YLABEL_FONT_SIZE,
    NEURIPS_HEIGHT,
    NEURIPS_WIDTH,
)
from plot_speedup import plot_speedup_on_ax
from format_patch_llm import load_perplexity_df, plot_perplexity_on_ax


def main() -> None:
    """Render the side-by-side speedup + relative-perplexity figure."""
    parser = get_base_parser()
    args = parser.parse_args()
    output_path = args.output_path

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NEURIPS_WIDTH, NEURIPS_HEIGHT),
        gridspec_kw={"width_ratios": [1, 1]},
        constrained_layout=True,
    )

    plot_speedup_on_ax(axes[0], output_path)
    # Override the helper's two-line ylabel with a one-line version.
    axes[0].set_ylabel("Speed-up over FlashAttention 2", fontsize=YLABEL_FONT_SIZE)

    acc, _flash_ppl = load_perplexity_df(output_path)
    spec = {
        "col": "ppls_ratio",
        "ylabel": "Relative perplexity",
        "exact_value": 1.0,
        "exact_label": "Exact",
        "ylim_top": 1.2,
    }
    plot_perplexity_on_ax(axes[1], acc, spec)
    axes[1].set_ylabel(spec["ylabel"], fontsize=YLABEL_FONT_SIZE)

    # fig.suptitle("$n = 32768$", fontsize=LABEL_FONT_SIZE)

    figure_dir = os.path.join(output_path, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig_path = os.path.join(figure_dir, "combined.pdf")
    print(f"Saving figure to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
