"""Cumulative max-norm error vs token position, overlaid across h.

For each (batch, head) and token position n, the per-position error is the
max-norm over the D output dimensions:

    maxnorm(b, h, n) = max_d |approx[b, n, h, d] - ref[b, n, h, d]|

We take the cumulative (prefix) max over positions per sample, then the median
(and 25%/75% quantiles) over the (B, H) sample axis, and plot the guarantee-scaled
quantity  max_{m<=n} ||o_m - o_hat_m||_inf / sqrt(log n * log log n)  vs n.

One curve is drawn per Thinformer Express output found in
<output_path>/attn_outputs/<input_source>/ for
--src_seq, all compared against the single shared exact reference, so the
settings (log2 target cache sizes h) can be overlaid on one figure, saved to
<output_path>/figures/maxnorm_trunc_<input_source>.pdf.

Data layout (written by `benchmark_single_attention.py --save_outputs`, one
approx file per h, plus the shared exact reference), all key "out":
    <output_path>/attn_outputs/<input_source>/Out_layer_thinformer_triton_h<H>_seq<SRC>.npz   approx, (B, N, H, D)
    <output_path>/attn_outputs/<input_source>/Out_layer_exact_seq<SRC>.npz                    ref
A "layout" key ("bnhd" / "bhnd") says how the array is laid out. Legacy dumps
without it are (B, H, N, D) for exact and (B, N, H, D) otherwise, and legacy
approx files named Out_layer_express_triton_2_geps<G>_seq<SRC>.npz are read
with h = 2 * G.

Prerequisite -- generate the outputs first (run from examples/prefill): the
exact reference once, and thinformer_triton once per h you want to overlay:
    python benchmark_single_attention.py -m exact -bs 1 -hs 32 -d 128 -s 1 \\
        -op out-test --force --num_runs 1 --num_warmup 0 --input_source glm_embed --save_outputs
    python benchmark_single_attention.py -m thinformer_triton --log2_target_cache_size 9 \\
        -bs 1 -hs 32 -d 128 -s 1 -op out-test --force --num_runs 1 --num_warmup 0 \\
        --input_source glm_embed --save_outputs

Usage (run from examples/prefill):
    python compute_maxnorm_truncate.py -op out-test --plot_stride 1
    python compute_maxnorm_truncate.py -op out-test --h 8 9 10   # subset
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
from tabulate import tabulate
from plotting_utils import (
    XLABEL_FONT_SIZE,
    YLABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_ALPHA,
    LINE_WIDTH,
    OUTWARD,
    NEURIPS_WIDTH,
    NEURIPS_HEIGHT,
    get_method_style,
)

APPROX_METHOD = "thinformer_triton"
# Approx files carry an h<H> tag (benchmark_single_attention.py --save_outputs)
# or, for legacy dumps, the old kernel name with a geps<G> tag (h = 2 * G).
APPROX_PATTERNS = [
    ("Out_layer_" + APPROX_METHOD + "_h*_seq{}.npz", r"_h(\d+(?:\.\d+)?)_seq", 1.0),
    ("Out_layer_express_triton_2_geps*_seq{}.npz", r"geps(\d+(?:\.\d+)?)_seq", 2.0),
]
REF_TMPL = "Out_layer_exact_seq{}.npz"


def load_out(path, legacy_layout="bnhd"):
    """Load one dump as (B, N, H, D), keeping the raw dtype (float16); upcast
    per chunk when comparing. Dumps carry a "layout" key; older ones do not,
    and use `legacy_layout`."""
    with np.load(path) as d:
        out = d["out"]
        layout = str(d["layout"]) if "layout" in d.files else legacy_layout
    if layout == "bhnd":
        out = np.transpose(out, (0, 2, 1, 3))
    elif layout != "bnhd":
        raise ValueError(f"{path}: unknown layout {layout!r}")
    return out


def per_position_maxnorm(approx, ref, chunk=1024):
    """max abs-diff over D at each token position n, keeping (B, H) as separate
    samples. approx/ref are (B, N, H, D) float16. Returns (B*H, N). Chunked over
    the position axis (dim 1) to bound peak memory."""
    B, N, H, D = approx.shape
    out = np.empty((B * H, N), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        a = np.ascontiguousarray(approx[:, s:e]).astype(np.float32)  # (B,c,H,D)
        r = np.ascontiguousarray(ref[:, s:e]).astype(np.float32)
        m = np.abs(a - r).max(axis=3)                    # (B, c, H) max over D
        out[:, s:e] = np.transpose(m, (0, 2, 1)).reshape(B * H, e - s)
    return out


def find_approx_files(data_dir, src_seq):
    """Return sorted [(h, path)] for every Thinformer Express dump at src_seq,
    h being the log2 target cache size (2 * g_eps for legacy geps files)."""
    found = {}
    for glob_tmpl, regex, scale in APPROX_PATTERNS:
        for path in glob.glob(os.path.join(data_dir, glob_tmpl.format(src_seq))):
            m = re.search(regex, os.path.basename(path))
            if m:
                found.setdefault(scale * float(m.group(1)), path)  # h-tagged wins
    return sorted(found.items(), key=lambda hp: hp[0])


def curve_color(h, fallback):
    """Release palette for the h values it knows; otherwise a cycle color."""
    try:
        return get_method_style(APPROX_METHOD, int(h) if float(h).is_integer() else h)["color"]
    except ValueError:
        return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_path", "-op", default="out",
                    help="same --output_path passed to benchmark_single_attention.py; "
                         "dumps are read from <output_path>/attn_outputs/<input_source>/")
    ap.add_argument("--input_source", default="glm_embed",
                    choices=["random", "glm_embed"],
                    help="same --input_source passed to benchmark_single_attention.py")
    ap.add_argument("--src_seq", type=int, default=524288,
                    help="seq length of the source outputs")
    ap.add_argument("--h", type=float, nargs="*", default=None,
                    help="restrict to these log2 target cache sizes h = 2*g_eps "
                         "(default: all found)")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="positions loaded per step (memory/speed knob)")
    ap.add_argument("--plot_stride", type=int, default=1024,
                    help="plot every k-th point (computation stays full-res); "
                         "keeps PDFs light (default: 1024)")
    args = ap.parse_args()
    data_dir = os.path.join(args.output_path, "attn_outputs", args.input_source)
    figures_dir = os.path.join(args.output_path, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, f"maxnorm_trunc_{args.input_source}.pdf")

    series = find_approx_files(data_dir, args.src_seq)
    if args.h is not None:
        wanted = set(args.h)
        series = [(h, p) for h, p in series if h in wanted]
    if not series:
        raise SystemExit(
            f"No {APPROX_METHOD} dumps found for seq {args.src_seq} under "
            f"{data_dir} (patterns "
            f"{[p.format(args.src_seq) for p, _, _ in APPROX_PATTERNS]})")
    print("h to overlay:", [h for h, _ in series])

    # Load the shared exact reference once.
    ref_path = os.path.join(data_dir, REF_TMPL.format(args.src_seq))
    ref = load_out(ref_path, legacy_layout="bhnd")   # (B, N, H, D)
    N = ref.shape[1]
    assert N == args.src_seq, f"expected N={args.src_seq}, got {N}"

    # Per-n factors, tick positions, and downsampling indices (shared across
    # h since N is fixed).
    n = np.arange(1, N + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_loglog = np.sqrt(np.log(n) * np.log(np.log(n)))  # sqrt(log n log log n)
    kmax = int(np.floor(np.log2(N)))
    pows2 = 2.0 ** np.arange(0, kmax + 1)
    idx = np.unique(np.r_[np.arange(0, N, args.plot_stride), N - 1])
    nd = n[idx]
    grid2 = [2 ** k for k in range(kmax + 1)]  # table positions, <= N

    CUM = r"\max_{m\leq n}\|o_m-\hat{o}_m\|_\infty"
    YLABEL = rf"${CUM}\,/\,\sqrt{{\log n\,\log\log n}}$"

    # Compute the scaled cumulative-max quantiles for each h (one pair
    # resident at a time to bound memory), and print a table per h.
    plots = []  # (h, t25, t50, t75)
    for h, approx_path in series:
        approx = load_out(approx_path)         # (B, N, H, D)
        assert approx.shape == ref.shape, \
            f"shape mismatch (h={h}): {approx.shape} vs {ref.shape}"
        curves = per_position_maxnorm(approx, ref, args.chunk)  # (B*H, N)
        cummax = np.maximum.accumulate(curves, axis=1)
        c25, c50, c75 = np.quantile(cummax, [0.25, 0.5, 0.75], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t25, t50, t75 = c25 / f_loglog, c50 / f_loglog, c75 / f_loglog
        for arr in (t25, t50, t75):
            arr[~np.isfinite(arr)] = np.nan
        plots.append((h, t25, t50, t75))
        rows = [[g, f"{t50[g - 1]:.6g}", f"{t25[g - 1]:.6g}", f"{t75[g - 1]:.6g}"]
                for g in grid2]
        print(f"\n## h={h:g}: cummax / sqrt(log n log log n)")
        print(tabulate(rows, headers=["n", "median", "25%", "75%"],
                       tablefmt="github"))
        del approx, curves, cummax

    def style_x(ax):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(pows2[::2]))   # even powers
        ax.xaxis.set_minor_locator(FixedLocator(pows2))        # every power
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.spines["bottom"].set_position(("outward", OUTWARD))
        ax.spines["left"].set_position(("outward", OUTWARD))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.grid(True, linestyle=":", alpha=0.2, axis="x", which="minor")
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    # Overlay one median + IQR band per h, format_patch_llm style.
    fig = plt.figure(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
    ax = plt.gca()
    for i, (h, t25, t50, t75) in enumerate(plots):
        color = curve_color(h, f"C{i}")
        mid = t50[idx]
        ax.plot(nd, mid, color=color, linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
        ax.fill_between(nd, t25[idx], t75[idx], color=color, alpha=0.2)
        # Label each curve with h at its right-most point, in place of a legend.
        ax.annotate(rf"$h={h:g}$", xy=(nd[-1], mid[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", color=color,
                    fontsize=LEGEND_FONT_SIZE, clip_on=False)
    ax.set_ylabel(YLABEL, fontsize=YLABEL_FONT_SIZE)
    style_x(ax)
    ax.set_xlabel("Token position $n$", fontsize=XLABEL_FONT_SIZE)
    plt.subplots_adjust(left=0.27, right=0.9, top=0.97, bottom=0.2)
    print(f"Saving figure to {fig_path}")
    plt.savefig(fig_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
