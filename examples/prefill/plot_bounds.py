"""Plot the two attention-error bound factors of one recorded layer as a
function of the token position n, each as the median over (sequence, head)
with a 25%-75% interquartile band:

    rho_n = n / D_nn * exp(s (||q_n||^2 + r_n^2) / 2) * vmax_n
    exp(2 R_n^2 / sqrt(d)) * vmax_n

with
    D_nn   = sum_{i<=n} exp(s <q_n, k_i>)        causal softmax denominator
    r_n    = max_{i<=n} ||k_i||_2                 running max key norm
    R_n    = max_{i<=n} max(||q_i||_2, ||k_i||_2) running max query/key norm
    vmax_n = max_{i<=n} ||v_i||_inf               running max value inf-norm
    s      = the layer's softmax temperature (1/sqrt(d) for Chameleon)

Centering follows the chatglm dumps of the paper: --center_keys subtracts
the per-head full-sequence mean key before r_n, R_n and D_nn (softmax weights
are invariant to this); --center_values subtracts the per-head, per-coordinate
midrange (max+min)/2 before the inf-norm.

Besides the per-position head medians, the tables also give the median
pooled over all heads and all positions i <= n.

Data layout (written by benchmark_patch_llm.py --record_qkv_layer L -od <output_dir>):
    <output_dir>/states/states__model=<slug>_layer=<L>_..._bid=<i>.pt
Derived per-head curves are cached under <output_dir>/states/derived/ and
figures go to <output_dir>/figures/ as <out>[_ckeys]_rho[_cvals]<ext>
and <out>[_ckeys]_expR2_vmax[_cvals]<ext>.

Usage (run from examples/prefill after benchmark_patch_llm.py has written states/):
    python plot_bounds.py -od out --pattern "states__model=facebook-chameleon-7b_layer=0_*bid=0.pt" --out cham_l0/bounds.pdf --center_keys --center_values

Options:
    --output_dir DIR  same -od as benchmark_patch_llm.py (default: out)
    --pattern GLOB    glob (within the states dir) selecting one model/layer
    --out PATH        figure name, relative to <output_dir>/figures (default: bounds.pdf)
    --center_keys     center keys (affects r_n, R_n, D_nn)
    --center_values   center values (affects vmax_n)
    --chunk K         n-columns per quantile pass (default: 2048)
"""
import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from plotting_utils import (
    XLABEL_FONT_SIZE,
    YLABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    LINE_ALPHA,
    LINE_WIDTH,
    OUTWARD,
    NEURIPS_WIDTH,
    NEURIPS_HEIGHT,
)
from states_utils import find_state_files, load_derived, chunked_quantiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", "--output-dir", "-od", default="out",
                    help="benchmark output dir: reads <output_dir>/states, writes <output_dir>/figures")
    ap.add_argument("--pattern", default="states__*.pt",
                    help="glob within <output_dir>/states selecting the files to aggregate")
    ap.add_argument("--out", default="bounds.pdf", help="figure name relative to <output_dir>/figures")
    ap.add_argument("--center_keys", action="store_true", help="center keys by their per-head mean")
    ap.add_argument("--center_values", action="store_true", help="center values by their per-head midrange")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    data_dir = os.path.join(args.output_dir, "states")
    files = find_state_files(data_dir, args.pattern)
    if not files:
        raise SystemExit(f"No files matching {args.pattern} found under {data_dir}")
    print(f"Found {len(files)} states files under {data_dir}")

    # Assembled in log space for numerical safety, exponentiated before the
    # quantiles so that the medians (the mean of the two middle heads for an
    # even head count) are taken on the quantities themselves.
    rows_rho, rows_exp = [], []
    meta = None
    for f in files:
        zk = load_derived(f, center_keys=args.center_keys)
        zv = load_derived(f, center_values=args.center_values)
        s, d = float(zk["scaling"]), int(zk["d"])
        q, r, Rq, ld = (zk[k].astype(np.float64) for k in ("q_norm", "Rk", "Rq", "log_Dnn"))
        H, N = q.shape
        logn = np.log(np.arange(1, N + 1, dtype=np.float64))[None, :]
        log_rho = logn - ld + 0.5 * s * (q ** 2 + r ** 2)
        R = np.maximum(Rq, r)
        log_exp = 2.0 * R ** 2 / np.sqrt(d)
        vmax = np.repeat(zv["vmax"].astype(np.float64), H // zv["vmax"].shape[0], axis=0)  # kv head -> query heads
        log_v = np.log(vmax)
        rows_rho.append(log_rho + log_v)
        rows_exp.append(log_exp + log_v)
        if meta is None:
            meta = dict(s=s, d=d, H=H, N=N)
    arr_rho = np.exp(np.concatenate(rows_rho, axis=0))
    arr_exp = np.exp(np.concatenate(rows_exp, axis=0))
    N = meta["N"]
    print(f"  {arr_rho.shape[0]} rows (seq*head), n={N}, head_dim={meta['d']}, s={meta['s']:.6g}, "
          f"center_keys={args.center_keys}, center_values={args.center_values}")

    r25, r50, r75 = chunked_quantiles(arr_rho, args.chunk)
    e25, e50, e75 = chunked_quantiles(arr_exp, args.chunk)

    grid = [4 ** k for k in range(int(np.log(N) / np.log(4)) + 1)]
    if grid[-1] != N:
        grid.append(N)

    def print_table(name, arr, lo, mid, hi):
        rows = [[g, f"{mid[g - 1]:.0f}", f"{lo[g - 1]:.0f}", f"{hi[g - 1]:.0f}",
                 f"{np.median(arr[:, :g]):.0f}"] for g in grid]
        print(f"\n## {name}")
        print(tabulate(rows, headers=["n", "head median", "25%", "75%", "pooled median (heads, i<=n)"], tablefmt="github"))

    print_table("rho_n = n/D_nn * exp(s(||q_n||^2 + r_n^2)/2) * vmax_n", arr_rho, r25, r50, r75)
    print_table("exp(2 R_n^2 / sqrt(d)) * vmax_n", arr_exp, e25, e50, e75)

    n = np.arange(1, N + 1)

    def render_one(path, lo, mid, hi, color, ylabel):
        fig, ax = plt.subplots(figsize=(NEURIPS_WIDTH / 2, NEURIPS_HEIGHT))
        ax.plot(n, mid, color=color, linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label="median")
        ax.fill_between(n, lo, hi, color=color, alpha=0.3, label="25%-75% quantiles")
        ax.set_yscale("log")
        ax.spines["bottom"].set_position(("outward", OUTWARD))
        ax.spines["left"].set_position(("outward", OUTWARD))
        ax.grid(True, linestyle=":", alpha=0.5)
        xticks = []
        t = 1024
        while t <= N:
            xticks.append(t)
            t *= 2
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x // 1024}k" for x in xticks], fontsize=TICK_FONT_SIZE)
        offset = matplotlib.transforms.ScaledTranslation(-0.05, 0, fig.dpi_scale_trans)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text():
                tick_label.set_transform(tick_label.get_transform() + offset)
                break
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.set_xlabel("Token position $n$", fontsize=XLABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=YLABEL_FONT_SIZE)
        plt.subplots_adjust(right=0.95, top=0.95, bottom=0.2, left=0.2)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    root, ext = os.path.splitext(os.path.join(args.output_dir, "figures", args.out))
    if args.center_keys:
        root += "_ckeys"
    vsuf = "_cvals" if args.center_values else ""
    vlab = r"\max_{i\leq n}\|v_i-\bar v\|_\infty" if args.center_values else r"\max_{i\leq n}\|v_i\|_\infty"
    render_one(f"{root}_rho{vsuf}{ext}", r25, r50, r75, "C6", r"$\rho_n$")
    render_one(f"{root}_expR2_vmax{vsuf}{ext}", e25, e50, e75, "C1", rf"$e^{{2R_n^2/\sqrt d}}\cdot{vlab}$")


if __name__ == "__main__":
    main()
