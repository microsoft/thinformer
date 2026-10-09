"""Script to plot the loss at every position in the context window.

Example usage:
```bash
python plot_loss.py --start 0 --end 32767 -wi 201 -p 2
```
"""

import os
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from argparse import ArgumentParser

parser = ArgumentParser()
# Arguments from benchmark_patch_llm.py
parser.add_argument(
    "--patch_config", type=str, default="last", choices=["last", "first", "even", "odd"]
)
parser.add_argument("--num_patch_layers", type=int, default=-1)
# Extra arguments
parser.add_argument("--output_dir", "-od", type=str, default="out")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--window_length", "-wi", type=int, default=None)
parser.add_argument("--polyorder", "-p", type=int, default=None)

args = parser.parse_args()
start = args.start
end = args.end if args.end is not None else start + 1024
window_length = args.window_length
polyorder = args.polyorder
if window_length is not None and polyorder is None:
    print("If window_length is specified, polyorder must also be specified.")
    exit(1)
if polyorder is not None and window_length is None:
    print("If polyorder is specified, window_length must also be specified.")
    exit(1)
num_patch_layers = args.num_patch_layers

output_dir = args.output_dir
loss_dir = os.path.join(output_dir, "loss")
figure_dir = os.path.join(output_dir, "figures")
# os.makedirs(loss_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

COLORS = {
    "flash": "grey",
    "exact": "black",
    "hyper-cuda": "orange",
    "thinformer_triton": "green",
}
for method in [
    "flash",
    "exact",
    "hyper-cuda",
    "thinformer_triton",
]:
    loss_path = os.path.join(loss_dir, f"loss_{num_patch_layers}_{method}.pt")
    if not os.path.exists(loss_path):
        print(f"File {loss_path} does not exist. Skipping {method}.")
        continue
    loss = torch.load(loss_path).cpu().numpy().flatten()
    # import pdb; pdb.set_trace()
    # plot loss versus position

    arr = loss[start:end]
    raw_mean = arr.mean()
    if window_length is not None and polyorder is not None:
        # apply savgol filter to smooth the loss
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        # "nearest" mode uses the nearest value for padding
        arr = savgol_filter(arr, window_length, polyorder, mode="nearest")

    plt.plot(range(start, end), arr, color=COLORS[method], alpha=0.5, linewidth=1)
    # add a horizontal line for the mean
    plt.axhline(
        y=raw_mean,
        color=COLORS[method],
        linestyle="--",
        label=f"{method}: {raw_mean}",
    )

n_first = 4
# add vertical line at multiples of 4096
for i in range(n_first + 4096, end, n_first + 4096):
    if start <= i <= end:
        plt.axvline(x=i, color="g", linestyle="-")
# add vertical lines at multiples of 1024 starting from 4096
for i in range(n_first + 4096, end, 1024):
    if start <= i <= end:
        plt.axvline(x=i, color="r", linestyle=":", alpha=0.2)
# get axes
ax = plt.gca()
# set xticks to multiples of increment
increment = 4096
xticks = list(range((start // increment) * increment, end + 1, increment))
# add start and end
xticks = [start] + xticks + [end]
minor_xticks = list(range((start // 256) * 256, end + 1, 256))
ax.set_xticks(xticks)
ax.set_xticks(minor_xticks, minor=True)
ax.tick_params(axis="x", which="major")
ax.tick_params(axis="x", which="minor")
ax.set_xticklabels(xticks, fontsize=10, rotation=45)

ax.set_xlabel("Position in Context Window")

ax.set_xlim(start, end)
ax.legend()

fig_path = os.path.join(
    figure_dir,
    f"loss_{num_patch_layers}_{start}_{end}_smoothed_wi{window_length}_p{polyorder}.pdf",
)
plt.savefig(fig_path)
print(f"Saved figure to {fig_path}")
