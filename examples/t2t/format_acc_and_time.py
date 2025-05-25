"""Script for generating Table 3 in https://arxiv.org/pdf/2502.12063

Example usage:
```bash
python format_acc_and_time.py -op PATH_TO_OUTPUT_DIR
```

NOTE: the output directory should contain the following files:
- `acc/`
- `times/`
"""

import os
import torch
import pandas as pd
from tabulate import tabulate
import glob
from util_experiments import get_base_parser

parser = get_base_parser()
parser.add_argument("--batch_size", "-bs", default=64, type=int, help="batch size")
parser.add_argument(
    "--batch_number", "-bn", default=1, type=int, help="batch number >= 1"
)
parser.add_argument(
    "--num_runs",
    "-n",
    default=1,
    type=int,
    help="number of runs (excluding warm-up runs)",
)
args, opt = parser.parse_known_args()
device = args.device if args.device else torch.device
batch_size = args.batch_size
batch_number = args.batch_number
output_path = args.output_path
num_runs = args.num_runs

methods = [
    "full",
    "performer",
    "reformer",
    "kdeformer",
    "scatterbrain",
    "thinformer",
]
METHOD_ALIASES = {
    "full": "\\Centerstack{\\bf Exact}",
    "performer": "\\Centerstack{\\bf Performer}",
    "reformer": "\\Centerstack{\\bf Reformer}",
    "kdeformer": "\\Centerstack{\\bf KDEformer}",
    "scatterbrain": "\\Centerstack{\\bf Scatterbrain}",
    "thinformer": "\\Centerstack{\\bf Thinformer (Ours)}",
}
COLUMN_ALIASES = {
    "accuracy": "\\Centerstack{\\bf Top-1 Accuracy (\\%)}",
    "attention1.attn.attention_layer": "\\Centerstack{\\bf Layer 1 Runtime (ms)}",
    "attention2.attn.attention_layer": "\\Centerstack{\\bf Layer 2 Runtime (ms)}",
}

# load files from output/acc/
mean_acc = {}
std_acc = {}
for method in methods:
    seeds = []
    # iterate over csv files including method name
    for save_path in glob.glob(
        os.path.join(output_path, "acc", f"acc-{method}-{method}-{device}-s*.csv")
    ):
        print(f"Reading {save_path}")
        df = pd.read_csv(save_path, index_col=0)
        # compute column means
        df_mean = df.mean()
        # add enty for accuracy
        df_mean["accuracy"] = df["corrects"].sum() / df["cnt"].sum() * 100
        # concat to df_all
        seeds.append(df_mean)
    if len(seeds) == 0:
        print(f"No accuracy data found for {method}")
        continue
    # concat all seeds
    df_all = pd.concat(seeds, axis=1).T
    # drop corrects and cnt
    df_all = df_all.drop(columns=["corrects", "cnt"])
    # compute mean and std across all batches and seeds
    mean_acc[method] = df_all.mean()
    std_acc[method] = df_all.std()

# create new dataframe where each entry has the format f"{mean:.2f} ± {std:.2f}"
df_acc = (
    pd.DataFrame(data=mean_acc).apply(lambda x: x.map("{:.2f}".format))
    + " ± "
    + pd.DataFrame(data=std_acc).apply(lambda x: x.map("{:.2f}".format))
)

# load files from output/times/
mean_time = {}
std_time = {}
for method in methods:
    df = pd.DataFrame()
    # iterate over csv files including method name
    # for save_path in glob.glob(os.path.join(output_path, "times", f"times-n{num_runs}-{method}-{device}-bs{batch_size}-bn*.csv")):
    for bn in range(1, 50 + 1):
        save_path = os.path.join(
            output_path,
            "times",
            f"times-n{num_runs}-{method}-{device}-bs{batch_size}-bn{bn}.csv",
        )
        print(f"Reading from {save_path}...")
        # read csv file using pandas
        df = pd.concat(
            [df, pd.read_csv(save_path, index_col=0)], axis=0
        )  # note: times are in milliseconds
    # get mean and std
    mean_time[method] = df.mean()
    std_time[method] = df.std()

df_time = (
    pd.DataFrame(data=mean_time).apply(lambda x: x.map("{:.2f}".format))
    + " ± "
    + pd.DataFrame(data=std_time).apply(lambda x: x.map("{:.2f}".format))
)
# only select the rows with index 'attention1.attn.attention_layer', 'attention2.attn.attention_layer'
df_time = df_time.loc[
    ["attention1.attn.attention_layer", "attention2.attn.attention_layer"]
]

df = pd.concat([df_acc.T, df_time.T], axis=1)
df.columns = [COLUMN_ALIASES[col] for col in df.columns]
df.index = [METHOD_ALIASES[col] for col in df.index]
df = df.reset_index(names=["\\textbf{Attention Algorithm}"])

latex_table = df.to_latex(column_format="c" * len(df.columns), index=False)
# Add [1mm] to all rows except the last one
lines = latex_table.split("\n")
idx = [i for i in range(len(lines)) if "\\\\" in lines[i]]
# drop the first and last line
idx = idx[1:-1]
for i in idx:
    lines[i] = lines[i].replace("\\\\", "\\\\[1mm]")
print("\n".join(lines))
print(tabulate(df, tablefmt="github", headers="keys", showindex=False))
