"""Script to format the FID and Inception scores into a table."""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from glob import glob


from util_experiments import get_base_parser

parser = get_base_parser()
args = parser.parse_args()
output_path = args.output_path

# Get all the csv files in the scores directory
scores_dir = os.path.join(output_path, 'scores')
csv_files = glob(os.path.join(scores_dir, '*.csv'))
print(f"Found {len(csv_files)} csv files in {scores_dir}")
# Read all the csv files
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    # if score is 'inception', then merge 'IS_mean' and 'IS_std' into a single column
    if 'IS_mean' in df.columns and 'IS_std' in df.columns:
        df['value'] = df.apply(lambda row: f"{row['IS_mean']:.2f} ± {row['IS_std']:.2f}", axis=1)
        df = df[['method', 'score', 'value']]
    elif 'fid' in df.columns:
        # rename the 'fid' column to 'value'
        df = df.rename(columns={'fid': 'value'})
    df_list.append(df)

# Concatenate all the dataframes
df = pd.concat(df_list)
# pivot the dataframe by method and score
df = df.pivot(index='method', columns='score', values='value')
# re-order the rows by ['exact', 'reformer', 'performer', 'sblocal', 'kdeformer', 'thinformer']
df = df.reindex(['exact', 'reformer', 'performer', 'sblocal', 'kdeformer', 'thinformer'])
# Print the table
print(tabulate(df, headers='keys', tablefmt='github'))

