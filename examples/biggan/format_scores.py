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

# Read all the csv files
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all the dataframes
df = pd.concat(df_list)

# Print the table
print(tabulate(df, headers='keys', tablefmt='github'))

