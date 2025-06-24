"""Script to format runtime measurements into a table."""

import os
import pandas as pd
from glob import glob
from tabulate import tabulate

from util_experiments import get_base_parser

parser = get_base_parser()
args = parser.parse_args()
output_path = args.output_path

# Get all the times CSV files
times_dir = os.path.join(output_path, 'times')
csv_files = glob(os.path.join(times_dir, '*.csv'))
print(f"Found {len(csv_files)} time measurement files in {times_dir}")

# Read and process all the CSV files
method_times = {}
for file in csv_files:
    # Extract method name from filename
    method = file.split('-')[-4]  # This gets the method name from the filename pattern
    df = pd.read_csv(file, index_col=0)
    
    # Only get attention-matrix timing
    if 'attention-matrix' in df.columns:
        mean_time = df['attention-matrix'].mean()
        std_time = df['attention-matrix'].std()
        method_times[method] = {
            'mean': mean_time,
            'std': std_time
        }

# Create the results table
methods = ['exact', 'reformer', 'performer', 'sblocal', 'kdeformer', 'thinformer']
results = []
for method in methods:
    if method in method_times:
        mean = method_times[method]['mean']
        std = method_times[method]['std']
        results.append({
            'Method': method,
            'Time (ms)': f"{mean:.2f} ± {std:.2f}"
        })

# Create and save the table
table_df = pd.DataFrame(results)
table_path = os.path.join(output_path, 'times', 'attention_matrix_times.csv')
table_df.to_csv(table_path, index=False)
print(f"Saved table to {table_path}")

# Print the table in a nice format
print("\nAttention Matrix Runtime Measurements (ms):")
print(tabulate(table_df, headers='keys', tablefmt='github', showindex=False)) 