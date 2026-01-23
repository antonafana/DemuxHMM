import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

# Parameters
sweep_dirs = {
    'DemuxHMM': 'sweeps/demuxHMM/',
    'Vireo': 'sweeps/vireo/',
    'scSplit': 'sweeps/scsplit/',
    'souporcell3': 'sweeps/sorc3/'
}

num_runs = 6
variable_sweep = 'offspring_per_generation'
common_name = 'Number of Individuals'

# Data containers
variable_values_set = set()
time_data = {method: {} for method in sweep_dirs}  # method -> var_val -> list of times
ari_data  = {method: {} for method in sweep_dirs}

# Load results
for method, base_dir in sweep_dirs.items():
    for run_i in range(num_runs):
        run_dir = os.path.join(base_dir, f'run_{run_i}')
        if not os.path.isdir(run_dir):
            continue

        # Find all results CSVs in this run folder
        results_csvs = [f for f in os.listdir(run_dir) if f.startswith('results_run_') and f.endswith('.csv')]
        for csv_file in results_csvs:
            results_path = os.path.join(run_dir, csv_file)
            # Corresponding params file
            params_file = os.path.join(run_dir, csv_file.replace('results_', 'params_').replace('.csv', '.json'))
            if not os.path.exists(params_file):
                continue

            # Load sweep variable
            params_json = json.load(open(params_file, 'r'))
            var_val = params_json[variable_sweep]
            variable_values_set.add(var_val)

            # Initialize containers
            if var_val not in time_data[method]:
                time_data[method][var_val] = []
                ari_data[method][var_val] = []

            # Load CSV
            results_df = pd.read_csv(results_path)

            # Append metrics
            if method == 'Vireo':
                time_data[method][var_val].append(results_df['vireo_time'].mean())
                ari_data[method][var_val].append(results_df['vireo_ARI'].mean())
            elif method == 'DemuxHMM':
                time_data[method][var_val].append(results_df['hmm_time'].mean())
                ari_data[method][var_val].append(results_df['hmm_score'].mean())
            elif method == 'scSplit':
                time_data[method][var_val].append(results_df['scSplit_time'].mean())
                ari_data[method][var_val].append(results_df['scSplit_ARI'].mean())
            elif method == 'souporcell3':
                time_data[method][var_val].append(results_df['souporcell_time'].mean())
                ari_data[method][var_val].append(results_df['souporcell_ARI'].mean())

# Sort variable values
variable_values = sorted(variable_values_set)

# Compute mean & std
time_means = {method: [] for method in sweep_dirs}
time_stds  = {method: [] for method in sweep_dirs}
ari_means  = {method: [] for method in sweep_dirs}
ari_stds   = {method: [] for method in sweep_dirs}

for method in sweep_dirs:
    for var_val in variable_values:
        vals_time = time_data[method].get(var_val, [])
        vals_ari  = ari_data[method].get(var_val, [])
        if vals_time:
            time_means[method].append(np.mean(vals_time))
            time_stds[method].append(np.std(vals_time))
        else:
            time_means[method].append(np.nan)
            time_stds[method].append(np.nan)
        if vals_ari:
            ari_means[method].append(np.mean(vals_ari))
            ari_stds[method].append(np.std(vals_ari))
        else:
            ari_means[method].append(np.nan)
            ari_stds[method].append(np.nan)

# Plot 1: Time with error bars
plt.figure(figsize=(10, 10))
for method in sweep_dirs:
    y = np.array(time_means[method])
    yerr = np.array(time_stds[method])
    x = np.array(variable_values)
    mask = ~np.isnan(y)
    plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=2,
                 linestyle='-', alpha=0.8, capsize=5)
plt.legend(list(sweep_dirs.keys()), fontsize=20)
plt.xlabel(common_name, fontsize=25)
plt.ylabel('Time Taken (s)', fontsize=25)
plt.yscale('log')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('figures/time_taken_errorbars.pdf')
plt.show()

# Plot 2: ARI with error bars
plt.figure(figsize=(10, 10))
for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values)
    mask = ~np.isnan(y)
    plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=2,
                 linestyle='-', alpha=0.8, capsize=5)
plt.legend(list(sweep_dirs.keys()), fontsize=20)
plt.xlabel(common_name, fontsize=25)
plt.ylabel('ARI', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 1)
plt.savefig('figures/methods_comparison_20k_errorbars.pdf')
plt.show()