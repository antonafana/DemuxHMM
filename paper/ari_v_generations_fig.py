import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

# Params for the sweep
sweep_dirs = {
    'DemuxHMM': 'sweeps/grid_sweep/',
}

num_runs = 6
variable_sweep = 'num_generations'  # The variable we are sweeping over
common_name = 'Number of Generations'
UMI_level=10000

# Data containers
variable_values_set = set()
ari_data = {method: {} for method in sweep_dirs}  # method -> var_val -> list of ARIs

# Load results
for method, base_dir in sweep_dirs.items():
    for run_i in range(num_runs):
        run_dir = os.path.join(base_dir, f'run_{run_i}')
        if not os.path.isdir(run_dir):
            continue

        # Find all result CSVs in this run folder
        results_csvs = [f for f in os.listdir(run_dir) if f.startswith('results_run_') and f.endswith('.csv')]
        for csv_file in results_csvs:
            results_path = os.path.join(run_dir, csv_file)
            # Corresponding params JSON
            params_file = os.path.join(run_dir, csv_file.replace('results_', 'params_').replace('.csv', '.json'))
            if not os.path.exists(params_file):
                continue

            # Load sweep variable
            params_json = json.load(open(params_file, 'r'))
            # This is a zoomed in view to a level of the grid sweep at 10k UMI
            if params_json['avg_UMI'] != UMI_level:
                continue

            var_val = params_json[variable_sweep]
            variable_values_set.add(var_val)

            # Initialize container
            if var_val not in ari_data[method]:
                ari_data[method][var_val] = []

            # Load results CSV
            results_df = pd.read_csv(results_path)

            # Append ARI
            if method == 'Vireo':
                ari_data[method][var_val].append(results_df['vireo_ARI'].mean())
            elif method == 'DemuxHMM':
                ari_data[method][var_val].append(results_df['hmm_score'].mean())
            elif method == 'scSplit':
                ari_data[method][var_val].append(results_df['scSplit_ARI'].mean())
            elif method.lower() == 'souporcell':
                ari_data[method][var_val].append(results_df['souporcell_ARI'].mean())

# Sort variable values
variable_values = sorted(variable_values_set)

# Compute mean & std
ari_means = {method: [] for method in sweep_dirs}
ari_stds  = {method: [] for method in sweep_dirs}

for method in sweep_dirs:
    for var_val in variable_values:
        vals = ari_data[method].get(var_val, [])
        if vals:
            ari_means[method].append(np.mean(vals))
            ari_stds[method].append(np.std(vals))
        else:
            ari_means[method].append(np.nan)
            ari_stds[method].append(np.nan)

# Plot: ARI vs Number of Generations with error bars
plt.figure(figsize=(10, 10))
for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values)
    mask = ~np.isnan(y)
    plt.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt='o', markersize=10, alpha=0.8, capsize=5, label=method)

plt.xlabel(common_name, fontsize=25)
plt.ylabel('ARI', fontsize=25)
plt.xticks([2*i for i in range(13)], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0.6, 1)
plt.legend(fontsize=20)
plt.savefig('figures/ari_v_generations.pdf')
plt.show()