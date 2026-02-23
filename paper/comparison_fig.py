import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

# Parameters
sweep_dirs = {
    'DemuxHMM': 'sweeps/demuxHMM/',
    'scSplit': 'sweeps/scsplit/',
    'Vireo': 'sweeps/vireo/',
    'souporcell3': 'sweeps/sorc3/',
    'DemuxHMM 2500 UMI': 'sweeps/demuxHMM_umi_2500/',
    'scSplit 2500 UMI': 'sweeps/scsplit_umi_2500/',
}

colormap = {
    'DemuxHMM': 'tab:blue',
    'DemuxHMM 2500 UMI': 'tab:blue',
    'scSplit': 'tab:green',
    'scSplit 2500 UMI': 'tab:green',
    'souporcell3': 'tab:red',
    'Vireo': 'tab:orange',
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
            elif method == 'DemuxHMM 2500 UMI':
                time_data[method][var_val].append(results_df['hmm_time'].mean())
                ari_data[method][var_val].append(results_df['hmm_score'].mean())
            elif method == 'scSplit':
                time_data[method][var_val].append(results_df['scSplit_time'].mean())
                ari_data[method][var_val].append(results_df['scSplit_ARI'].mean())
            elif method == 'scSplit 2500 UMI':
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

print('Variable values', variable_values)
print('ARI means: ', ari_means)

fig = plt.figure(figsize=(10, 20))
# Plot 1: ARI with error bars
ax = plt.subplot(211)
for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values)
    mask = ~np.isnan(y)
    if method == 'scSplit 2500 UMI' or method == 'DemuxHMM 2500 UMI':
        plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='d', linewidth=2,
                     linestyle='-', alpha=0.25, capsize=5, markersize=9, color=colormap[method])
    else:
        plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=2,
                 linestyle='-', alpha=0.8, capsize=5, markersize=9, color=colormap[method])
plt.legend(list(sweep_dirs.keys()), fontsize=23, frameon=False)
plt.title('A', loc='left', fontsize=30, fontweight='bold')
plt.ylabel('ARI', fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', length=10, width=3)
plt.ylim(0, 1)

ax = plt.subplot(212)
# Plot 2: Time with error bars
for method in sweep_dirs:
    # The 2500 UMI runs don't add much info for the time comparison
    if method == 'scSplit 2500 UMI' or method == 'DemuxHMM 2500 UMI':
        continue

    y = np.array(time_means[method])
    yerr = np.array(time_stds[method])
    x = np.array(variable_values)
    mask = ~np.isnan(y)
    plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=2,
                 linestyle='-', alpha=0.8, capsize=5, markersize=9, color=colormap[method])
plt.title('B', loc='left', fontsize=30, fontweight='bold')
plt.ylabel('Time Taken (s)', fontsize=35)
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
ax.tick_params(axis='both', which='major', length=10, width=3)
ax.tick_params(axis='both', which='minor', length=7, width=1)

# Make joint X label
fig.supxlabel(common_name, fontsize=35)
fig.tight_layout()

plt.savefig('figures/methods_comparison_20k_vertical.pdf')
plt.show()