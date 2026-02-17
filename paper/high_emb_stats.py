import pandas as pd
import os
import numpy as np

# Parameters
sweep_dirs = {
    'DemuxHMM': 'sweeps/high_emb/',
}

num_runs = 3

# Data containers
time_data = {method: [] for method in sweep_dirs} 
ari_data  = {method: [] for method in sweep_dirs}
emb_data = {method: [] for method in sweep_dirs}

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

            # Load CSV
            results_df = pd.read_csv(results_path)
            print(results_df)

            # Append metrics
            if method == 'DemuxHMM':
                time_data[method].append(results_df['hmm_time'].mean())
                ari_data[method].append(results_df['hmm_score'].mean())
                emb_data[method].append(results_df['emb_remaining'].mean())

# Compute mean & std
time_means = {method: np.mean(time_data[method]) for method in sweep_dirs}
time_stds  = {method: np.std(time_data[method]) for method in sweep_dirs}
ari_means  = {method: np.mean(ari_data[method]) for method in sweep_dirs}
ari_stds   = {method: np.std(ari_data[method]) for method in sweep_dirs}
emb_means = {method: np.mean(emb_data[method]) for method in sweep_dirs}
emb_stds = {method: np.std(emb_data[method]) for method in sweep_dirs}

# Conversion factor for seconds to hours
factor = 1/60*1/60
time_means = {method: time_means[method] * factor for method in sweep_dirs}
time_stds = {method: time_stds[method] * factor for method in sweep_dirs}

print('ARI means: ', ari_means)
print('ARI stds: ', ari_stds)
print('Time means (hrs): ', time_means)
print('Time stds (hrs): ', time_stds)
print('Embryo means: ', emb_means)
print('Embryo stds: ', emb_stds)