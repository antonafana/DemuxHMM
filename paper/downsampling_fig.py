import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import json
import os

sweep_dirs = {
    'DemuxHMM': 'sweeps/downsample_sweep/',
}

num_runs = 10
variable_sweep = 'snp_usage_percent'
common_name = 'Proportion of SNPs Used (1,233 Total)'

variable_values_set = set()
ari_data = {method: {} for method in sweep_dirs}
cells_remaining = {}
emb_remaining = {}
base_cells = {}
base_emb = {}

for method, base_dir in sweep_dirs.items():
    for run_i in range(num_runs):
        print(f'Doing run {run_i}')
        run_dir = os.path.join(base_dir, f'run_{run_i}')
        if not os.path.isdir(run_dir):
            continue

        results_csvs = [f for f in os.listdir(run_dir)
                        if f.startswith('results_run_') and f.endswith('.csv')]

        for csv_file in results_csvs:
            results_path = os.path.join(run_dir, csv_file)
            params_file = os.path.join(run_dir,
                                       csv_file.replace('results_', 'params_').replace('.csv', '.json'))
            if not os.path.exists(params_file):
                continue

            params_json = json.load(open(params_file))
            var_val = params_json[variable_sweep]
            variable_values_set.add(var_val)

            if var_val not in ari_data[method]:
                ari_data[method][var_val] = []
                cells_remaining[var_val] = []
                emb_remaining[var_val] = []
                base_cells[var_val] = []
                base_emb[var_val] = []

            results_df = pd.read_csv(results_path)

            if method == 'DemuxHMM':
                ari_data[method][var_val].append(results_df['hmm_score'].mean())

            cells_remaining[var_val].append(results_df['cells_remaining'].iloc[0])
            emb_remaining[var_val].append(results_df['emb_remaining'].iloc[0])

            base_cells[var_val].append(
                params_json['num_cells_per_org'] * params_json['offspring_per_generation']
            )
            base_emb[var_val].append(params_json['offspring_per_generation'])

variable_values = sorted(variable_values_set)

ari_means, ari_stds = {}, {}
cells_prop, emb_prop = [], []
cells_prop_std, emb_prop_std = [], []

for method in sweep_dirs:
    ari_means[method], ari_stds[method] = [], []
    for v in variable_values:
        vals = ari_data[method].get(v, [])
        ari_means[method].append(np.mean(vals))
        ari_stds[method].append(np.std(vals))

for v in variable_values:
    cells_prop.append(np.mean(cells_remaining[v]) / np.mean(base_cells[v]))
    cells_prop_std.append(np.std(cells_remaining[v]) / np.mean(base_cells[v]))
    emb_prop.append(np.mean(emb_remaining[v] / np.mean(base_emb[v])))
    emb_prop_std.append(np.std(emb_remaining[v]  / np.mean(base_cells[v])))

# An attempt to classify the tradeoff between ARI and remaining embryos
effectiveness = [ari_means['DemuxHMM'][i]*emb_prop[i] for i in range(len(emb_prop))]

cells_prop = np.array(cells_prop)
emb_prop = np.array(emb_prop)

print('Making figure')

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.1)

axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[:, 1])

for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values)
    axA.errorbar(x, y, yerr=yerr, fmt='o-', markersize=7, capsize=4, label=method)

axA.set_ylabel('ARI', fontsize=20)
axA.set_ylim(0, 1)
axA.tick_params(labelsize=16)
axA.set_title('A', loc='left', fontsize=22, fontweight='bold')

print('emb_prop', emb_prop)
axB.errorbar(variable_values, emb_prop, yerr=emb_prop_std, fmt='o-', capsize=4, label='Embryos')
axB.errorbar(variable_values, cells_prop, yerr=cells_prop_std, fmt='s--', capsize=4, label='Cells')
axB.set_ylabel('Proportion Remaining', fontsize=20)
axB.set_ylim(0, 1.05)
axB.tick_params(labelsize=16)
axB.legend(fontsize=12)
axB.set_title('B', loc='left', fontsize=22, fontweight='bold')

# Create a "big" subplot to span the entire grid
ax_joint = fig.add_subplot(111, frameon=False)
# Hide tick and tick labels of the big subplot
ax_joint.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax_joint.grid(False)

# Set the label
ax_joint.set_xlabel(common_name, labelpad=10, fontsize=20)

plt.savefig('figures/snp_downsampling.pdf', bbox_inches='tight')
plt.show()