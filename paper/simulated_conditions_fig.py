import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import figures_utils as fig_utils

# Grid sweep
GRID_DATA_DIR = "sweeps/simulated_grid_sweep/"
GRID_PARAM_X = "avg_UMI"
GRID_PARAM_Y = "num_generations"
GRID_PERF_KEY = "hmm_ari_consensus"
GRID_PARAM_X_LABEL = "UMI Per Cell"
GRID_PARAM_Y_LABEL = "Number of Generations"
GRID_REPEATS = 6

# SNP downsampling
SNP_DS_DIR = 'sweeps/simulated_snp_downsample_sweep/'
NUM_RUNS_SNP_DS = 10
SNP_DS_PERF_KEY = 'hmm_ari_consensus'
variable_sweep = 'snp_usage_percent'
common_name = 'Proportion of SNPs Used (1,233 Total)'

# Calculate the number of starting cells before filtering / downsampling
OFFSPRING_PER_GENERATION = 100
NUM_CELLS_PER_ORG = 250
BASE_CELLS  = OFFSPRING_PER_GENERATION * NUM_CELLS_PER_ORG

# Font settings for readability in LaTeX
FONT_SIZE = 13
PANEL_LABEL_SIZE = 12
LABEL_SIZE = 13
TICK_SIZE = 10
ANNOT_SIZE = 9
LEGEND_SIZE = 10
FIGURE_SIZE = (7.5, 6.5)

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "figure.titlesize": FONT_SIZE
})

# Collect all grid sweep runs
records = []

for i in range(GRID_REPEATS):
    repeat_dir = os.path.join(GRID_DATA_DIR, f"run_{i}/")
    for filename in os.listdir(repeat_dir):
        if filename.startswith("params_run_") and filename.endswith(".json"):
            run_id = filename[len("params_run_"):-len(".json")]
            json_path = os.path.join(repeat_dir, filename)
            csv_path = os.path.join(repeat_dir, f"results_run_{run_id}.csv")

            if not os.path.exists(csv_path):
                continue

            # Read params
            with open(json_path, "r") as f:
                params = json.load(f)

            # Read results
            df_res = pd.read_csv(csv_path)

            hmm_score = df_res.loc[0, GRID_PERF_KEY] if GRID_PERF_KEY in df_res.columns else None

            records.append({
                GRID_PARAM_X: float(params[GRID_PARAM_X]),
                GRID_PARAM_Y: float(params[GRID_PARAM_Y]),
                GRID_PERF_KEY: hmm_score,
            })

# Convert to DataFrame
df = pd.DataFrame(records)
df = df.groupby([GRID_PARAM_Y, GRID_PARAM_X], as_index=False).agg(mean_hmm_score=(GRID_PERF_KEY, "mean"))

# Pivot for heatmaps (numeric sort of both axes)
hmm_pivot = df.pivot(index=GRID_PARAM_Y, columns=GRID_PARAM_X, values="mean_hmm_score")
hmm_pivot = hmm_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

# Integer tick labels if possible
x_vals = [int(v) if abs(v - round(v)) < 1e-9 else v for v in hmm_pivot.columns]
y_vals = [int(v) if abs(v - round(v)) < 1e-9 else v for v in hmm_pivot.index]

# Read SNP downsampling sweep
variable_values, emb_remaining = fig_utils.read_sweep_data(SNP_DS_DIR, NUM_RUNS_SNP_DS,
                                                           variable_sweep, 'emb_remaining')
_, cells_remaining = fig_utils.read_sweep_data(SNP_DS_DIR, NUM_RUNS_SNP_DS,
                                               variable_sweep, 'cells_remaining')
_, ari_data = fig_utils.read_sweep_data(SNP_DS_DIR, NUM_RUNS_SNP_DS,
                                        variable_sweep, SNP_DS_PERF_KEY)

# Get means and std for ARI
ari_means, ari_stds = fig_utils.compute_means_stds(variable_values, ari_data)

cells_prop, emb_prop = [], []
cells_prop_std, emb_prop_std = [], []

for v in variable_values:
    cells_prop.append(np.mean(cells_remaining[v]) / BASE_CELLS)
    cells_prop_std.append(np.std(cells_remaining[v]) / BASE_CELLS)
    emb_prop.append(np.mean(emb_remaining[v]) / OFFSPRING_PER_GENERATION)
    emb_prop_std.append(np.std(emb_remaining[v]) / OFFSPRING_PER_GENERATION)

cells_prop = np.array(cells_prop)
emb_prop = np.array(emb_prop)

# Making figure
fig = plt.figure(figsize=FIGURE_SIZE)
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3, top=0.95)

# Create a "big" subplot to span the entire top row
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# Plot A: Grid sweep
sns.heatmap(hmm_pivot, ax=axA, cmap="viridis", annot=True, fmt=".2f",
            vmin=0, vmax=1, cbar=True,
            xticklabels=x_vals, yticklabels=y_vals,
            annot_kws={"size": ANNOT_SIZE},
            cbar_kws={"label": "ARI", "shrink": 0.85})

axA.set_xlabel(GRID_PARAM_X_LABEL, fontsize=LABEL_SIZE)
axA.set_ylabel(GRID_PARAM_Y_LABEL, fontsize=LABEL_SIZE)
axA.tick_params(labelsize=TICK_SIZE)
axA.set_title('A', loc='left', fontsize=PANEL_LABEL_SIZE, fontweight='bold')

# Plot B: SNP downsampling filtering
axB.errorbar(variable_values, emb_prop, yerr=emb_prop_std,
             fmt='o--', markersize=5, capsize=3,
             label='Embryos', color='tab:purple', alpha=0.8)
print(f'Embryos remaining are {emb_prop}')

axB.errorbar(variable_values, cells_prop, yerr=cells_prop_std,
             fmt='s--', markersize=5, capsize=3,
             label='Cells', color='tab:cyan', alpha=0.8)

axB.set_ylabel('Proportion Remaining', fontsize=LABEL_SIZE)
axB.set_ylim(0, 1.05)
axB.set_xlim(0, 1)
axB.tick_params(labelsize=TICK_SIZE)
axB.legend(fontsize=LEGEND_SIZE)
axB.set_title('B', loc='left', fontsize=PANEL_LABEL_SIZE, fontweight='bold')

# Plot C: SNP downsampling performance
y = np.array(ari_means)
yerr = np.array(ari_stds)
x = np.array(variable_values)
axC.errorbar(x, y, yerr=yerr, fmt='o-', markersize=5, capsize=3, label='DemuxHMM')

axC.set_ylim(0, 1)
axC.set_xlim(0, 1)
axC.tick_params(labelsize=TICK_SIZE)
axC.set_ylabel('ARI', fontsize=LABEL_SIZE)
axC.set_title('C', loc='left', fontsize=PANEL_LABEL_SIZE, fontweight='bold')

# Common label for lower x axis
fig.text(0.5, 0.04, common_name, ha='center', fontsize=LABEL_SIZE)

plt.savefig('figures/simulated_conditions.pdf')
plt.savefig('journal_figures/Fig7.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()