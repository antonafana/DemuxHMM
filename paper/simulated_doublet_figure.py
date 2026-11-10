import figures_utils as fig_utils
import numpy as np
import matplotlib.pyplot as plt
# Parameters
sweep_dirs = {
    'DemuxHMM': {'dir': 'sweeps/simulated_doublets/demuxHMM/',
                 'ari_key': 'hmm_ari_consensus',
                 'precision_key': 'hmm_doublet_precision',
                 'recall_key': 'hmm_doublet_recall',
                 'tnr_key': 'hmm_tnr',
                 'aupr_key': 'hmm_aupr',
                 'two_pass_ari_key': 'hmm_ari_consensus_doublets_removed'},
    'scSplit': {'dir': 'sweeps/simulated_doublets/scSplit/',
                'ari_key': 'scSplit_ARI',
                'precision_key': 'scSplit_doublet_precision',
                'recall_key': 'scSplit_doublet_recall',
                'tnr_key': 'scSplit_doublet_tnr',
                'aupr_key': None,
                'two_pass_ari_key': 'scSplit_ari_doublets_removed'},
    'Vireo': {'dir': 'sweeps/simulated_doublets/vireo/',
              'ari_key': 'vireo_ARI',
              'precision_key': 'vireo_doublet_precision',
              'recall_key': 'vireo_doublet_recall',
              'tnr_key': 'vireo_tnr',
              'aupr_key': 'vireo_aupr',
              'two_pass_ari_key': 'vireo_ari_doublets_removed'},
    'souporcell3': {'dir': 'sweeps/simulated_doublets/sorc3/',
                     'ari_key': 'souporcell_ARI',
                     'precision_key': 'souporcell_doublet_precision',
                     'recall_key': 'souporcell_doublet_recall',
                     'tnr_key': 'souporcell_tnr',
                     'aupr_key': 'souporcell_aupr',
                     'two_pass_ari_key': 'souporcell_ari_doublets_removed'},
}

colormap = {
    'DemuxHMM': 'tab:blue',
    'scSplit': 'tab:green',
    'Vireo': 'tab:orange',
    'souporcell3': 'tab:red',
}

max_runs = 6
variable_sweep = 'doublet_rate'
common_name = 'Doublet Rate'

# Data containers
variable_values = {}

ari_means = {}
ari_stds = {}
two_pass_means = {}
two_pass_stds = {}
precision_means = {}
precision_stds = {}
recall_means = {}
recall_stds = {}
tnr_means = {}
tnr_stds = {}
aupr_means = {}
aupr_stds = {}

for method in sweep_dirs:
    # Read values for the keys
    vars_method, ari_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                        variable_sweep, sweep_dirs[method]['ari_key'])
    _, two_pass_ari_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                        variable_sweep, sweep_dirs[method]['two_pass_ari_key'])
    _, precision_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                    variable_sweep, sweep_dirs[method]['precision_key'])
    _, recall_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                 variable_sweep, sweep_dirs[method]['recall_key'])
    _, tnr_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                              variable_sweep, sweep_dirs[method]['tnr_key'])

    variable_values[method] = vars_method

    ari_means[method], ari_stds[method] = fig_utils.compute_means_stds(vars_method, ari_method)
    two_pass_means[method], two_pass_stds[method] = fig_utils.compute_means_stds(vars_method, two_pass_ari_method)
    precision_means[method], precision_stds[method] = fig_utils.compute_means_stds(vars_method, precision_method)
    recall_means[method], recall_stds[method] = fig_utils.compute_means_stds(vars_method, recall_method)
    tnr_means[method], tnr_stds[method] = fig_utils.compute_means_stds(vars_method, tnr_method)

    if sweep_dirs[method]['aupr_key'] is not None:
        _, aupr_method = fig_utils.read_sweep_data(
            sweep_dirs[method]['dir'], max_runs, variable_sweep,
            sweep_dirs[method]['aupr_key'])

        aupr_means[method], aupr_stds[method] = fig_utils.compute_means_stds(
            vars_method, aupr_method)
    else:
        aupr_means[method] = [np.nan] * len(vars_method)
        aupr_stds[method] = [np.nan] * len(vars_method)

print('Variable values', variable_values)
print('ARI means', ari_means)

# Plot
metrics = [
    ('Precision', precision_means, precision_stds),
    ('Recall', recall_means, recall_stds),
    ('AUPR', aupr_means, aupr_stds),
    ('ARI Two Pass', two_pass_means, two_pass_stds),
]

doublet_rates = sorted(set(
    x for method in variable_values
    for x in variable_values[method]
))

methods = list(sweep_dirs.keys())

fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), sharey=True)
axes = axes.flatten()

for ax, (metric, metric_means, metric_stds) in zip(axes, metrics):
    for method in methods:
        x_values = np.asarray(variable_values[method])
        means = np.asarray(metric_means[method], dtype=float)
        stds = np.asarray(metric_stds[method], dtype=float)

        mask = ~np.isnan(means)

        ax.errorbar(x_values[mask], means[mask], yerr=stds[mask],
                    marker='o', markersize=4, linewidth=1.5, capsize=3,
                    color=colormap[method], label=method)

    ax.set_xticks(doublet_rates)
    ax.set_xticklabels([f'{100 * x:g}%' for x in doublet_rates], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(metric, fontsize=13)
    ax.tick_params(axis='both', which='major', length=4, width=1, labelsize=9)

axes[0].set_ylabel('Score', fontsize=12)
axes[2].set_ylabel('Score', fontsize=12)

handles = [plt.Line2D([0], [0], marker='o', linewidth=1.5,
               markersize=4, color=colormap[method], label=method)
          for method in methods]

fig.legend(handles, methods, loc='lower center', ncol=4,
           fontsize=10, frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig('figures/doublet_detection_comparison.pdf', bbox_inches='tight')
plt.savefig('journal_figures/Fig8.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()

# Make a new figure for single-pass ARI
fig = plt.figure(figsize=(3.75, 3.75))
ax = plt.gca()

metric_means = ari_means
metric_stds = ari_stds

for method in methods:
    if method == 'scSplit':
        # scSplit handles doublets differently. Not fair to display the worse performance.
        continue

    x_values = np.asarray(variable_values[method])
    means = np.asarray(metric_means[method], dtype=float)
    stds = np.asarray(metric_stds[method], dtype=float)

    mask = ~np.isnan(means)

    ax.errorbar(x_values[mask], means[mask], yerr=stds[mask],
                marker='o', markersize=4, linewidth=1.5, capsize=3,
                color=colormap[method], label=method)

ax.set_xticks(doublet_rates)
ax.set_xticklabels([f'{100 * x:g}%' for x in doublet_rates], fontsize=9)
ax.set_ylim(0, 1)
ax.set_title('Single Pass ARI', fontsize=13)
ax.tick_params(axis='both', which='major', length=4, width=1, labelsize=9)

# Set labels
ax.set_ylabel('ARI', fontsize=12)
ax.set_xlabel('Doublet Rate', fontsize=12)

methods_no_scsplit = [method for method in methods if 'scSplit' != method]
ax.legend(methods_no_scsplit, loc='lower left', ncol=1, fontsize=10, frameon=False)

plt.tight_layout()

plt.savefig('figures/doublet_detection_first_pass_ari.pdf', bbox_inches='tight')
plt.savefig('journal_figures/Fig10.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()