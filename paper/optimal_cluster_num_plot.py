import matplotlib.pyplot as plt
import numpy as np
import figures_utils as fig_utils

# Params for the sweep
sweep_dirs = {
    'Mouse': {'dir': 'sweeps/clusters/demuxHMM_mouse/',
              'prob_key': 'hmm_model_prob'},
    'Simulated': {'dir': 'sweeps/clusters/demuxHMM_simulated/',
                  'prob_key': 'hmm_model_prob'},
}

max_runs = 6
variable_sweep = 'num_clusters'  # The variable we are sweeping over
common_name = 'Number of Clusters'
UMI_level=10000
TEXT_FONT_SIZE = 10

# Data containers
variable_values = {}
model_prob_means  = {}
model_prob_stds   = {}

for method in sweep_dirs:
    # Read model_probs and ARI
    vars_method, aris_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                          variable_sweep, sweep_dirs[method]['prob_key'])
    variable_values[method] = vars_method

    # Compute means and stds
    method_ari_means, method_ari_stds = fig_utils.compute_means_stds(vars_method, aris_method)
    model_prob_means[method] = method_ari_means
    model_prob_stds[method] = method_ari_stds

# Plot: ARI vs Number of Generations with error bars
plt.figure(figsize=(7.5, 3.25))

for i, dataset in enumerate(sweep_dirs):
    plt.subplot(1, 2, i + 1)
    y = np.array(model_prob_means[dataset])
    yerr = np.array(model_prob_stds[dataset])
    x = np.array(variable_values[dataset])
    mask = ~np.isnan(y)

    # Plot data and a vertical line for the true number of clusters
    if dataset == 'Simulated':
        plt.errorbar(x[mask], y[mask], yerr=yerr[mask], fmt='o', markersize=5, alpha=0.8, capsize=5, label=dataset)
        plt.axvline(x=50, color='red', alpha=0.5, linewidth=2)
        plt.text(-0.1, 1.1, 'B)', transform=plt.gca().transAxes,
                 fontsize=TEXT_FONT_SIZE + 2, va='top', ha='left', fontweight='bold')
    elif dataset == 'Mouse':
        plt.scatter(x[mask], y[mask], s=25, alpha=0.8, label=dataset)
        plt.axvline(x=77, color='red', alpha=0.5, linewidth=2)
        plt.text(-0.1, 1.1, 'A)', transform=plt.gca().transAxes,
                 fontsize=TEXT_FONT_SIZE + 2, va='top', ha='left', fontweight='bold')

    plt.xlabel(common_name, fontsize=TEXT_FONT_SIZE)

    if dataset == 'Mouse':
        plt.ylabel('Log Probability ( $J$ )', fontsize=TEXT_FONT_SIZE)
    plt.yticks(fontsize=TEXT_FONT_SIZE)
    plt.xticks(fontsize=TEXT_FONT_SIZE)
    plt.legend(fontsize=TEXT_FONT_SIZE, loc='lower right')

plt.tight_layout()
plt.savefig('figures/clusters_log_prob.pdf')
plt.savefig('journal_figures/Fig9.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()