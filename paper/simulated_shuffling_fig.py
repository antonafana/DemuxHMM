import matplotlib.pyplot as plt
import numpy as np
import figures_utils as fig_utils

sweep_dirs = {
    'DemuxHMM': {'dir': 'sweeps/simulated_snp_shuffle_Z_200/',
                 'ari_key': 'hmm_ari_consensus'},
    'Vireo': {'dir': 'sweeps/simulated_snp_shuffle_Z_200/',
              'ari_key': 'vireo_ARI'}
}

# For mouse dataset
# sweep_dirs = {
#     'DemuxHMM': {'dir': 'sweeps/DO_mouse_shuffle/demuxHMM/',
#                  'ari_key': 'hmm_score_consensus'},
#     'Vireo': {'dir': 'sweeps/DO_mouse_shuffle/vireo/',
#               'ari_key': 'vireo_homog_score'}
# }

max_runs = 10
variable_sweep = 'snp_shuffle_percent'
common_name = 'Proportion of SNPs Shuffled'

# Data containers
variable_values = {}
ari_means  = {}
ari_stds   = {}

for method in sweep_dirs:
    # Read ARI
    vars_method, aris_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                          variable_sweep, sweep_dirs[method]['ari_key'])
    variable_values[method] = vars_method

    method_ari_means, method_ari_stds = fig_utils.compute_means_stds(vars_method, aris_method)
    ari_means[method] = method_ari_means
    ari_stds[method] = method_ari_stds


fig = plt.figure(figsize=(6, 6))

for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values[method])
    plt.errorbar(x, y, yerr=yerr, fmt='o-', markersize=7, capsize=4, label=method)

plt.ylim(0, 1)
plt.xlim(0, 1)
plt.gca().tick_params(labelsize=16)

plt.ylabel('ARI', fontsize=20)
plt.xlabel('SNP shuffle percentage', fontsize=20)
plt.legend(sweep_dirs.keys(), fontsize=20)

plt.tight_layout()
plt.savefig('figures/snp_shuffling.pdf', bbox_inches='tight')
plt.show()