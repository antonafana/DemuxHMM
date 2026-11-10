import matplotlib.pyplot as plt
import numpy as np
import figures_utils as fig_utils
from matplotlib.gridspec import GridSpec

# Parameters
sweep_dirs = {
    'DemuxHMM': {'dir': 'sweeps/simulated_demuxHMM',
                 'time_key': 'hmm_time',
                 'ari_key': 'hmm_ari_consensus',
                 'genotype_key': 'hmm_genotype_error'},
    'scSplit': {'dir': 'sweeps/simulated_scsplit/',
                 'time_key': 'scSplit_time',
                 'ari_key': 'scSplit_ARI'},
    'Vireo': {'dir': 'sweeps/simulated_vireo/',
              'time_key': 'vireo_time',
              'ari_key': 'vireo_ARI',
              'genotype_key': 'vireo_genotype_error'},
    'souporcell3': {'dir': 'sweeps/simulated_sorc3/',
                     'time_key': 'souporcell_time',
                     'ari_key': 'souporcell_ARI'},
    'DemuxHMM 2500 UMI': {'dir': 'sweeps/simulated_demuxHMM_umi_2500',
                         'time_key': 'hmm_time',
                         'ari_key': 'hmm_ari_consensus'},
    'scSplit 2500 UMI': {'dir': 'sweeps/simulated_scsplit_umi_2500/',
                         'time_key': 'scSplit_time',
                         'ari_key': 'scSplit_ARI'},
}

# Parameters for SNP shuffling
sweep_dirs_shuffle = {
    'DemuxHMM': {'dir': 'sweeps/simulated_snp_shuffle_Z_200/',
                 'ari_key': 'hmm_ari_consensus'},
    'Vireo': {'dir': 'sweeps/simulated_snp_shuffle_Z_200/',
              'ari_key': 'vireo_ARI'}
}

max_runs_shuffle = 10
variable_sweep_shuffle = 'snp_shuffle_percent'
common_name_shuffle = 'Proportion of SNPs Shuffled'

colormap = {
    'DemuxHMM': 'tab:blue',
    'DemuxHMM 2500 UMI': 'tab:blue',
    'scSplit': 'tab:green',
    'scSplit 2500 UMI': 'tab:green',
    'souporcell3': 'tab:red',
    'Vireo': 'tab:orange',
}

max_runs = 6
variable_sweep = 'offspring_per_generation'
common_name = 'Number of Individuals'

# Data containers
variable_values = {}

time_means = {}
time_stds  = {}
ari_means  = {}
ari_stds   = {}

# Read times and ARI
for method in sweep_dirs:
    vars_method, times_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                          variable_sweep, sweep_dirs[method]['time_key'])
    _, aris_method = fig_utils.read_sweep_data(sweep_dirs[method]['dir'], max_runs,
                                                          variable_sweep, sweep_dirs[method]['ari_key'])
    variable_values[method] = vars_method

    # Compute means and stds
    method_time_means, method_time_stds = fig_utils.compute_means_stds(vars_method, times_method)
    time_means[method] = method_time_means
    time_stds[method] = method_time_stds

    method_ari_means, method_ari_stds = fig_utils.compute_means_stds(vars_method, aris_method)
    ari_means[method] = method_ari_means
    ari_stds[method] = method_ari_stds

print('Variable values', variable_values)
print('ARI means: ', ari_means)

# Read filtering results
demuxhmm_filter_dir = sweep_dirs['DemuxHMM']['dir']

_, emb_remaining = fig_utils.read_sweep_data(demuxhmm_filter_dir, max_runs,
                                             variable_sweep, 'emb_remaining')
_, cells_remaining = fig_utils.read_sweep_data(demuxhmm_filter_dir, max_runs,
                                               variable_sweep, 'cells_remaining')

emb_remaining_means, emb_remaining_stds = fig_utils.compute_means_stds(variable_values['DemuxHMM'],
                                                                       emb_remaining)
cells_remaining_means, cells_remaining_stds = fig_utils.compute_means_stds(variable_values['DemuxHMM'],
                                                                            cells_remaining)

# Convert to percentages
emb_remaining_means = 100 * np.array(emb_remaining_means) / np.array(variable_values['DemuxHMM'])
emb_remaining_stds = 100 * np.array(emb_remaining_stds) / np.array(variable_values['DemuxHMM'])

cells_remaining_means = 100 * np.array(cells_remaining_means) / 20000
cells_remaining_stds = 100 * np.array(cells_remaining_stds) / 20000

# Read SNP shuffling results
variable_values_shuffle = {}
ari_means_shuffle = {}
ari_stds_shuffle = {}

for method in sweep_dirs_shuffle:
    vars_method, aris_method = fig_utils.read_sweep_data(sweep_dirs_shuffle[method]['dir'], max_runs_shuffle,
                                                         variable_sweep_shuffle, sweep_dirs_shuffle[method]['ari_key'])
    variable_values_shuffle[method] = vars_method

    method_ari_means, method_ari_stds = fig_utils.compute_means_stds(vars_method, aris_method)
    ari_means_shuffle[method] = method_ari_means
    ari_stds_shuffle[method] = method_ari_stds

# Read HMM genotype errors
(vars_genotype, genotype_errors) = fig_utils.read_sweep_data(sweep_dirs['DemuxHMM']['dir'], max_runs,
                                                             variable_sweep, sweep_dirs['DemuxHMM']['genotype_key'])
means_genotype, stds_genotype = fig_utils.compute_means_stds(vars_genotype, genotype_errors)

# Read vireo genotype errors
(vars_genotype_vireo, genotype_errors_vireo) = fig_utils.read_sweep_data(sweep_dirs['Vireo']['dir'], max_runs,
                                                             variable_sweep, sweep_dirs['Vireo']['genotype_key'])
means_genotype_vireo, stds_genotype_vireo = fig_utils.compute_means_stds(vars_genotype_vireo, genotype_errors_vireo)

fig = plt.figure(figsize=(7.5, 7.5))
gs = GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.8])

LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
MARKER_SIZE = 3
LINE_WIDTH = 1
CAP_SIZE = 2.5
MAJOR_TICK_LENGTH = 5
MAJOR_TICK_WIDTH = 1
MINOR_TICK_LENGTH = MAJOR_TICK_LENGTH / 2
MINOR_TICK_WIDTH = MAJOR_TICK_WIDTH / 2

# Plot 1: ARI with error bars
ax_ari = fig.add_subplot(gs[:2, 0])
for method in sweep_dirs:
    y = np.array(ari_means[method])
    yerr = np.array(ari_stds[method])
    x = np.array(variable_values[method])
    mask = ~np.isnan(y)
    if method == 'scSplit 2500 UMI' or method == 'DemuxHMM 2500 UMI':
        ax_ari.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='d', linewidth=LINE_WIDTH,
                        linestyle='-', alpha=0.25, capsize=CAP_SIZE, markersize=MARKER_SIZE,
                        color=colormap[method], label=method)
    else:
        ax_ari.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=LINE_WIDTH,
                        linestyle='-', alpha=0.8, capsize=CAP_SIZE, markersize=MARKER_SIZE,
                        color=colormap[method], label=method)

ax_ari.set_title('A', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax_ari.set_ylabel('ARI', fontsize=LABEL_FONT_SIZE)
ax_ari.tick_params(axis='both', which='major', length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, labelsize=TICK_FONT_SIZE)
ax_ari.set_ylim(0, 1)

# Place the legend in the next column over
ax_legend = fig.add_axes([0.52, 0.8, 0.43, 0.15])
ax_legend.axis('off')
handles, labels = ax_ari.get_legend_handles_labels()
ax_legend.legend(handles, labels, fontsize=LEGEND_FONT_SIZE,
                 frameon=False, loc='upper left',
                 ncol=2, columnspacing=0.5)

# Plot 2: Time with error bars
ax_time = fig.add_subplot(gs[2:4, 0])

for method in sweep_dirs:
    # The 2500 UMI runs don't add much info for the time comparison
    if method == 'scSplit 2500 UMI' or method == 'DemuxHMM 2500 UMI':
        continue

    y = np.array(time_means[method])
    yerr = np.array(time_stds[method])
    x = np.array(variable_values[method])
    mask = ~np.isnan(y)
    ax_time.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=LINE_WIDTH,
                 linestyle='-', alpha=0.8, capsize=CAP_SIZE, markersize=MARKER_SIZE, color=colormap[method])
ax_time.set_title('B', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax_time.set_ylabel('Time Taken (s)', fontsize=LABEL_FONT_SIZE)
ax_time.set_yscale('log')
ax_time.tick_params(axis='both', which='major', length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH)
ax_time.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH, width=MINOR_TICK_WIDTH)

# Plot 4: Percentage of cells and embryos remaining after filtering
ax_filter = fig.add_subplot(gs[4, 0])
x = np.array(variable_values['DemuxHMM'])

ax_filter.errorbar(x, emb_remaining_means, yerr=emb_remaining_stds,
                   linewidth=LINE_WIDTH, fmt='o--', alpha=0.8,
                   capsize=0, markersize=MARKER_SIZE, color='tab:purple',
                   label='Embryos')

ax_filter.errorbar(x, cells_remaining_means, yerr=cells_remaining_stds,
                   linewidth=LINE_WIDTH, fmt='s--', alpha=0.8,
                   capsize=0, markersize=MARKER_SIZE, color='tab:cyan',
                   label='Cells')

ax_filter.set_ylabel('Remaining (%)', fontsize=LABEL_FONT_SIZE)
ax_filter.set_xlabel(common_name, fontsize=LABEL_FONT_SIZE)
ax_filter.set_ylim(0, 105)
ax_filter.set_yticks([0, 25, 50, 75, 100])
ax_filter.tick_params(axis='both', which='major', length=MAJOR_TICK_LENGTH,
                      width=MAJOR_TICK_WIDTH, labelsize=TICK_FONT_SIZE)
ax_filter.legend(fontsize=LEGEND_FONT_SIZE, frameon=False,
                 loc='lower left', bbox_to_anchor=(0.1, 1), ncol=2)
ax_filter.text(0, 1.2, 'C', fontsize=LABEL_FONT_SIZE, fontweight='bold',
               transform=ax_filter.transAxes)

# Plot 3: ARI with SNP shuffling
ax_shuffle = fig.add_subplot(gs[1:3, 1])
for method in sweep_dirs_shuffle:
    y = np.array(ari_means_shuffle[method])
    yerr = np.array(ari_stds_shuffle[method])
    x = np.array(variable_values_shuffle[method])
    ax_shuffle.errorbar(x, y, yerr=yerr, fmt='o-', markersize=MARKER_SIZE,
                        linewidth=LINE_WIDTH, capsize=CAP_SIZE, label=method)

ax_shuffle.set_title('D', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax_shuffle.set_ylabel('ARI', fontsize=LABEL_FONT_SIZE)
ax_shuffle.set_xlabel('SNP shuffle fraction', fontsize=LABEL_FONT_SIZE)
ax_shuffle.set_xlim(0, 1)
ax_shuffle.set_ylim(0, 1)
ax_shuffle.tick_params(axis='both', which='major', length=MAJOR_TICK_LENGTH,
                       width=MAJOR_TICK_WIDTH, labelsize=TICK_FONT_SIZE)
ax_shuffle.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH,
                       width=MINOR_TICK_WIDTH)

# Plot 4: Genotype errors for demuxHMM
ax_genotype = fig.add_subplot(gs[3:, 1])

# Plot HMM genotyping error
y = np.array(means_genotype)
yerr = np.array(stds_genotype)
x = np.array(vars_genotype)
ax_genotype.errorbar(x, y, yerr=yerr, fmt='o-', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, capsize=CAP_SIZE, label='demuxHMM')

# Plot Vireo genotyping error
y = np.array(means_genotype_vireo)
yerr = np.array(stds_genotype_vireo)
x = np.array(vars_genotype_vireo)
ax_genotype.errorbar(x, y, yerr=yerr, fmt='o-', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, capsize=CAP_SIZE, label='Vireo')

ax_genotype.set_title('E', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
ax_genotype.set_ylabel('Proportion SNPs Miscalled', fontsize=LABEL_FONT_SIZE)
ax_genotype.set_xlabel('Number of Individuals', fontsize=LABEL_FONT_SIZE)
ax_genotype.tick_params(axis='both', which='major', length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, labelsize=TICK_FONT_SIZE)
#ax_genotype.set_ylim(0, 0.08)

fig.tight_layout()

plt.savefig('figures/methods_comparison_20k_vertical.pdf')
plt.savefig('journal_figures/Fig5.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()