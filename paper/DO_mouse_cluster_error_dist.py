import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import figures_utils as fig_utils
from matplotlib.ticker import MultipleLocator
import glob

# User parameters
MODEL_ENTRIES = {
    'demuxHMM': {
        'ds_base_dir': 'sweeps/DO_mouse_downsample/demuxHMM_doublets_True/',
        'homog_key': 'hmm_homog_doublets_removed',
        'doublet_key': 'hmm_num_doublets_removed',
        'time_key': 'hmm_time_doublets_removed',
        'cluster_col': 'assignments_hmm_dedub',
        'pool_col': 'gt_hmm'
    },
    'vireo': {
        'ds_base_dir': 'sweeps/DO_mouse_downsample/vireo_doublets_True/',
        'homog_key': 'vireo_homog_doublets_removed',
        'doublet_key': 'vireo_num_doublets_removed',
        'time_key': 'vireo_time_doublets_removed',
        'cluster_col': 'assignments_vireo_dedub',
        'pool_col': 'gt_vireo'
    },
  'souporcell3': {
        'ds_base_dir': 'sweeps/DO_mouse_downsample/sorc3_doublets_True/',
        'homog_key': 'souporcell_homog_doublets_removed',
        'doublet_key': 'souporcell_num_doublets_removed',
        'time_key': 'souporcell_time_doublets_removed',
        'cluster_col': 'assignments_souporcell_dedub',
        'pool_col': 'gt_souporcell'
    },
}
colormap = {'demuxHMM': 'tab:blue',
            'vireo': 'tab:orange',
            'souporcell3': 'tab:red'}
MAX_RUNS = 5
PANEL_B_DS_FRAC = 1

DS_FRACS = [0.1, 0.25, 0.5, 1]
DS_HOMOG_SCORES = {method: [] for method in MODEL_ENTRIES}

# Make two plots (first one shows clustering performance with downsampling, second shows cumulative error)
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
MARKER_SIZE = 4
LINE_WIDTH = 1.5
CAP_SIZE = 2.5
MAJOR_TICK_LENGTH = 5
MAJOR_TICK_WIDTH = 1
MINOR_TICK_LENGTH = MAJOR_TICK_LENGTH / 2
MINOR_TICK_WIDTH = MAJOR_TICK_WIDTH / 2

homog_means = {method: [] for method in MODEL_ENTRIES}
homog_stds = {method: [] for method in MODEL_ENTRIES}
doublet_means = {method: [] for method in MODEL_ENTRIES}
doublet_stds = {method: [] for method in MODEL_ENTRIES}
time_means = {method: [] for method in MODEL_ENTRIES}
time_stds = {method: [] for method in MODEL_ENTRIES}

# Read the results from each downsampling and each method
for ds_frac in DS_FRACS:
    for method in MODEL_ENTRIES:
        results_path = MODEL_ENTRIES[method]['ds_base_dir'] + f'ds_frac_{ds_frac}/'
        PLACEHOLDER_VARIABLE = 'num_clusters'

        # Read homogeneity scores
        vars_method, homogs_method_frac = fig_utils.read_sweep_data(results_path, MAX_RUNS,
                                                              PLACEHOLDER_VARIABLE, MODEL_ENTRIES[method]['homog_key'])
        method_frac_homog_means, method_frac_homog_stds = fig_utils.compute_means_stds(vars_method, homogs_method_frac)

        if len(vars_method) > 0:
            homog_means[method].append(method_frac_homog_means[0])
            homog_stds[method].append(method_frac_homog_stds[0])
        else:
            homog_means[method].append(np.nan)
            homog_stds[method].append(np.nan)

        # Read times
        _, times_method_frac = fig_utils.read_sweep_data(results_path, MAX_RUNS,
                                                          PLACEHOLDER_VARIABLE, MODEL_ENTRIES[method]['time_key'])

        # Convert seconds to hours
        SECONDS_IN_HOUR = 3600
        times_method_frac = {var: np.array(times_method_frac[var]) / SECONDS_IN_HOUR for var in times_method_frac}

        # Get means and standard deviations for time taken
        method_frac_time_means, method_frac_time_stds = fig_utils.compute_means_stds(vars_method, times_method_frac)

        if len(vars_method) > 0:
            time_means[method].append(method_frac_time_means[0])
            time_stds[method].append(method_frac_time_stds[0])
        else:
            time_means[method].append(np.nan)
            time_stds[method].append(np.nan)

        # Read doublet counts
        _, doublets_method_frac = fig_utils.read_sweep_data(results_path, MAX_RUNS,
                                                          PLACEHOLDER_VARIABLE, MODEL_ENTRIES[method]['doublet_key'])
        method_frac_doublet_means, method_frac_doublet_stds = fig_utils.compute_means_stds(vars_method, doublets_method_frac)

        if len(vars_method) > 0:
            doublet_means[method].append(method_frac_doublet_means[0])
            doublet_stds[method].append(method_frac_doublet_stds[0])
        else:
            doublet_means[method].append(np.nan)
            doublet_stds[method].append(np.nan)

plt.figure(figsize=(7.5, 3.75))

# Subplot 1: Reads downsampling
plt.subplot(121)
plt.title('A', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
plt.ylabel('Homogeneity', fontsize=LABEL_FONT_SIZE)
plt.xlabel('Fraction of Reads Used', fontsize=LABEL_FONT_SIZE)

# Plot each method
for method in MODEL_ENTRIES:
    y = np.array(homog_means[method])
    yerr = np.array(homog_stds[method])
    x = np.array(DS_FRACS)
    mask = ~np.isnan(y)

    print(f'{method} has homogs {y} with stdevs {yerr}')
    print(f'{method} detected {doublet_means[method]} with stdevs {doublet_stds[method]}')
    print(f'{method} took {time_means[method]} hours with stdevs {time_stds[method]}')
    plt.errorbar(x[mask], y[mask], yerr=yerr[mask], marker='o', linewidth=LINE_WIDTH,
                    linestyle='-', capsize=CAP_SIZE, markersize=MARKER_SIZE,
                    color=colormap[method], label=method)

plt.ylim(0.81, 1)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))

# Plot 2: Cumulative error function
plt.subplot(122)

def read_fits(model, ds_frac):
    '''
    Reads the clustering results for a given model and downsampling fraction.

    :param model: The model to read.
    :param ds_frac: The fraction of reads used.
    :return:
    '''
    results_path = MODEL_ENTRIES[model]['ds_base_dir'] + f'ds_frac_{ds_frac}/'

    assignment_files = []

    # Read the single assignment file from each data repeat
    for run in range(MAX_RUNS):
        run_path = results_path + f'run_{run}/'
        assignment_files.extend(glob.glob(run_path + 'assignments_run_*.csv'))

    return assignment_files

# Read the results for each model
model_results_dfs = []
model_clusters = []

# Read the results for each model
for model in MODEL_ENTRIES:
    assignment_files = read_fits(model, PANEL_B_DS_FRAC)

    errors = []

    if model == 'souporcell3':
        print(assignment_files)

    for assignment_file in assignment_files:
        # Load data
        df = pd.read_csv(assignment_file, index_col=0)
        clusters = sorted(df[MODEL_ENTRIES[model]['cluster_col']].unique())

        # Calculate the global pool entropy for the normalization
        # H(C) in the Rosenberg paper
        pool_counts = df[MODEL_ENTRIES[model]['pool_col']].value_counts()
        pool_probs = pool_counts / pool_counts.sum()
        pool_entropy = -(pool_probs * np.log(pool_probs)).sum()

        for cluster in clusters:
            # This computes H(C | K = k) rather than summing over all k terms in Rosenberg
            cluster_df = df[df[MODEL_ENTRIES[model]['cluster_col']] == cluster]
            counts = cluster_df[MODEL_ENTRIES[model]['pool_col']].value_counts()
            probs = counts / counts.sum()

            # Calculate per-cluster homogeneity from conditional entropy
            cluster_entropy = -(probs * np.log(probs)).sum()
            homogeneity = 1 - cluster_entropy / pool_entropy
            errors.append(1 - homogeneity)

    errors = np.sort(errors)
    ccdf = 1 - np.arange(1, len(errors) + 1) / len(errors)

    # Quantify what fraction of clusters have more than x% error
    frac_over = lambda y: (errors > y).sum() / len(errors)

    for x in [0.01, 0.2]:
        print(f'For {model}, {frac_over(x)} of clusters have over {x} error')

    plt.step(errors, ccdf, where="post", lw=2, color=colormap[model])

plt.xlabel("Cluster error", fontsize=LABEL_FONT_SIZE)
plt.ylabel("Fraction of clusters with error ≥ x", fontsize=LABEL_FONT_SIZE)
plt.title('B', loc='left', fontsize=LABEL_FONT_SIZE, fontweight='bold')
plt.legend(MODEL_ENTRIES.keys())
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('figures/DO_mouse_clustering_error.pdf')
plt.savefig('journal_figures/Fig3.tif', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()