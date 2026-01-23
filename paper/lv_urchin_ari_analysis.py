import wot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import anndata
from tqdm.notebook import tqdm
from scipy.spatial import distance_matrix
import ot
from sklearn.metrics import adjusted_rand_score
from matplotlib.lines import Line2D

DATA_DIR = "data_dir/urchin_data/"
DATA_RAW_PATH = DATA_DIR + "adata_raw_0925_loom.h5ad"
EXP_RATES_PATH = DATA_DIR + "cell_growth_hpf_expected.csv"
TMAP_SAVE_DIR = "urchin_tmaps/"
RATIOS = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T = 24 # The terminal time
NUM_REPEATS = 10
ERROR_TIME = 10

# Load anndata and growth rates
adata = anndata.read_h5ad(DATA_RAW_PATH)
exp_rates = pd.read_csv(EXP_RATES_PATH)
exp_rates.set_index('hr', inplace = True)
exp_rates = exp_rates.rename(columns={'other':'maternal'})

# Apply growth rates to anndata
adata.obs['exp_growth'] = adata.obs.apply(lambda cell: exp_rates.loc[cell.hpf, cell.type], axis=1)

# Remove unneeded data for memory savings
adata.obs = adata.obs[['type', 'hpf', 'exp_growth', 'subtype']]
adata.var = adata.var[[]]
adata.obsm.clear()
adata.varm.clear()

def shuffle_timepoints(adata, frac=0.1, seed=None):
    """
    Return a new AnnData object where a fraction of cells from each timepoint
    are reassigned to random *other* timepoints.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing obs['hpf'].
    frac : float
        Fraction of cells per timepoint to reassign.
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    adata_copy : AnnData
        A new AnnData object with modified obs['hpf'].
    """
    if seed is not None:
        np.random.seed(seed)

    # Make a full copy so original adata is untouched
    adata_copy = adata.copy()

    hpf = adata_copy.obs['hpf'].copy()
    timepoints = hpf.unique()

    to_modify = []

    # Choose cells to shuffle from each timepoint
    for tp in timepoints:
        tp_cells = hpf[hpf == tp].index
        n = int(len(tp_cells) * frac)
        if n > 0:
            chosen = np.random.choice(tp_cells, n, replace=False)
            to_modify.extend(chosen)

    # Reassign new timepoints (but not the same one)
    for idx in to_modify:
        current_tp = hpf.loc[idx]
        other_tps = [tp for tp in timepoints if tp != current_tp]
        new_tp = np.random.choice(other_tps)
        hpf.loc[idx] = new_tp

    # Write changes into the copy
    adata_copy.obs['hpf'] = hpf

    # Recompute growth rates
    adata_copy.obs['exp_growth'] = adata_copy.obs.apply(lambda cell: exp_rates.loc[cell.hpf, cell.type], axis=1)

    return adata_copy

# Code for making triangle plot
def make_triangle_plot(fate_ds, init_fates, name1, name2, day, subplot_idx, cols, ratio):
    plt.figure(figsize=(5,5))
    #plt.subplot(1, cols, subplot_idx)

    fate1 = fate_ds[:,name1][fate_ds.obs['day']==day].X.flatten()
    fate2 = fate_ds[:,name2][fate_ds.obs['day']==day].X.flatten()

    init_fate1 = init_fates[:,name1][init_fates.obs['day']==day].X.flatten()
    init_fate2 = init_fates[:,name2][init_fates.obs['day']==day].X.flatten()

    Nrows = len(fate1)
    x = np.zeros(Nrows)
    y = np.zeros(Nrows)
    P = np.array([[1,0],[np.cos(2*np.pi/3),np.sin(2*np.pi/3)],[np.cos(4*np.pi/3),np.sin(4*np.pi/3)]])

    for i in range(0,Nrows):
        ff = np.array([fate1[i],fate2[i],1-(fate1[i]+fate2[i])])
        x[i] = (ff @ P)[0]
        y[i] = (ff @ P)[1]

    init_Nrows = len(init_fate1)
    init_x = np.zeros(init_Nrows)
    init_y = np.zeros(init_Nrows)

    for i in range(0,init_Nrows):
        ff = np.array([init_fate1[i],init_fate2[i],1-(init_fate1[i]+init_fate2[i])])
        init_x[i] = (ff @ P)[0]
        init_y[i] = (ff @ P)[1]

    vx = P[:,0]
    vy = P[:,1]
    t1 = plt.Polygon(P, color=(0,0,0,0.1))
    plt.gca().add_patch(t1)

    plt.scatter(init_x, init_y, c='lightsalmon')
    plt.scatter(x,y)
    plt.scatter(vx,vy)
    # plt.text(P[0,0]+.1, P[0,1], name1)
    # plt.text(P[1,0]-.1, P[1,1]+.1, name2)
    # plt.text(P[2,0]-.1, P[2,1]-.2, 'Other')
    plt.axis('equal')
    plt.axis('off')

    #plt.title(f'Mixing percentage {ratio:.3f}')
    plt.savefig(f'figures/{ratio:.2f}.pdf', transparent=True)

all_aris = []
all_errors = []
base_fate = None

for run_num in range(NUM_REPEATS):
    # Create a set of shuffled adatas at the specified ratios
    adatas_shuffled = [shuffle_timepoints(adata, ratio) for ratio in RATIOS]

    # Specify ot models for each
    get_ot = lambda ad: wot.ot.OTModel(ad, day_field = 'hpf', growth_rate_field='exp_growth', epsilon = 0.05, lambda1= 1, lambda2= 50, growth_iters = 1)
    ot_models = [get_ot(adata_s) for adata_s in adatas_shuffled]
    tmap_save_dirs = [TMAP_SAVE_DIR + str(ratio) + "/" for ratio in RATIOS]

    # Compute tmaps
    for i, (model, save_dir) in tqdm(enumerate(zip(ot_models, tmap_save_dirs)), total=len(ot_models), desc=f'Run num {run_num}'):
        model.compute_all_transport_maps(tmap_out=save_dir)

    # Load tmap models
    tmap_models = [wot.tmap.TransportMapModel.from_directory(tmap_path) for tmap_path in tmap_save_dirs]

    # Quantify mixing ratio as ARI for the sake of interpretability to readers
    ari_scores = []

    for adata_s in adatas_shuffled:
        shuffled_hpf = adata_s.obs['hpf']
        true_hpf = adata.obs['hpf']
        ari = adjusted_rand_score(shuffled_hpf, true_hpf)
        ari_scores.append(ari)

    all_aris.append(ari_scores)

    # Create cell sets and append to tmaps
    # Compute cell fates relative to the final timepoint
    fates = []

    for tmap_model, adata_s in zip(tmap_models, adatas_shuffled):
        types = list(pd.unique(adata_s.obs.type))
        types.sort()
        cell_sets = {}
        for t in types:
            cell_sets[t] = list(adata_s.obs.index[adata_s.obs.type == t])

        target_destinations = tmap_model.population_from_cell_sets(cell_sets, at_time=T)
        fate_ds = tmap_model.fates(target_destinations)
        fates.append(fate_ds)

    if run_num == 0:
        # Print triangle plots on the first repeat. This should convey the point for visualization.
        for i, fate_ds in enumerate(fates):
            make_triangle_plot(fate_ds, fates[0], 'endoderm', 'SMC', 10, i + 1, cols=len(RATIOS), ratio=ari_scores[i])

    """
    Compute fate errors using optimal transport. We will use EMD1 on the fate matrices. Each cell at a given time will have a uniform weight. The cost matrix between cells will be the L1 distance between their fate vectors.
    """
    # Make distance matrices for each mixing percentage and each time
    times = adata.obs['hpf'].unique()
    times = np.sort(times)
    distances = {percentage: {time: [] for time in times} for percentage in RATIOS}

    # Because shuffled times can have maternal cells in the final time, we need to account for those fates
    # The ground truth should just be zero
    if run_num == 0:
        zero_col = np.zeros((fates[0].n_obs, 1))
        X_new = np.hstack([fates[0].X, zero_col])
        new_var = pd.DataFrame(index=(list(fates[0].var.index) + ["maternal"]))
        base_fate = anndata.AnnData(X=X_new, obs=fates[0].obs.copy(), var=new_var)

    for i, percentage in enumerate(RATIOS):
        for time in times:
            base_fates = base_fate[base_fate.obs['day']==time, :].X

            # We had to do manipulations to the structure of the first timepoint so there is a dimensionality difference here
            if i == 0:
                comparison_fates = base_fates
            else:
                comparison_fates = fates[i][fates[i].obs['day']==time, :].X

            distances[percentage][time] = distance_matrix(comparison_fates, base_fates, 1)

    # Compute the EMD1 distance at the given time for each mixing percentage
    errors = []

    for i, percentage in enumerate(RATIOS):
        cost_mat = distances[percentage][ERROR_TIME]

        # Make uniform distributions for the pairing
        num_cells_shuffled_data = fates[i][fates[i].obs['day']==ERROR_TIME].shape[0]
        num_cells_base = fates[0][fates[0].obs['day']==ERROR_TIME].shape[0]
        uniform_dist_shuffled = np.ones(num_cells_shuffled_data) / num_cells_shuffled_data
        uniform_dist_base = np.ones(num_cells_base) / num_cells_base

        # Get EMD distance
        emd_dist = ot.emd2(uniform_dist_shuffled, uniform_dist_base, cost_mat, numItermax=500000)
        errors.append(emd_dist)

    all_errors.append(errors)

# Aggregate all our runs
all_aris_arr = np.array(all_aris)
all_errors_err = np.array(all_errors)

aris_mean = np.mean(all_aris_arr, axis=0)
errors_mean = np.mean(all_errors_err, axis=0)
aris_sd = np.std(all_aris_arr, axis=0)
errors_sd = np.std(all_errors_err, axis=0)

plt.figure(figsize=(7,7))
plt.xlabel('Adjusted Rand Index', fontsize=20)
plt.ylabel('Earth Mover\'s Distance (L1 Norm)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.errorbar(aris_mean, errors_mean, xerr=aris_sd, yerr=errors_sd, fmt='o', capsize=4, color='springgreen')
plt.gca().invert_xaxis()

# Make a figure legend compatible with
custom_legend = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='springgreen', markersize=8,
           label='EMD error'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='lightsalmon', markersize=6,
           label='Baseline fates'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='tab:blue', markersize=6,
           label='Perturbed fates'),
]

plt.legend(handles=custom_legend, loc='upper left', fontsize=14)

# Save figure
plt.savefig('figures/urchin_emd_errors.pdf', transparent=True)
plt.show()