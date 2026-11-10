import pandas as pd
import matplotlib.pyplot as plt

METHODS_PARAMS = {
    "demuxHMM": {
        "assignment_file": "sweeps/DO_mouse_downsample/demuxHMM_doublets_True/ds_frac_1/run_0/assignments_run_3.csv",
        "cluster_joint_probs": "sweeps/DO_mouse_downsample/demuxHMM_doublets_True/ds_frac_1/run_0/cluster_joints_run_3.npy",
        "cluster_col": "assignments_hmm_dedub",
        "pool_col": "gt_hmm"
    },
    "vireo": {
        "assignment_file": "sweeps/DO_mouse_downsample/vireo_doublets_True/ds_frac_1/run_4/assignments_run_19.csv",
        "cluster_col": "assignments_vireo_dedub",
        "pool_col": "gt_vireo"
    },
    "souporcell3": {
        "assignment_file": "sweeps/DO_mouse_downsample/sorc3_doublets_True/ds_frac_1/run_1/assignments_run_1.csv",
        "cluster_col": "assignments_souporcell_dedub",
        "pool_col": "gt_souporcell"
    }
}

COMPARE_METHOD = 'souporcell3' # vireo or souporcell3
METHOD_FONT_SIZE = 12
CLUSTER_FONT_SIZE = 5
POOL_FONT_SIZE = 6

# Load data
dfs = {}
for method in ['demuxHMM', COMPARE_METHOD]:
    params = METHODS_PARAMS[method]
    dfs[method] = pd.read_csv(params["assignment_file"], index_col=0)

# Get all pools so colors are consistent across plots
all_pools = sorted(dfs["demuxHMM"]["gt_hmm"].unique())

# Consistent color mapping
cmap = plt.get_cmap("tab20")
colors = {pool: cmap(i % cmap.N) for i, pool in enumerate(all_pools)}

# Determine grid size
clusters = {}
for method in ['demuxHMM', COMPARE_METHOD]:
    params = METHODS_PARAMS[method]
    clusters[method] = sorted(dfs[method][params["cluster_col"]].unique())
    print(f'{method} has {clusters[method]} clusters')

max_clusters = 7
nrows = len(all_pools)
ncols = 2 * max_clusters

fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 8.75))
axes = axes.reshape(nrows, ncols)

# Adjust subplot spacing
fig.subplots_adjust(left=0.12, right=0.98, bottom=0, top=0.95,
                    wspace=0.02, hspace=0.08)

# Draw one pie chart per cluster, grouped by pool
for row, pool in enumerate(all_pools):
    for method_idx, method in enumerate(['demuxHMM', COMPARE_METHOD]):
        params = METHODS_PARAMS[method]

        df = dfs[method]
        df = df[df[params['cluster_col']].notna()].copy()
        cluster_col = params["cluster_col"]
        pool_col = params["pool_col"]

        # Total number of cells in this pool
        pool_total = (df[pool_col] == pool).sum()

        # Find clusters belonging to this pool and sort by majority-pool fraction
        pool_clusters = []

        for cluster in clusters[method]:
            cluster_df = df[df[cluster_col] == cluster]
            counts = cluster_df[pool_col].value_counts()

            if len(counts) > 0 and counts.idxmax() == pool:
                majority_fraction = counts.max() / counts.sum()
                pool_clusters.append((cluster, majority_fraction))

        pool_clusters.sort(key=lambda x: x[1], reverse=True)

        for i, (cluster, majority_fraction) in enumerate(pool_clusters):
            col = method_idx * max_clusters + i
            ax = axes[row, col]

            cluster_df = df[df[cluster_col] == cluster]

            # Only include non-zero count pools
            counts = cluster_df[pool_col].value_counts().reindex(all_pools, fill_value=0)
            nonzero = counts > 0

            ax.pie(counts[nonzero],
                colors=[colors[p] for p in counts.index[nonzero]],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(edgecolor="white", linewidth=0.5))

            # Number of cells from the majority pool in this cluster
            pool_cells_in_cluster = (cluster_df[pool_col] == pool).sum()

            # Fraction of all cells in this pool represented by this cluster
            pool_fraction = pool_cells_in_cluster / pool_total

            cluster_title = f"{len(cluster_df)} cells"
            ax.text(0.5, 1.08, cluster_title, transform=ax.transAxes,
                    ha="center", va="top", fontsize=CLUSTER_FONT_SIZE)
            ax.set_aspect("equal")

        # Hide unused axes
        for i in range(len(pool_clusters), max_clusters):
            axes[row, method_idx * max_clusters + i].axis("off")

# Method labels
demux_pos = (axes[0, 0].get_position().x0 + axes[0, max_clusters - 1].get_position().x1) / 2
compare_method_pos = (axes[0, max_clusters].get_position().x0 + axes[0, -1].get_position().x1) / 2

fig.text(demux_pos, 0.96, "DemuxHMM", ha="center", va="bottom", fontsize=METHOD_FONT_SIZE)
fig.text(compare_method_pos, 0.96, COMPARE_METHOD.capitalize(), ha="center", va="bottom", fontsize=METHOD_FONT_SIZE)

# Pool labels
for row, pool in enumerate(all_pools):
    pool_name = pool.replace('-', ',')
    bbox = axes[row, 0].get_position()

    fig.text(0.06, bbox.y0 + bbox.height / 2, f"Pool\n{pool_name}",
             ha="center", va="center", fontsize=POOL_FONT_SIZE)

# Horizontal separator below each pool row
for row in range(len(all_pools) - 1):
    bbox_left = axes[row, 0].get_position()
    bbox_right = axes[row, -1].get_position()
    y = bbox_left.y0

    fig.add_artist(plt.Line2D([bbox_left.x0, bbox_right.x1], [y, y],
            transform=fig.transFigure, color="black", linewidth=1, alpha=0.5))

plt.savefig(f'figures/pi_charts_demuxHMM_v_{COMPARE_METHOD}.pdf')

journal_save_name = ''

if COMPARE_METHOD == 'souporcell3':
    journal_save_name='journal_figures/Fig4.tif'
elif COMPARE_METHOD == 'vireo':
    journal_save_name='journal_figures/Fig11.tif'

# Save in journal format
plt.savefig(journal_save_name, dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
plt.show()