import numpy as np
from sklearn import metrics
import anndata
import argparse
from demuxHMM import HMMModel
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser(description="Run HMM on mouse sci-RNA-seq data")

# Paths and constants
ADATA_PATH = 'data_dir/cellSNP_full_unfiltered.h5ad'
METADATA_PATH = 'data_dir/jy-c.seqaggr.eagle.10.sm.best'
ALLOWED_CHR = [f'{i}' for i in range(1, 23)] + ['X']
THREADS = 50
TOL = 100
NUM_REPEATS = 30

# Load data
adata = anndata.read_h5ad(ADATA_PATH)

# Read the metadata file
metadata = pd.read_csv(METADATA_PATH, sep="\t", dtype=str)
metadata.set_index("BARCODE", inplace=True)

# Split 'BEST' column into 'singlet_status' and 'id'
metadata[["singlet_status", "id"]] = metadata["BEST"].str.split("-", n=1, expand=True)

# Optionally convert 'id' to numeric if needed
metadata["id"] = pd.to_numeric(metadata["id"], errors="coerce")

# Add to .obs of the AnnData object
adata.obs = adata.obs.join(metadata[["singlet_status", "id"]])

print(f'After filtering adata is {adata.shape}')

# Sort variants by chromosome and position
adata.var['variant_idx'] = np.arange(adata.var.shape[0])
adata.var = adata.var.sort_values(by=['chrom', 'pos'])
adata = adata[:, adata.var['variant_idx'].values]

# Subset to only singlets
adata = adata[adata.obs['singlet_status'] == 'SNG', :].copy()

depth_cells = adata.layers['depth'].sum(axis=1)
depth_cells = np.array(depth_cells)
depth_cells = depth_cells.flatten()
adata.obs['depth'] = depth_cells

# Determine number of organisms from unique cell IDs
ground_truth = adata.obs['id'].astype(str).values
unique_ids = np.unique(ground_truth)
NUM_ORGANISMS = len(unique_ids)
NUM_CHR = len(ALLOWED_CHR)
T_init = np.array([[[1/3, 1/3, 1/3], [1/3, 1/3,1/3], [1/3, 1/3,1/3]] for chrom in ALLOWED_CHR])
theta_init = np.array([0.006, 0.469, 0.988])
pi_init = np.array([0.57711514, 0.17719314, 0.24569172])

print(f"Running HMM with {NUM_ORGANISMS} organisms and {NUM_CHR} chromosomes")

# ─── HMM Input Matrices ───────────────────────────────────────────────────────
A = [adata[:, adata.var['chrom'] == chrom].X.toarray() for chrom in ALLOWED_CHR]
D = [adata[:, adata.var['chrom'] == chrom].layers['depth'].toarray() for chrom in ALLOWED_CHR]

scores = []
model_probs = []
models = []

start_time = datetime.now()

for _ in range(NUM_REPEATS):
    # ─── Run HMM ──────────────────────────────────────────────────────────────────
    model = HMMModel(A, D, NUM_ORGANISMS, None, threads=THREADS,
                     T_init=T_init, theta_init=theta_init, pi_init=pi_init, do_pi_update=False)
    model.solve(TOL, num_inits=0)
    model_out = model.get_output()
    assignments_new = np.argmax(model_out['z_probs'], axis=1)

    # ─── Evaluate ─────────────────────────────────────────────────────────────────
    rand_score = metrics.adjusted_rand_score(ground_truth, assignments_new)
    scores.append(rand_score)
    model_probs.append(model_out['probability'])
    models.append(model_out)
    print(f'ARI: {rand_score:.4f} with model prob {model_out["probability"]:.4f}')

end_time = datetime.now()
print('Took', end_time - start_time)

# Save scores over all runs
scores = np.array(scores)
model_probs = np.array(model_probs)
print('Best model ARI was: ', scores[np.argmax(model_probs)])

np.savetxt('scores.txt', scores)
np.savetxt('model_probs.txt', model_probs)

