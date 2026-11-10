import numpy as np
import anndata
import argparse
import os

parser = argparse.ArgumentParser(description="Downsample DO Mouse reads")
parser.add_argument('--base_path', type=str, default='data_dir/DO_mouse_cellsnp_001.h5ad',
                    help='Path to the base dataset')
parser.add_argument('--save_path', type=str, default='data_dir/downsampled_mouse/downsampled_do_mouse.h5ad',
                    help='Where to save downsampled data.')
parser.add_argument("--downsample_fraction", type=float, default=1, help="Fraction of reads to keep")
parser.add_argument("--snp_fraction", type=float, default=1, help="Fraction of SNPs to retain on each chromosome")
parser.add_argument("--num_chromosomes", type=int, default=20, help="Number of chromosomes to retain")
parser.add_argument("--num_cells", type=int, default=-1, help="Fraction of reads to keep")
params = vars(parser.parse_args())
print('Running with params: ', params)

def downsample_reads(adata, fraction, random_state=None):
    """
    Downsample sequencing reads in place by independently sampling ALT and REF reads.

    :param adata: AnnData object containing ALT reads in X and total depth in layers["depth"].
    :param fraction: Fraction of reads to retain, between 0 and 1.
    :param random_state: Random seed or NumPy random generator.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    X = adata.X.tocsr()
    depth = adata.layers["depth"].tocsr()

    ref = depth - X

    X.data = rng.binomial(X.data.astype(np.int64), fraction)
    X.eliminate_zeros()

    ref.data = rng.binomial(ref.data.astype(np.int64), fraction)
    ref.eliminate_zeros()

    depth = X + ref

    adata.X = X
    adata.layers["depth"] = depth

def downsample_snps(adata, fraction, random_state=None):
    """
    Downsample SNPs independently within each chromosome.

    :param adata: AnnData object containing chromosome labels in var["chrom"].
    :param fraction: Fraction of SNPs to retain from each chromosome, between 0 and 1.
    :param random_state: Random seed or NumPy random generator.
    :return: AnnData object with downsampled SNPs.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    keep_indices = []

    for chrom in adata.var["chrom"].unique():
        chrom_indices = np.flatnonzero(adata.var["chrom"].to_numpy() == chrom)
        num_keep = max(1, int(np.floor(len(chrom_indices) * fraction)))
        keep_indices.extend(rng.choice(chrom_indices, size=num_keep, replace=False))

    keep_indices.sort()

    return adata[:, keep_indices].copy()

def downsample_chromosomes(adata, num_chromosomes, random_state=None):
    """
    Randomly retain a specified number of whole chromosomes.

    :param adata: AnnData object containing chromosome labels in var["chrom"].
    :param num_chromosomes: Number of chromosomes to retain, between 1 and 20.
    :param random_state: Random seed or NumPy random generator.
    :return: AnnData object containing only the selected chromosomes.
    """
    chromosomes = adata.var["chrom"].unique()

    if not 1 <= num_chromosomes <= min(20, len(chromosomes)):
        raise ValueError(f"num_chromosomes must be between 1 and {min(20, len(chromosomes))}")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    selected_chromosomes = rng.choice(chromosomes, size=num_chromosomes, replace=False)
    keep_mask = adata.var["chrom"].isin(selected_chromosomes)

    return adata[:, keep_mask].copy()

def downsample_cells(adata, num_cells, random_state=None):
    """
    Randomly downsample cells to exactly num_cells.

    :param adata: AnnData object containing cells as observations.
    :param num_cells: Number of cells to retain.
    :param random_state: Random seed or NumPy random generator.
    """
    if not 0 < num_cells <= adata.n_obs:
        raise ValueError(f"num_cells must be between 1 and {adata.n_obs}")

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    cell_indices = rng.choice(adata.n_obs, size=num_cells, replace=False)
    cell_indices.sort()

    adata._inplace_subset_obs(cell_indices)

# Read base adata
adata = anndata.read_h5ad(params['base_path'])

# Downsample chromosomes, SNPs, reads, and then cells
if params['num_chromosomes'] < 20:
    adata = downsample_chromosomes(adata, params['num_chromosomes'])

if params['snp_fraction'] < 1:
    adata = downsample_snps(adata, params['snp_fraction'])

if params['downsample_fraction'] < 1:
    downsample_reads(adata, params['downsample_fraction'])

if params['num_cells'] > 0:
    downsample_cells(adata, params['num_cells'])

# Make directories if needed
dir_name = os.path.dirname(params['save_path'])
os.makedirs(dir_name, exist_ok=True)

# Save result
adata.write_h5ad(params['save_path'], compression='gzip')