import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, hstack

CHROMOSOMES = [f"chr{i}" for i in range(1, 20)] + ["chrX"]

def load_chromosome(chrom_dir):
    """
    Load a single chromosome's cellSNP-lite output.

    Returns
    -------
    AD : scipy.sparse.csr_matrix
        ALT count matrix (cells × variants)
    DP : scipy.sparse.csr_matrix
        Total depth matrix (cells × variants)
    variants : pandas.DataFrame
        Variant metadata with columns: chrom, pos, ref, alt
    cell_names : numpy.ndarray
        Cell barcodes.
    """
    chrom_dir = Path(chrom_dir)
    vcf = chrom_dir / "cellSNP.base.vcf.gz"
    ad_file = chrom_dir / "cellSNP.tag.AD.mtx"
    dp_file = chrom_dir / "cellSNP.tag.DP.mtx"
    samples = chrom_dir / "cellSNP.samples.tsv"

    print(f"Loading {chrom_dir.name}")

    # cellSNP matrices are variants × cells; transpose to cells × variants
    AD = mmread(ad_file, spmatrix=True).tocsr().T
    DP = mmread(dp_file, spmatrix=True).tocsr().T

    cell_names = pd.read_csv(samples, header=None)[0].astype(str).to_numpy()

    variants = pd.read_csv(vcf, comment="#", sep="\t", header=None,
                            usecols=[0, 1, 3, 4], names=["chrom", "pos", "ref", "alt"])

    return AD, DP, variants, cell_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="var_calls",
                        help="Directory containing chr*/ outputs from cellsnp")

    parser.add_argument("--output", default="../data_dir/DO_mouse_cellsnp_001.h5ad",
                        help="Directory to store the processed and merged cellsnp output as a anndata h5ad.")

    parser.add_argument("--min-depth", type=int, default=20,
                        help="Minimum total DP across all cells")

    parser.add_argument("--min-cells", type=int, default=20,
                    help="The minimum number of cells that see the variant for it to be considered useful")

    parser.add_argument("--min-umis", type=int, default=100,
                    help="Minimum total UMIs per cell")

    parser.add_argument("--top-fraction", type=float, default=0.01,
        help="Fraction of variants to retain on each chromosome after filtering.")

    args = parser.parse_args()

    all_AD = []
    all_DP = []
    all_vars = []

    cell_names = None
    chrom_boundaries = {}
    current = 0

    # Load each chromosome
    for chrom in CHROMOSOMES:
        chrom_dir = Path(args.input) / chrom

        if not chrom_dir.exists():
            raise RuntimeError(f"Chromosome {chrom} directory not found")

        AD, DP, variants, cells = load_chromosome(chrom_dir)

        # Sanity check that we always order cells the same way
        if cell_names is None:
            cell_names = cells
        else:
            if not np.array_equal(cell_names, cells):
                raise RuntimeError(f"Cell ordering differs in {chrom}")

        # Keep track of the indices defining each chromosome
        start = current
        end = current + AD.shape[1]
        chrom_boundaries[chrom] = (start, end)
        current = end

        all_AD.append(AD)
        all_DP.append(DP)
        all_vars.append(variants)

    AD = hstack(all_AD).tocsr()
    DP = hstack(all_DP).tocsr()

    var = pd.concat(all_vars, ignore_index=True)

    # Filter to best variants (minimum depth and minimum cell coverage)
    print("Filtering variants")

    total_depth = np.asarray(DP.sum(axis=0)).ravel()
    covered_cells = np.asarray((DP > 0).sum(axis=0)).ravel()
    keep_var = (total_depth >= args.min_depth) & (covered_cells >= args.min_cells)

    AD = AD[:, keep_var]
    DP = DP[:, keep_var]
    var = var.iloc[keep_var].reset_index(drop=True)

    if args.top_fraction < 1.0:
        print(f"Selecting top {args.top_fraction*100} percent of SNPs per chromosome")

        keep_var = np.zeros(AD.shape[1], dtype=bool)

        for chrom in CHROMOSOMES:
            idx = np.where(var.chrom.values == chrom)[0]

            if len(idx) == 0:
                continue

            AD_chr = AD[:, idx]
            DP_chr = DP[:, idx]

            alt = np.asarray(AD_chr.sum(axis=0)).ravel().astype(np.float64)
            depth = np.asarray(DP_chr.sum(axis=0)).ravel().astype(np.float64)

            # Estimated allele frequency
            p = alt / np.maximum(depth, 1.0)
            p = np.clip(p, 1e-6, 1 - 1e-6)

            # Fisher information score
            info = depth / (p * (1.0 - p))

            n_keep = max(1, int(np.ceil(args.top_fraction * len(idx))))
            best = np.argpartition(info, -n_keep)[-n_keep:]
            keep_var[idx[best]] = True

        AD = AD[:, keep_var]
        DP = DP[:, keep_var]
        var = var.iloc[keep_var].reset_index(drop=True)

    # Compute chromosome boundary indices
    chrom_boundaries = {}
    current = 0

    for chrom in CHROMOSOMES:
        mask = var.chrom == chrom
        n = mask.sum()
        chrom_boundaries[chrom] = (current, current + n)
        current += n


    # Filter cells both by maximum depth and by making sure each cell has an alt read on each chromosome
    print("Filtering cells")
    total_alt = np.asarray(DP.sum(axis=1)).ravel()
    keep_cells = total_alt >= args.min_umis

    for chrom in CHROMOSOMES:
        # Get the chromosome indices
        start, end = chrom_boundaries[chrom]
        if end == start:
            continue

        alt = np.asarray(AD[:, start:end].sum(axis=1)).ravel()
        keep_cells &= alt > 0

    AD = AD[keep_cells]
    DP = DP[keep_cells]

    # Create the cells df
    kept_cells = cell_names[keep_cells]
    cell_pools = [cell_name.split('_')[0] for cell_name in kept_cells]
    obs = pd.DataFrame(index=kept_cells, data={'pool': cell_pools})

    # Build Anndata and save
    adata = ad.AnnData(X=csr_matrix(AD), obs=obs, var=var)
    adata.layers["depth"] = csr_matrix(DP)

    print(adata)
    print(f"Writing {args.output}")
    adata.write_h5ad(args.output, compression="gzip")


if __name__ == "__main__":
    main()