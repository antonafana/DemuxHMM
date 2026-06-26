import os
import gzip
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack
import anndata as ad
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='The directory containing the vcfs')
args = vars(parser.parse_args())

# Paths
data_dir = args['data_dir']
output_file = os.path.join(data_dir, "cellSNP_full_unfiltered.h5ad")

# Chromosomes to include
chromosomes = [f"{i}" for i in range(1, 23)] + ["X"]

# Read barcodes
barcode_file = os.path.join(data_dir, chromosomes[0], "cellSNP.samples.tsv")
barcodes = pd.read_csv(barcode_file, header=None)[0].tolist()

# Collect matrices and variant data
X_list, DP_list, var_dfs = [], [], []

for chrom in chromosomes:
    chr_path = os.path.join(data_dir, chrom)
    ad_path = os.path.join(chr_path, "cellSNP.tag.AD.mtx")
    dp_path = os.path.join(chr_path, "cellSNP.tag.DP.mtx")
    vcf_path = os.path.join(chr_path, "cellSNP.base.vcf.gz")

    # Read and transpose matrices
    X = mmread(ad_path).tocsr().transpose()
    DP = mmread(dp_path).tocsr().transpose()
    X_list.append(X)
    DP_list.append(DP)

    # Parse VCF
    records = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            records.append({
                "chrom": parts[0],
                "pos": int(parts[1]),
                "ref": parts[3],
                "alt": parts[4],
                "variant": f"{parts[3]}>{parts[4]}"            })

    var_df = pd.DataFrame(records)
    var_df["total_depth"] = np.array(DP.sum(axis=0)).flatten()
    var_dfs.append(var_df)

# Combine matrices and annotations
X_all = hstack(X_list)
DP_all = hstack(DP_list)
var_all = pd.concat(var_dfs, ignore_index=True)

# Create AnnData
adata = ad.AnnData(X=X_all)
adata.layers["depth"] = DP_all
adata.var = var_all
adata.obs_names = barcodes

# Save
adata.write_h5ad(output_file)
print(f"Saved AnnData to: {output_file}")

                                                     
