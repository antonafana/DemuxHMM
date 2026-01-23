import anndata
import pandas as pd
import re
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)

BASE_PATH = 'data_dir/'
ADATA_PATH_BODY = BASE_PATH + 'v2_fca_biohub_body_10x_raw.h5ad'
ADATA_PATH_HEAD = BASE_PATH + 'v2_fca_biohub_body_10x_raw.h5ad'

ADATA_SAVE_PATH = BASE_PATH + 'head_body_lifted_genes.h5ad'
WILD5B_LIFTED_GENES_FROM_DM6_PATH = BASE_PATH + 'liftoff_lifted.gtf'
VCF_PATH = BASE_PATH + 'KSA2_snps_on_ref_Wild5B_filtered.vcf'
SAVE_PATH = BASE_PATH + 'adata_genes_map.csv'

CUTOFF_UMI = 1000 # The minimum number of UMI we want to allow

# Load adata and merge
adata_body = anndata.read_h5ad(ADATA_PATH_BODY)
adata_head = anndata.read_h5ad(ADATA_PATH_HEAD)
adata = anndata.concat([adata_body, adata_head], join='outer')
adata.obs_names_make_unique()
del adata_body, adata_head

adata_genes = adata.var.index.unique()   # <---- gene names now in index

# Parse GFF
def parse_line(gff_line):
    if gff_line.startswith('#'):
        return None
    fields = gff_line.split('\t')
    chrom, _, feature, start, end, *rest = fields
    return {
        'chromosome': chrom,
        'line_type': feature,
        'start': int(start),
        'end': int(end),
        'extra': '\t'.join(rest)
    }

parsed_lines = {'chromosome': [], 'start': [], 'end': [], 'extra': []}
with open(WILD5B_LIFTED_GENES_FROM_DM6_PATH, 'r') as f:
    for line in f:
        parsed_line = parse_line(line)
        if parsed_line and parsed_line['line_type'] == 'transcript':
            parsed_lines['chromosome'].append(parsed_line['chromosome'])
            parsed_lines['start'].append(parsed_line['start'])
            parsed_lines['end'].append(parsed_line['end'])
            parsed_lines['extra'].append(parsed_line['extra'])

gene_df = pd.DataFrame(parsed_lines)

# Exclude mitochondrial chromosome if needed
gene_df = gene_df[gene_df['chromosome'] != 'CM010575.1'].copy()

# Extract gene names
gene_name_pattern = r'gene\s+"([^"]+)"'
gene_df['gene_name'] = gene_df['extra'].apply(
    lambda x: re.search(gene_name_pattern, x).group(1)
)

# Merge duplicates: one row per gene
merged = []
for gene in gene_df['gene_name'].unique():
    g = gene_df[gene_df['gene_name'] == gene]
    merged.append({
        'chromosome': g['chromosome'].iloc[0],
        'start': g['start'].min(),
        'end': g['end'].max(),
        'gene_name': gene
    })
gene_df = pd.DataFrame(merged)

# Match adata genes
adata_genes_found = [g for g in adata_genes if g in gene_df['gene_name'].values]
adata_genes_not_found = [g for g in adata_genes if g not in adata_genes_found]

print(f"Genes in adata: {len(adata_genes)}")
print(f"Matched to GFF: {len(adata_genes_found)}")
print(f"Unmatched: {len(adata_genes_not_found)}")

adata.var['genes'] = adata.var.index
adata_found = adata[:, adata.var['genes'].isin(adata_genes_found)].copy()
adata_found.var['chromosome'] = adata_found.var['genes'].apply(lambda x: gene_df[gene_df['gene_name'] == x]['chromosome'].values[0])

# Check how many counts out of the total we are representing with these genes
count_sums_found = adata_found.X.sum(axis=1)
count_sums = adata.X.sum(axis=1)
prop_umi_retained = count_sums_found / count_sums
adata_found.obs['prop_umi_retained'] = prop_umi_retained

# Remove some low quality cells and data we don't use
row_sums = adata_found.X.sum(axis=1)
adata_found = adata_found[row_sums >= CUTOFF_UMI, :].copy()
adata_found.obsm.clear()
adata_found.varm.clear()

# Save the filtered adata
adata_found.write_h5ad(ADATA_SAVE_PATH)

# plt.figure(figsize=(10, 10))
# plt.hist(count_sums_found/count_sums, bins=100)
# plt.show()

# Read VCF of SNPs
with open(VCF_PATH, 'r') as f:
    vcf_lines = f.read().splitlines()

# Parse VCF into a dataframe
vcf_dict = {'chromosome': [], 'loc': [], 'start_base': [], 'end_base': []}
for line in vcf_lines:
    if line[0] == '#':
        continue

    split_line = line.split('\t')
    vcf_dict['chromosome'].append(split_line[0])
    vcf_dict['loc'].append(split_line[1])
    vcf_dict['start_base'].append(split_line[3])
    vcf_dict['end_base'].append(split_line[4])

vcf_df = pd.DataFrame(vcf_dict)
vcf_df['loc'] = vcf_df['loc'].astype(int)

def map_to_gene(row):
    """
    Maps a variant to a gene.
    Args:
        row: A row from the vcf_df. Has properties chromosome, loc, start_base, end_base

    Returns: maps_to_gene (boolean), gene_name (string or None), last_150 (whether it maps to the first 150 bases)
    """
    # Restrict our gene_df to the chromosome of interest
    chromosome = row['chromosome']

    # Now search the genes and see if they contain our variant
    found = False
    gene_name = None
    last_150 = False
    variant_loc = row['loc']

    # Subset our gene_df to our variant location
    gene_df_chromosome = gene_df[(gene_df['chromosome'] == chromosome) & \
                                  (gene_df['start'] <= variant_loc) & (gene_df['end'] >= variant_loc)]

    # If we have a match
    if gene_df_chromosome.shape[0] == 1:
        row = gene_df_chromosome.iloc[0]
        found = True
        gene_name = row['gene_name']

        if variant_loc > row['end'] - 150:
            last_150 = True

    return [found, gene_name, last_150]

vcf_df[['found','gene_name','last_150']] = vcf_df.parallel_apply(
    map_to_gene, axis=1, result_type="expand"
)

# Print some summary stats about our variants
print(f'Total variants = {vcf_df.shape[0]}')
print(f'Found variants = {vcf_df['found'].sum()}')
print(f'Variants in last 150 = {vcf_df['last_150'].sum()}')

# We don't need to keep variants we haven't matched to a gene
vcf_df = vcf_df[vcf_df['found']].copy()

# Save the dataframe of variants
vcf_df.to_csv(SAVE_PATH, sep='\t', index=False)