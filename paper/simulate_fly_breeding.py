import numpy as np
import argparse
import json
import pandas as pd
import os
import simulation_utils as utils
import pickle
import anndata

parser = argparse.ArgumentParser(description="Script Parameters")

# Add arguments with default values specified directly
parser.add_argument('--VCF_df_path', type=str,
                    default='data_dir/adata_genes_map.csv',
                    help='Path to the VCF dataframe.')
parser.add_argument('--VCF_FILENAME', type=str,
                    default='data_dir/KSA2_snps_on_ref_Wild5B_filtered.vcf',
                    help='Path to the VCF file.')
parser.add_argument('--scrnaseq_adata', type=str,
                     default='data_dir/head_body_lifted_genes.h5ad',
                     help='Path to the adata file containing scRNA-seq reads.')
parser.add_argument('--data_path', type=str,
                     default=None,
                     help='Path to a previously generated datasets.')
parser.add_argument('--transition_save_path', type=str,
                     default=None,
                     help='Path to where to save a generated dataset.')
parser.add_argument('--cutoff_UMI', type=int, default=1500, help='Minimum UMI for cells in our sc dataset.')
parser.add_argument('--valid_contigs', type=str, nargs='+',
                    default=['CM010569.1', 'CM010570.1', 'CM010571.1', 'CM010572.1', 'CM010573.1', 'CM010574.1'],
                    help='List of valid contigs.')
                    # Chromosome 4 is mostly conserved and does more harm than good
parser.add_argument('--num_generations', type=int, default=2, help='Number of generations in the breeding experiment.')
parser.add_argument('--offspring_per_generation', type=int, default=100, help='Number of offspring per generation.')
parser.add_argument('--num_cells_per_org', type=int, default=10, help='Number of cells per organism.')
parser.add_argument('--avg_UMI', type=int, default=1500, help='Average UMI.')
parser.add_argument('--variance_UMI', type=int, default=0, help='Variance in UMI.')
parser.add_argument('--num_threads', type=int, default=80, help='Number of threads for the HMM model.')
parser.add_argument('--TOL', type=int, default=10, help='Tolerance value for the HMM model.')
parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats for the HMM model.')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save results.')
parser.add_argument('--run_num', type=int, default=0, help='Run number.')
parser.add_argument('--males_cross_over', type=bool, default=False, help='Whether or not to use Males cross over.')
parser.add_argument('--gender_filter', type=str, default='mix', choices={'mix', 'male', 'female'},
                    help='Determines which genders are used in the final demultiplexing. Options are mix, male, or female.')
parser.add_argument('--no_vireo',action="store_true", default=False, help='Whether or not to use vireoSNP.')
parser.add_argument('--no_scsplit',action="store_true", default=False, help='Whether or not to use scSplit.')
parser.add_argument('--no_souporcell',action="store_true", default=False, help='Whether or not to use souporcell.')
parser.add_argument('--no_demuxHMM',action="store_true", default=False, help='Whether or not to use demuxHMM.')
parser.add_argument('--backcross',action="store_true", default=False, help='Whether or not to backcross.')
parser.add_argument("--snp_usage_percent", type=float, default=1, help="Percentage of SNPs to use.")

params = vars(parser.parse_args())
print('Running with params: ', params)

contig_chrom_map = {'CM010569.1':'X', 'CM010570.1':'2', 'CM010571.1':'2',
                             'CM010572.1':'3', 'CM010573.1':'3', 'CM010574.1':'4'}
valid_chromosomes = ['X', '2', '3'] # We filter out 4 because it is always conserved and adds no info
num_chr = len(valid_chromosomes)

# Read pre-processed df of SNPs that can be captured with scRNA-seq (last 150 bp)
snp_df = pd.read_csv(params['VCF_df_path'], sep='\t')

# Filter to SNPs on genes that contain multiple
prop_to_keep = 0.15
num_snps_by_gene = snp_df['gene_name'].value_counts()
top_n = int(prop_to_keep*len(num_snps_by_gene))
top_genes = num_snps_by_gene.head(top_n)
snp_df = snp_df[snp_df['gene_name'].isin(set(top_genes.index))].copy()

# Now further keep only the ones scRNA-seq can see
snp_df = snp_df[snp_df['last_150']].copy()

# Rename contigs to chromosomes
snp_df = snp_df[snp_df['chromosome'].isin(params['valid_contigs'])].copy()
snp_df['chromosome'] = snp_df['chromosome'].apply(lambda x: contig_chrom_map[x])

# Subset snps by chromosome and re-index
snp_dfs_chrom = {}
num_snps_chrom = {}

for chrom in valid_chromosomes:
    snps_sub = snp_df[snp_df['chromosome'] == chrom].copy()
    snps_sub = snps_sub.sort_values(by='loc')
    snps_sub = snps_sub.reset_index(drop=True)
    snp_dfs_chrom[chrom] = snps_sub
    num_snps_chrom[chrom] = len(snps_sub)

# Get a starting scRNA-seq dataset to base our organisms from
adata = anndata.read_h5ad(params['scrnaseq_adata'])

# Filter our genes from our mapping that are never actually expressed
expressed_genes = set(adata.var[adata.X.sum(axis=0) > 0]['genes'])
for chrom in valid_chromosomes:
    chrom_snp_df = snp_dfs_chrom[chrom]
    snp_dfs_chrom[chrom] = chrom_snp_df[chrom_snp_df['gene_name'].isin(expressed_genes)].copy()
    snp_dfs_chrom[chrom].reset_index(drop=True, inplace=True)  # reset indices to 0..n-1

# Either load or generate the desired dataset
if params['data_path'] is not None:
    print('Loading data...')
    try:
        with open(params['data_path'], 'rb') as f:
            data = pickle.load(f)
        A, D, organisms = data['A'], data['D'], data['organisms']
        print('Loaded data')
    except (FileNotFoundError, pickle.UnpicklingError, KeyError):
        # The data doesn't exist yet. Generate and save.
        A, D, organisms = utils.simulate_dataset(params, snp_dfs_chrom, adata)

        directory = os.path.dirname(params['data_path'])
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        pickle.dump({'A': A, 'D': D, 'organisms': organisms}, open(params['data_path'], 'wb'))
else:
    A, D, organisms = utils.simulate_dataset(params, snp_dfs_chrom, adata)

# Save some memory
del adata

# Optionally downsample SNPs
def downsample_snps_even(A, D, params, rng=None):
    """
    Downsample SNPs evenly across chromosomes.

    A, D : lists of arrays, shape (num_cells, num_snps_chr_i)
    params["snp_usage_percent"] : float in (0, 1]
    rng : np.random.Generator (optional, for reproducibility)
    """

    p = params.get("snp_usage_percent", 1.0)
    if p >= 1.0:
        return A, D

    if rng is None:
        rng = np.random.default_rng()

    A_ds = []
    D_ds = []

    for Ai, Di in zip(A, D):
        n = Ai.shape[1]
        k = max(1, int(np.floor(p * n)))   # keep at least 1 SNP per chromosome

        idx = rng.choice(n, size=k, replace=False)
        idx.sort()  # keep genomic order

        A_ds.append(Ai[:, idx])
        D_ds.append(Di[:, idx])

    return A_ds, D_ds

A, D = downsample_snps_even(A, D, params)

for i in range(len(A)):
    print('A shape: ', A[i].shape)

# A and D are lists per chromosome, with A[i] shape (n_cells, n_snps_chr)
# valid_chromosomes order is the same as A/D
chr_index = valid_chromosomes.index('X')
A_X = A[chr_index]   # shape (n_cells, n_snps)
D_X = D[chr_index]

# fraction of cells with zero alt reads on X
zero_alt_per_cell = (A_X.sum(axis=1) == 0)
print("Cells with zero alt on X:", zero_alt_per_cell.sum(), "/", A_X.shape[0],
      f"({100*zero_alt_per_cell.mean():.2f}%)")

# distribution of total coverage on X per cell
total_cov_per_cell = D_X.sum(axis=1)
import numpy as np
print("X coverage per cell: mean, median, 10th, 90th:",
      np.mean(total_cov_per_cell), np.median(total_cov_per_cell),
      np.percentile(total_cov_per_cell, 10), np.percentile(total_cov_per_cell, 90))

alt_per_site = A_X.sum(axis=0)    # total alt reads observed at each SNP across all cells
print("Sites with zero alt reads (never observed):", (alt_per_site == 0).sum(), "/", len(alt_per_site))
print("Top 10 alt counts per site:", np.sort(alt_per_site)[-10:])

# final_gen organism_list contains organisms; each org.chromosomes['X'].states is 1xN or 2xN
for i, org in enumerate(organisms[:20]):
    states = org.chromosomes['X'].states
    print("org", i, "gender", org.gender, "states shape", states.shape, "mean alt per chromatid:",
          [s.mean() for s in states])

genders = [org.gender for org in organisms]
from collections import Counter
print("Gender counts:", Counter(genders))


num_organisms = len(organisms)

if params['transition_save_path'] is not None:
    mean_transitions = np.zeros(num_chr)

    for i, chr_name in enumerate(valid_chromosomes):
        for organism in organisms:
                chromosome = organism.chromosomes[chr_name]
                num_transitions1, num_transitions2 = utils.count_transitions(chromosome)

                # Adjust for x-inactivation
                if organism.gender == 'F' and chr_name == 'X':
                    mean_transitions[i] += (num_transitions1 + num_transitions2) / 2
                else:
                    mean_transitions[i] += num_transitions1 + num_transitions2

    mean_transitions = mean_transitions / num_organisms
    pickle.dump(mean_transitions, open(params['transition_save_path'], 'wb'))
else:
    mean_transitions = utils.get_transitions(params['num_generations'])
    print(f'Saw {mean_transitions} transitions on average')

# Make a ground truth
ground_truth = np.zeros(params['num_cells_per_org']*num_organisms)

for i in range(num_organisms):
    ground_truth[params['num_cells_per_org']*i:params['num_cells_per_org']*i+params['num_cells_per_org']] = i

# Filter out cells with no reads in one of the chromosomes
# For the HMM model to run you technically only have to filter out the ones with D reads = 0, but performance
# is better this way for all models, so this represents a reasonable filtering step a group could take
total_cells = A[0].shape[0]
mask = np.full(total_cells, True)
 
for i in range(len(D)):
    count_sums_D = D[i].sum(axis=1)
    count_sums_A = A[i].sum(axis=1)
    mask = np.bitwise_and(mask, count_sums_D != 0)
    mask = np.bitwise_and(mask, count_sums_A != 0)

keep_inds = np.arange(total_cells, dtype=int)[mask]
A_filtered = [A[j][keep_inds, :] for j in range(num_chr)]
D_filtered = [D[j][keep_inds, :] for j in range(num_chr)]
ground_truth_filtered = ground_truth[mask]

emb_remaining = len(np.unique(ground_truth_filtered))
cells_remaining = len(keep_inds)
print('Unique emb remaining', emb_remaining, 'Original', num_organisms)
print(f'The number of total cells is {total_cells}, the number of filtered cells is {cells_remaining}')

# Create an initial T that differentiates between X and other chromosomes
num_vars = [mat.shape[1] for mat in A_filtered]
num_states = 3
T = np.zeros(shape=(num_chr, num_states, num_states))

for c, name in enumerate(valid_chromosomes):
    eps = 10e-10
    p = mean_transitions[c] / num_vars[c]

    T[c, :, :] = np.array([[1 - p, p, 0],
                               [p / 2, 1 - p, p / 2],
                               [0, p, 1 - p]])

    T[c, :, :] = (T[c, :, :] + eps) / (1 + 3*eps)

# Run the HMM model
theta_init = np.array([0.006, 0.469, 0.988])
pi_init = None
if not params['no_demuxHMM']:
    assignments_hmm, ari_score_hmm, time_hmm, model = utils.run_demuxHMM(A_filtered, D_filtered, num_organisms,
                                                             params['TOL'], ground_truth_filtered,
                                                             mean_transitions=mean_transitions,
                                                             num_threads=params['num_threads'],
                                                             num_repeats=params['num_repeats'],
                                                             theta_init=theta_init,
                                                             pi_init=pi_init,
                                                             T_init=T)
    print(f' HMM score is {ari_score_hmm} and took {time_hmm} seconds to run')

if not params['no_scsplit']:
    assignments_scsplit, gt_filt_scsplit, ari_score_scsplit, time_scsplit = utils.run_scsplit(A_filtered, D_filtered,
                                                                             valid_chromosomes, 'temp_dir',
                                                                             num_organisms,
                                                                             true_labels=ground_truth_filtered)
    print(f'The scSplit ARI is {ari_score_scsplit} \n Took {time_scsplit} seconds')

if not params['no_souporcell']:
    (sorc_true_labels, sorc_pred_labels,
     ari_score_souporcell, time_souporcell) = utils.run_souporcell(A_filtered, D_filtered,
                                                                             valid_chromosomes, 'temp_dir',
                                                                             num_organisms,
                                                                             true_labels=ground_truth_filtered,
                                                                             threads=params['num_threads'])
    print(f'The souporcell ARI is {ari_score_souporcell} \n Took {time_souporcell} seconds')

if not params['no_vireo']:
    # Run Vireo
    assignments_vireo, ari_score_vireo, time_vireo = utils.run_vireo(A_filtered, D_filtered,
                                                                     num_organisms, ground_truth_filtered)
    print(f'The vireo homogeneity score is {ari_score_vireo} \n Took {time_vireo} seconds')

# Make the save dir if needed
if not os.path.exists(params['save_dir']):
    os.makedirs(params['save_dir'])

# Save params and results
with open(f"{params['save_dir']}params_run_{params['run_num']}.json", 'w') as f:
    json.dump(params, f)

results_dict = {}
assignments_dict = {}

if not params['no_demuxHMM']:
    results_dict['hmm_score'] = [ari_score_hmm]
    results_dict['hmm_time'] = [time_hmm]
    results_dict['emb_remaining'] = emb_remaining
    results_dict['cells_remaining'] = cells_remaining
    assignments_dict['assignments_hmm'] = assignments_hmm
    assignments_dict['gt_hmm'] = ground_truth_filtered
if not params['no_scsplit']:
    results_dict['scSplit_ARI'] = [ari_score_scsplit]
    results_dict['scSplit_time'] = [time_scsplit]
if not params['no_souporcell']:
    results_dict['souporcell_ARI'] = [ari_score_souporcell]
    results_dict['souporcell_time'] = [time_souporcell]
    assignments_dict['assignments_souporcell'] = sorc_pred_labels
    assignments_dict['gt_souporcell'] = sorc_true_labels
if not params['no_vireo']:
    results_dict['vireo_ARI'] = [ari_score_vireo]
    results_dict['vireo_time'] = [time_vireo]
    assignments_dict['assignments_vireo'] = assignments_vireo
    assignments_dict['gt_vireo'] = ground_truth_filtered

results_df = pd.DataFrame(results_dict)
results_df.to_csv(f'{params["save_dir"]}results_run_{params["run_num"]}.csv')

assignments_df = pd.DataFrame(assignments_dict)
assignments_df.to_csv(f'{params["save_dir"]}assignments_run_{params["run_num"]}.csv')

# Handle scsplit separately due to its slightly weird results format
assignments_dict_scsplit = {}
if not params['no_scsplit']:
    assignments_dict_scsplit['assignments_scsplit'] = assignments_scsplit
    assignments_dict_scsplit['gt_scsplit'] = gt_filt_scsplit
    scsplit_df = pd.DataFrame(assignments_dict_scsplit)
    scsplit_df.to_csv(f'{params["save_dir"]}assignments_scsplit_run_{params["run_num"]}.csv')
