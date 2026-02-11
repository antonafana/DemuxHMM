import numpy as np
from datetime import datetime
import vireoSNP
from numba import njit
import re
from demuxHMM import HMMModel
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import subprocess
from scipy import sparse
from scipy.io import mmwrite
from scipy.optimize import linear_sum_assignment
import sklearn.preprocessing
from pysam import VariantFile
import data_simulator
import pickle
import os, uuid


@njit(fastmath=True)
def calculate_drosophila_rate(chrom, coord):
    """
    Given a chromosome and coordinates, calculates the rate of recombination for Drosophila according to
    https://github.com/asfistonlavie/RRC/blob/master/RRC-open-v2.2.1.pl

    Args:
        chrom: (str) chromosome
        coord: (int) coordinate

    Returns: (float) rate of recombination
    """
    coord = coord / 10**6

    # !Temporary until we incorporate L and R into the solver!
    if chrom == "2R" or chrom == "2":
        chrom = "2L"
    elif chrom == "3R" or chrom == "3":
        chrom = "3L"

    if chrom == "2L":
        if 0.53 <= coord <= 18.87:
            rate = 2.58909 + 0.40558 * coord - 0.02886 * coord ** 2
        else:
            rate = 0
    elif chrom == "2R":
        if 1.87 <= coord <= 20.86:
            rate = -1.435345 + 0.698356 * coord - 0.023364 * coord ** 2
        else:
            rate = 0
    elif chrom == "3L":
        if 0.75 <= coord <= 19.02:
            rate = 2.940315 + 0.19103 * coord - 0.017793 * coord ** 2
        else:
            rate = 0
    elif chrom == "3R":
        if 2.58 <= coord <= 27.44:
            rate = -1.629112 + 0.484498 * coord - 0.012117 * coord ** 2
        else:
            rate = 0
    elif chrom == "X":
        if 1.22 <= coord <= 21.21:
            rate = 1.151769 + 0.588228 * coord - 0.029142 * coord ** 2
        else:
            rate = 0
    elif chrom == "4":
        rate = 0
    else:
        rate = None

    return rate

def find_contig_length(data: str, contig_id: str) -> int:
    """
    Finds the length of a contig given a VCF header
    Args:
        data: The VCF header
        contig_id: The name of the contig

    Returns: An integer indicating the length of the contig
    """
    # Regular expression pattern to match the contig line
    pattern = rf"##contig=<ID={contig_id},length=(\d+)>"
    match = re.search(pattern, data)

    # Return the length if a match is found, else return None
    if match:
        return int(match.group(1))
    else:
        raise Exception("No contig length found!")

def make_vireo_matrices(A, D):
    """
    Makes the matrices for Vireo given the HMM style A and D matrices.
    Args:
        A: A list of n x num_var[c] matrices of variant counts for every c in chromosomes
        D: A list of n x num_var[c] matrices of counts for every c in chromosomes

    Returns:
    A_vireo, D_vireo
    """
    num_chroms = len(A)
    snps_per_chr = [A[i].shape[1] for i in range(num_chroms)]
    total_snps = np.sum(snps_per_chr)
    total_cells = A[0].shape[0]

    A_vireo = np.zeros((total_snps, total_cells))
    D_vireo = np.zeros((total_snps, total_cells))


    for i in range(num_chroms):
        A_vireo[:snps_per_chr[i], :] = A[i].T
        D_vireo[:snps_per_chr[i], :] = D[i].T

    return A_vireo, D_vireo

def run_vireo(A, D, num_organisms, ground_truth):
    """
    Runs the vireo simulation given the A, and D HMM matrices.
    Args:
        A: A list of n x num_var[c] matrices of variant counts for every c in chromosomes
        D: A list of n x num_var[c] matrices of counts for every c in chromosomes
        num_organisms: The number of clusters
        ground_truth: The ground truth clustering
    Returns: assignments_vireo, ARI, time_taken
    """
    start_time = datetime.now()

    A_vireo, D_vireo = make_vireo_matrices(A, D)
    vireo_model = vireoSNP.BinomMixtureVB(n_var=A_vireo.shape[0], n_cell=A_vireo.shape[1], n_donor=num_organisms)
    vireo_model.fit(AD=A_vireo, DP=D_vireo, verbose=True, n_init=50, max_iter=5000, max_iter_pre=2500)
    assignments_vireo = np.argmax(vireo_model.ID_prob, axis=1)

    ari_score_vireo = adjusted_rand_score(ground_truth, assignments_vireo)

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()

    return assignments_vireo, ari_score_vireo, time_taken

def run_demuxHMM(A, D, num_organisms, tolerance, ground_truth, mean_transitions=None,
                 num_threads=10, num_repeats=30, theta_init=None, T_init=None, pi_init=None):
    time_start = datetime.now()

    best_model = None
    best_model_score = -np.inf
    all_models = []

    for _ in range(num_repeats):
        model = HMMModel(A, D, num_organisms, mean_transitions,
                         threads=num_threads, theta_init=theta_init, T_init=T_init, pi_init=pi_init)
        model.solve(tolerance, iters=50, num_inits=0)
        model_out = model.get_output()
        all_models.append(model_out)
        assignments = np.argmax(model_out['z_probs'], axis=1)
        if ground_truth is not None:
            ari_score_current = adjusted_rand_score(ground_truth, assignments)
            print("Current ARI:", ari_score_current)

        if model_out['probability'] > best_model_score:
            best_model_score = model_out['probability']
            best_model = model_out

    assignments_new = np.argmax(best_model['z_probs'], axis=1)
    if ground_truth is not None:
        ari_score = adjusted_rand_score(ground_truth, assignments_new)
    else:
        ari_score = None
    time_end = datetime.now()
    time_taken = (time_end - time_start).total_seconds()

    return assignments_new, ari_score, time_taken, best_model



def write_allele_matrices(A, D, valid_chromosomes, temp_dir_loc):
    """
    Writes alternate (A) and reference (R) allele count matrices to CSV format,
    concatenating across all chromosomes.
    """
    os.makedirs(temp_dir_loc, exist_ok=True)

    A_dfs, R_dfs = [], []

    for i, chrom in enumerate(valid_chromosomes):
        snv_labels = [chrom + ":" + str(i) for i in range(A[i].shape[1])]

        A_matrix = A[i]
        R_matrix = D[i] - A[i]

        barcodes = [f"bc_{j+1}" for j in range(A_matrix.shape[0])]

        A_df = pd.DataFrame(A_matrix.T, index=snv_labels, columns=barcodes, dtype=int)
        R_df = pd.DataFrame(R_matrix.T, index=snv_labels, columns=barcodes, dtype=int)

        A_df.insert(0, "SNV", A_df.index)
        R_df.insert(0, "SNV", R_df.index)

        A_dfs.append(A_df)
        R_dfs.append(R_df)

    A_all = pd.concat(A_dfs, axis=0)
    R_all = pd.concat(R_dfs, axis=0)

    unique_tag = uuid.uuid4().hex
    A_path = os.path.join(temp_dir_loc, f"A_{unique_tag}.csv")
    R_path = os.path.join(temp_dir_loc, f"R_{unique_tag}.csv")

    A_all.to_csv(A_path, index=False)
    R_all.to_csv(R_path, index=False)

    return A_path, R_path


def run_scsplit(A, D, valid_chromosomes, temp_dir_loc,
                num_organisms, true_labels):
    """
    Runs scSplit with Ref/Alt matrices and parses assignments.
    Args:
        A, D, snp_dfs, valid_chromosomes, temp_dir_loc: inputs for allele matrix writing
        num_organisms: expected number of mixed samples (-n); use 0 for autodetect
        true_labels: optional dict mapping barcode -> true cluster for ARI calculation
    Returns:
        assignments_scsplit (dict),
        ari_score_scsplit (float or None),
        time_scsplit (float, seconds)
    """

    # Write allele matrices
    A_csv, R_csv = write_allele_matrices(A, D, valid_chromosomes, temp_dir_loc)

    # Unique run identifier
    uid = uuid.uuid4().hex

    # Prepare output dir
    out_dir = os.path.join(temp_dir_loc, f"scsplit_output_{uid}")
    os.makedirs(out_dir, exist_ok=True)

    # Run scSplit
    cmd = [
        "python", "scSplit/scSplit", "run",
        "-r", R_csv,
        "-a", A_csv,
        "-n", str(num_organisms),
        "-o", out_dir
    ]

    start = datetime.now()
    subprocess.run(cmd, check=True)
    time_elapsed = (datetime.now() - start).total_seconds()

    # Parse assignments
    assign_file = os.path.join(out_dir, "scSplit_result.csv")
    if not os.path.exists(assign_file):
        raise FileNotFoundError(f"scSplit output not found: {assign_file}")

    assign_df = pd.read_csv(assign_file, sep=r"\s+|,", engine="python")  # handle space or comma
    # Expect columns: Barcode, Cluster
    assignments = dict(zip(assign_df["Barcode"], assign_df["Cluster"]))

    # Compute ARI if true labels given
    ari_score = None
    # Map clusters like "SNG-0" -> 0, "SNG-1" -> 1, ...
    cluster_map = {cl: i for i, cl in enumerate(sorted(assign_df["Cluster"].unique()))}

    pred_labels = []
    true_labels_matched = []

    for bc, clust in assignments.items():
        if bc.startswith("bc_"):
            idx = int(bc.split("_")[1])
            if idx < len(true_labels):
                pred_labels.append(cluster_map[clust])
                true_labels_matched.append(true_labels[idx])

    if pred_labels and true_labels_matched:
        ari_score = adjusted_rand_score(true_labels_matched, pred_labels)

    return pred_labels, true_labels_matched, ari_score, time_elapsed

def write_vartrix_matrices(A, D, valid_chromosomes, temp_dir_loc):
    """
    Writes ALT and REF allele matrices in vartrix (Matrix Market) format for souporcell.
    Produces: alt.mtx, ref.mtx, barcodes.tsv
    """
    os.makedirs(temp_dir_loc, exist_ok=True)

    # Unique run identifier to allow multiple instances at once
    uid = uuid.uuid4().hex

    all_A = []
    all_R = []

    # Concatenate across chromosomes
    for i, chrom in enumerate(valid_chromosomes):
        A_matrix = A[i]            # shape: n_cells x n_variants
        R_matrix = D[i] - A[i]

        all_A.append(A_matrix)
        all_R.append(R_matrix)

    # Souporcell3 is highly sensitive to zeros. We need to add a pseudocount to get good performance
    A_full = np.hstack(all_A) + 1   # concatenate across variants
    R_full = np.hstack(all_R) + 1
    print('sums', A_full[:, 1].sum(), R_full[:, 1].sum())

    n_cells, _ = A_full.shape
    barcodes = [f"bc_{j}" for j in range(n_cells)]

    # Convert to sparse COO matrices (vartrix expects variants as rows, cells as columns)
    # Force integer dtype (32-bit is plenty for counts)
    A_sparse = sparse.coo_matrix(A_full.T.astype(np.int32))
    R_sparse = sparse.coo_matrix(R_full.T.astype(np.int32))

    alt_path = os.path.join(temp_dir_loc, f"alt_{uid}.mtx")
    ref_path = os.path.join(temp_dir_loc, f"ref_{uid}.mtx")
    bc_path = os.path.join(temp_dir_loc, f"barcodes_{uid}.tsv")

    mmwrite(alt_path, A_sparse)
    mmwrite(ref_path, R_sparse)

    with open(bc_path, "w") as f:
        f.write("\n".join(barcodes) + "\n")

    return alt_path, ref_path, bc_path


def run_souporcell(A, D, valid_chromosomes, temp_dir_loc,
                   num_organisms, true_labels=None, threads=8):
    """
    Runs souporcell clustering with vartrix-style matrices.
    Args:
        A, D, snp_dfs, valid_chromosomes, temp_dir_loc: input matrices/SNP info
        num_organisms: number of clusters (k)
        true_labels: optional list of ground-truth cluster assignments (indexed by bc_i)
        threads: number of threads
    Returns:
        assignments, ari_score, runtime
    """
    # Write matrices
    alt_mtx, ref_mtx, barcodes_tsv = write_vartrix_matrices(
        A, D, valid_chromosomes, temp_dir_loc
    )

    # Unique run identifier
    uid = uuid.uuid4().hex
    clusters_file = os.path.join(temp_dir_loc, f"clusters_tmp_{uid}.tsv")

    # Build command
    cmd = [
        "souporcell/souporcell/target/release/souporcell",
        "-a", alt_mtx,
        "-r", ref_mtx,
        "-b", barcodes_tsv,
        "-k", str(num_organisms),
        "-t", str(threads),
        "-s", "true",
        "-m", "khm"
    ]

    start = datetime.now()
    with open(clusters_file, "w") as fout:
        subprocess.run(cmd, check=True, stdout=fout)
    time_elapsed = (datetime.now() - start).total_seconds()

    # Parse assignments (souporcell outputs: barcode, cluster, then log-likelihoods, no header)
    assign_df = pd.read_csv(clusters_file, sep="\t", header=None)
    assign_df = assign_df.iloc[:, :2]  # keep only first two columns
    assign_df.columns = ["barcode", "cluster"]

    assignments = dict(zip(assign_df["barcode"], assign_df["cluster"]))

    # Compute ARI if truth available
    ari_score = None
    if true_labels is not None:
        pred_labels = []
        true_labels_matched = []

        for bc, clust in assignments.items():
            if bc.startswith("bc_"):
                idx = int(bc.split("_")[1])
                if idx < len(true_labels):
                    pred_labels.append(clust)
                    true_labels_matched.append(true_labels[idx])

        if pred_labels:
            ari_score = adjusted_rand_score(true_labels_matched, pred_labels)

    return true_labels, pred_labels, ari_score, time_elapsed

def count_transitions(chrom):
    """
    Finds the number of state transitions on each allele of a chromosome.
    Args:
        chrom: The chromosome to analyze
    Returns: number of transitions on allele 1 and number of transitions on allele 2

    """
    num_transitions = [0, 0]

    for c in range(chrom.states.shape[0]):
        for j in range(1, chrom.states.shape[1]):
            if chrom.states[c, j - 1] != chrom.states[c, j]:
                num_transitions[c] += 1

    return num_transitions[0], num_transitions[1]

def best_cluster_label_mapping(cluster_assignments, ground_truth_labels):
    """
    Find the best mapping between cluster assignments and ground truth labels.

    Args:
        cluster_assignments (array-like): predicted cluster IDs (ints)
        ground_truth_labels (array-like): true class labels (ints)

    Returns:
        dict: mapping from cluster ID -> ground truth label
    """
    cluster_assignments = np.asarray(cluster_assignments)
    ground_truth_labels = np.asarray(ground_truth_labels)

    # Get unique ids
    clusters = np.unique(cluster_assignments)
    labels = np.unique(ground_truth_labels)

    # Build confusion matrix
    confusion = np.zeros((len(clusters), len(labels)), dtype=np.int64)
    for i, c in enumerate(clusters):
        for j, l in enumerate(labels):
            confusion[i, j] = np.sum((cluster_assignments == c) & (ground_truth_labels == l))

    # Hungarian algorithm maximizes match → so we negate counts to minimize
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Build mapping
    mapping = {int(clusters[i]): int(labels[j]) for i, j in zip(row_ind, col_ind)}
    return mapping

def genotype_error_across_organisms(organism_list, inferred, valid_chromosomes, cluster_map):
    """
    Compute genotype error across organisms and chromosomes.

    Args:
        organism_list : list
            Each element has `.chromosomes`, a dict {chrom_name: array of shape (2, num_alleles)}.
        inferred : np.ndarray
            Shape (num_chromosomes, num_organisms, num_alleles[chromosome]), values in {0,1,2}.
        valid_chromosomes : list of str
            Ordered chromosome names; must match axis 0 order in `inferred`.

    Returns:
        dict with:
            - 'per_organism_error': np.ndarray of shape (num_organisms,)
            - 'per_chromosome_error': dict {chrom_name: float}
            - 'mean_error': float
    """
    num_chroms = len(inferred)
    num_orgs = len(cluster_map.keys())
    num_sites = np.array([len(inferred[i][0]) for i in range(num_chroms)])
    if num_chroms != len(valid_chromosomes):
        raise ValueError("Mismatch between inferred chromosomes and valid_chromosomes length")

    # Track per-organism and per-chromosome errors
    per_org_errors = []

    for org_idx in cluster_map.keys():
        chrom_errors = []
        mapped_org_idx = int(cluster_map[org_idx]) # Map is HMM keys to GT keys

        for chrom_idx, chrom_name in enumerate(valid_chromosomes):
            # Sum the chromatids together to get the ground truth states
            gt_states = organism_list[mapped_org_idx].chromosomes[chrom_name].states
            gt_states_joined = gt_states.sum(axis=0)

            # Get the inferred states
            inferred_states = inferred[chrom_idx][org_idx]
            num_not_matching = np.sum(inferred_states != gt_states_joined)
            chrom_errors.append(num_not_matching)

            #print(f'Organism {org_idx} has {num_not_matching} mismatches on {chrom_idx}')

        per_org_errors.append(np.sum(chrom_errors) / np.sum(num_sites))

    return {
        "per_organism_error": per_org_errors,
        "mean_error": float(np.mean(per_org_errors)),
    }

def simulate_dataset(params, snp_dfs_chrom, adata):
    contig_chrom_map = {'CM010569.1': 'X', 'CM010570.1': '2', 'CM010571.1': '2',
                        'CM010572.1': '3', 'CM010573.1': '3', 'CM010574.1': '4'}
    valid_chromosomes = ['X', '2', '3']  # We filter out 4 because it is always conserved and adds no info
    num_chr = len(valid_chromosomes)

    # Read vcf and get basic stats
    vcf = VariantFile(params['VCF_FILENAME'], 'r')

    # Make the length of the chromosome the length of its longest arms. Contigs are assumed to be at least one arm.
    chrom_lengths = {chrom: 0 for chrom in valid_chromosomes}
    for contig in params['valid_contigs']:
        if contig_chrom_map[contig] in valid_chromosomes:
            chrom_lengths[contig_chrom_map[contig]] = np.max([find_contig_length(str(vcf.header), contig),
                                                              chrom_lengths[contig_chrom_map[contig]]])

    # The adata gives contigs not chromosomes, this fixes it. Also normalize to make prob distributions.
    adata = adata[:, adata.var['chromosome'].isin(params['valid_contigs'])].copy()
    adata.var['chromosome'] = adata.var['chromosome'].apply(lambda x: contig_chrom_map[x])
    adata = adata[:, adata.var['chromosome'].isin(valid_chromosomes)].copy()

    row_sums = adata.X.sum(axis=1)
    print(f'Adata before', adata.shape)
    adata = adata[row_sums >= params['cutoff_UMI'], :].copy()
    print(f'Adata After', adata.shape)

    adata.X = sklearn.preprocessing.normalize(adata.X, norm='l1')

    # Make the gene to snp mapping for each chromosome
    gene_to_snp_maps = {}

    for chrom in valid_chromosomes:
        genes_chrom = list(adata.var[adata.var['chromosome'] == chrom]['genes'].unique())
        chrom_snp_df = snp_dfs_chrom[chrom]
        gene_to_snp_maps[chrom] = {}

        for gene in genes_chrom:
            snps_featured = list(chrom_snp_df[chrom_snp_df['gene_name'] == gene].index)
            gene_to_snp_maps[chrom][gene] = snps_featured

    # Fill recombination rates in intervals of 100 because they don't change that much
    interval_size = 100

    # Make arrays of rates for each base in each chromosome
    recombination_rates = {chrom: np.zeros(np.astype(np.ceil(chrom_lengths[chrom] / interval_size), int)) \
                           for chrom in valid_chromosomes}
    expected_cross_numbers = {chrom: 0 for chrom in valid_chromosomes}

    for chrom in valid_chromosomes:
        num_intervals = np.astype(np.ceil(chrom_lengths[chrom] / interval_size), int)

        for i in range(num_intervals):
            start_base = interval_size * i

            recombination_rates[chrom][i] = np.max([calculate_drosophila_rate(chrom, start_base), 0])

            # Integrate the centimorgan rates to get centimorgans
            expected_cross_numbers[chrom] += recombination_rates[chrom][i] * 10 ** (-6) * interval_size

        # Normalize the recombination rates
        if expected_cross_numbers[chrom] != 0:
            recombination_rates[chrom] = (recombination_rates[chrom] / expected_cross_numbers[chrom]) * 10 ** (
                -6) * interval_size

    from datetime import datetime
    print('Starting breeding experiment')
    print(datetime.now())

    # Make a wild type organism's chromosomes and a variant organism's chromosomes
    var_adam_chromosomes = {}
    wild_type_eve_chromosomes = {}

    for chrom in valid_chromosomes:
        chrom_len = chrom_lengths[chrom]
        chrom_snp_locs = snp_dfs_chrom[chrom]['loc'].to_numpy()

        # Male (Adam): make X haploid (one chromosome), others diploid
        if chrom == 'X':
            var_states = np.ones(shape=(1, len(chrom_snp_locs)))
        else:
            var_states = np.ones(shape=(2, len(chrom_snp_locs)))

        # Female (Eve): still diploid and wild-type
        wild_type_states = np.zeros(shape=(2, len(chrom_snp_locs)))

        var_adam_chromosomes[chrom] = data_simulator.Chromosome(
            chrom,
            chrom_len,
            chrom_snp_locs,
            var_states,
            gene_to_snp_maps[chrom],
            recombination_rate=recombination_rates[chrom],
            expected_cross_count=expected_cross_numbers[chrom],
            interval_size=interval_size,
        )

        wild_type_eve_chromosomes[chrom] = data_simulator.Chromosome(
            chrom,
            chrom_len,
            chrom_snp_locs,
            wild_type_states,
            gene_to_snp_maps[chrom],
            recombination_rate=recombination_rates[chrom],
            expected_cross_count=expected_cross_numbers[chrom],
            interval_size=interval_size,
        )

    # Compile the chromosomes to form organisms
    var_adam = data_simulator.Organism(valid_chromosomes, var_adam_chromosomes, 'M')
    wild_type_eve = data_simulator.Organism(valid_chromosomes, wild_type_eve_chromosomes, 'F')
    starting_organism_list = [var_adam, wild_type_eve]

    # In a real experiment we would likely breed a huge pool
    # and then subset it down to the number of individuals we want. Simulate this too.
    offspring_to_use = np.max([params['offspring_per_generation'], 500])

    experiment = data_simulator.BreedingExperiment(starting_organism_list,
                                                   offspring_to_use,
                                                   males_cross_over=params['males_cross_over'])

    if params['backcross']:
        final_gen = experiment.run_experiment(params['num_generations'],
                                                n_proc=params['num_threads'],
                                                backcross=var_adam)
    else:
        final_gen = experiment.run_experiment(params['num_generations'],
                                                n_proc=params['num_threads'])

    if params['gender_filter'] == 'male':
        final_gen.organism_list = [org for org in final_gen.organism_list if org.gender == 'M']
    elif params['gender_filter'] == 'female':
        final_gen.organism_list = [org for org in final_gen.organism_list if org.gender == 'F']

    # Sample cells from the final generation
    final_gen.organism_list = final_gen.organism_list[:params['offspring_per_generation']]
    wt_umi, var_umi = final_gen.sample_cells(params['avg_UMI'],
                                             params['variance_UMI'],
                                             params['num_cells_per_org'],
                                             adata, n_processes=params['num_threads'])

    print('Done cell sampling')
    print(datetime.now())

    # Convert to a data format compatible with the HMM model
    A = var_umi
    D = [var_umi[i].copy() + wt_umi[i] for i in range(num_chr)]

    # count SNPs for genes that have >0 expression in the adata used for sampling
    expressed_genes = set(adata.var[adata.X.sum(axis=0) > 0]['genes'].values)
    counts = []
    for chrom in valid_chromosomes:
        mapping = gene_to_snp_maps[chrom]
        n = 0
        for g, snps in mapping.items():
            if g in expressed_genes:
                n += len(snps)
        counts.append((chrom, n))
    print("SNPs in expressed genes per chrom:", counts)

    return A, D, final_gen.organism_list

def get_transitions(num_generations):
    """
    Loads a pre-computed T matrix for a simulated dataset.
    Args:
        num_generations: The number of generations simulated

    Returns:T

    """
    transitions = pickle.load(open('data_dir/transitions/transitions_{' + str(num_generations) + '}_gen.pickle', 'rb'))
    return transitions

