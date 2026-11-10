import sys

import numpy as np
from datetime import datetime
import vireoSNP
from vireoSNP.utils import vireo_doublet
from numba import njit
import re
from demuxHMM import HMMModel
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import subprocess
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import vstack
import sklearn.preprocessing
from pysam import VariantFile
import data_simulator
import pickle
import os, uuid
import multiprocessing

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
    A_vireo = vstack([Ac.T for Ac in A], format="csc")
    D_vireo = vstack([Dc.T for Dc in D], format="csc")

    return A_vireo, D_vireo

def _fit_vireo(model, AD, DP, max_iter):
    model.fit(AD, DP, min_iter=5, max_iter=max_iter, verbose=False)
    return model

def fit_vireo(AD, DP, n_donor, n_init=50, max_iter=5000,
              max_iter_init=20, nproc=1):
    """
    Wrapper matching the CLI's multiple-initialization strategy.
    Returns the fitted Vireo model. We need this because their main model
    code ignore the n_inits parameters...
    """

    _models_all = []
    for im in range(n_init):
        _model = vireoSNP.Vireo(n_var=AD.shape[0], n_cell=AD.shape[1],
                                n_donor=n_donor, learn_GT=True)
        _models_all.append(_model)

    if nproc > 1:
        pool = multiprocessing.Pool(processes=nproc)
        result = []
        for im in range(n_init):
            result.append(pool.apply_async(_fit_vireo,
                (_models_all[im], AD, DP, max_iter_init),
                callback=None))
        pool.close()
        pool.join()
        _models_all = [res.get() for res in result]
    else:
        for im in range(n_init):
            _models_all[im].fit(AD, DP, min_iter=5,
                                max_iter=max_iter_init, verbose=False)

    elbo_all = np.array([x.ELBO_[-1] for x in _models_all])
    _idx = np.argmax(elbo_all)
    model = _models_all[_idx]

    model.fit(AD, DP, min_iter=5, max_iter=max_iter, verbose=True)

    return model

def run_vireo(A, D, num_organisms, ground_truth, detect_doublets=False, doublet_prior=0, num_threads=1):
    """
    Runs the vireo simulation given the A, and D HMM matrices.
    Args:
        A: A list of n x num_var[c] matrices of variant counts for every c in chromosomes
        D: A list of n x num_var[c] matrices of counts for every c in chromosomes
        num_organisms: The number of clusters
        ground_truth: The ground truth clustering (list of len num_cells)
    Returns: assignments_vireo, ARI, time_taken
    """
    start_time = datetime.now()

    A_vireo, D_vireo = make_vireo_matrices(A, D)
    vireo_model = fit_vireo(A_vireo, D_vireo, n_donor=num_organisms, nproc=num_threads)
    assignments_vireo = np.argmax(vireo_model.ID_prob, axis=1)

    prob_doublet_out = None
    if detect_doublets:
        prob_doublet, prob_singlet, logLik_ratio = vireo_doublet.predict_doublet(vireo_model, A_vireo, D_vireo,
                                                                                 update_GT=False,
                                                                                 doublet_rate_prior=doublet_prior)
        prob_doublet_out = np.max(prob_doublet, axis=1)
        print('Prob doublet is: ', prob_doublet_out)

    ari_score_vireo = None
    if ground_truth is not None:
        ari_score_vireo = adjusted_rand_score(ground_truth, assignments_vireo)

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()

    return assignments_vireo, ari_score_vireo, time_taken, prob_doublet_out, vireo_model

def _run_demuxHMM_repeat(args):
    (A, D, num_organisms, mean_transitions, num_threads,
     theta_init, T_init, pi_init, z_init, T_eps, tolerance, ground_truth, detect_doublets, do_theta_update) = args

    model = HMMModel(A, D, num_organisms, mean_transitions, threads=num_threads,
                     theta_init=theta_init, T_init=T_init, pi_init=pi_init,
                     z_init=z_init, T_eps=T_eps, do_theta_update=do_theta_update)
    model.solve(tolerance, iters=50)

    model_out = model.get_output()
    assignments = np.argmax(model_out['z_probs'], axis=1)

    if ground_truth is not None:
        ari_score = adjusted_rand_score(ground_truth, assignments)
    else:
        ari_score = None

    model_obj = None
    if detect_doublets:
        model_obj = model

    return model_out, model_obj, ari_score

def run_demuxHMM(A, D, num_organisms, tolerance, ground_truth, mean_transitions=None,
                 num_threads=10, num_repeats=30,
                 theta_init=None, T_init=None, pi_init=None, z_init=None,
                 detect_doublets=False, doublet_prior=0,
                 T_eps=10e-10, num_processes=20, do_theta_update=False):
    time_start = datetime.now()

    # Do as many initializations in parallel as possible
    args = (A, D, num_organisms, mean_transitions, num_threads, theta_init, T_init, pi_init, z_init, T_eps, tolerance,
            ground_truth, detect_doublets, do_theta_update)
    tasks = [args] * num_repeats

    best_model = None
    best_model_obj = None
    best_model_score = -np.inf
    all_assignments = []

    if num_processes > 1:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(num_processes) as pool:
            results = pool.map(_run_demuxHMM_repeat, tasks)

        for model_out, model, ari_score_current in results:
            assignments = np.argmax(model_out['z_probs'], axis=1)
            all_assignments.append(assignments)

            if ground_truth is not None:
                print("Current ARI:", ari_score_current)

            if model_out['probability'] > best_model_score:
                best_model_score = model_out['probability']
                best_model = model_out
                best_model_obj = model
    else:
        for arg in tasks:
            model_out, model_obj, ari_score = _run_demuxHMM_repeat(arg)
            assignments = np.argmax(model_out['z_probs'], axis=1)
            all_assignments.append(assignments)

            if model_out['probability'] > best_model_score:
                best_model_score = model_out['probability']
                best_model = model_out
                best_model_obj = model_obj

    all_assignments = np.array(all_assignments).T

    # Optionally detect doublets
    if detect_doublets:
        best_model['doublet_probs'] = best_model_obj.identify_doublets(k=int(num_organisms),
                                                                       prior_doublet=doublet_prior)
    assignments_new = np.argmax(best_model['z_probs'], axis=1)

    if ground_truth is not None:
        ari_score = adjusted_rand_score(ground_truth, assignments_new)
    else:
        ari_score = None
    time_end = datetime.now()
    time_taken = (time_end - time_start).total_seconds()

    return assignments_new, ari_score, time_taken, best_model, all_assignments

def write_allele_matrices(A, D, valid_chromosomes, temp_dir_loc):
    """
    Writes alternate (A) and reference (R) allele count matrices to CSV format,
    concatenating across all chromosomes.
    """
    os.makedirs(temp_dir_loc, exist_ok=True)

    A_dfs, R_dfs = [], []

    for i, chrom in enumerate(valid_chromosomes):
        snv_labels = [chrom + ":" + str(i) for i in range(A[i].shape[1])]

        A_matrix = A[i].toarray()
        R_matrix = (D[i] - A[i]).toarray()

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
                num_organisms, true_labels, detect_doublets=False, doublet_prior=0):
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
    cmd = [sys.executable, # Use the same python that we are running this code with
            "scSplit/scSplit", "run",
            "-r", R_csv,
            "-a", A_csv,
            "-n", str(num_organisms),
            "-o", out_dir
           ]

    if detect_doublets:
        cmd.append('-d')
        cmd.append(str(doublet_prior))

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

    # Only map singlet clusters to organism IDs
    singlet_clusters = sorted(cl for cl in assign_df["Cluster"].unique() if cl.startswith("SNG-"))
    cluster_map = {cl: i for i, cl in enumerate(singlet_clusters)}

    pred_labels = []
    pred_is_doublet = []
    true_labels_matched = []
    indices = []

    for bc, clust in assignments.items():
        # Header
        if not bc.startswith("bc_"):
            continue

        # Extract cell index and perform sanity check
        # The cell index is 1-indexed in scsplit
        idx = int(bc.split("_")[1]) - 1
        if idx >= len(true_labels):
            continue

        indices.append(idx)

        # Treat singlets and doublets separately
        if clust.startswith("DBL-"):
            pred_labels.append(-1)
            pred_is_doublet.append(True)
        else:
            pred_labels.append(cluster_map[clust])
            pred_is_doublet.append(False)

        true_labels_matched.append(true_labels[idx])

    ari_score = adjusted_rand_score(np.asarray(true_labels_matched), np.asarray(pred_labels))

    return pred_labels, pred_is_doublet, true_labels_matched, ari_score, time_elapsed, indices

def _write_matrix_market(path, rows, cols, data, shape):
    """
    Writes a Matrix Market coordinate matrix.
    """
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write("%\n")
        f.write(f"{shape[0]} {shape[1]} {len(data)}\n")
        for r, c, v in zip(rows, cols, data):
            # Matrix Market uses 1-based indexing
            f.write(f"{r + 1} {c + 1} {int(v)}\n")


def write_vartrix_matrices(A, D, valid_chromosomes, temp_dir_loc):
    """
    Writes ALT and REF allele matrices in vartrix (Matrix Market) format for souporcell.
    Produces: alt.mtx, ref.mtx, barcodes.tsv

    ALT and REF are written with the union of nonzero coordinates so that
    the two files have identical sparsity patterns.
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

    A_full = sparse.hstack(all_A)
    R_full = sparse.hstack(all_R)

    print('sums', A_full[:, 1].sum(), R_full[:, 1].sum())

    n_cells, _ = A_full.shape
    barcodes = [f"bc_{j}" for j in range(n_cells)]

    # Convert to variant x cell COO matrices
    A_sparse = A_full.T.tocoo()
    R_sparse = R_full.T.tocoo()
    A_sparse.sum_duplicates()
    R_sparse.sum_duplicates()

    # There is a bug in troublet that expects coordinates to match between A and R
    # We need to use the union of the coordinate set between the two matrices
    A_dict = {(r, c): v for r, c, v in zip(A_sparse.row, A_sparse.col, A_sparse.data)}
    R_dict = {(r, c): v for r, c, v in zip(R_sparse.row, R_sparse.col, R_sparse.data)}

    # Union of coordinates
    coords = sorted(set(A_dict) | set(R_dict))

    rows = [r for r, c in coords]
    cols = [c for r, c in coords]

    A_data = [A_dict.get((r, c), 0) for r, c in coords]
    R_data = [R_dict.get((r, c), 0) for r, c in coords]

    alt_path = os.path.join(temp_dir_loc, f"alt_{uid}.mtx")
    ref_path = os.path.join(temp_dir_loc, f"ref_{uid}.mtx")
    bc_path = os.path.join(temp_dir_loc, f"barcodes_{uid}.tsv")

    # This preserves zeros (discarding them causes a kernel panic)
    _write_matrix_market(alt_path, rows, cols, A_data, A_sparse.shape)
    _write_matrix_market(ref_path, rows, cols, R_data, R_sparse.shape)

    with open(bc_path, "w") as f:
        f.write("\n".join(barcodes) + "\n")

    return alt_path, ref_path, bc_path

def run_souporcell(A, D, valid_chromosomes, temp_dir_loc,
                   num_organisms, true_labels=None, threads=8, detect_doublets=False,
                   doublet_prior=0, random_seed=None):
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
    alt_mtx, ref_mtx, barcodes_tsv = write_vartrix_matrices(A, D, valid_chromosomes, temp_dir_loc)

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

    if random_seed is not None:
        if not isinstance(random_seed, (int, np.integer)) or not 0 <= random_seed <= 255:
            raise ValueError("random_seed must be an integer between 0 and 255")

        cmd.extend(["--seed", str(random_seed)])

    start = datetime.now()
    with open(clusters_file, "w") as fout:
        subprocess.run(cmd, check=True, stdout=fout)

    # Parse assignments (souporcell outputs: barcode, cluster, then log-likelihoods, no header)
    assign_df = pd.read_csv(clusters_file, sep="\t", header=None)
    assign_df = assign_df.iloc[:, :2]  # keep only first two columns
    assign_df.columns = ["barcode", "cluster"]

    assignments = dict(zip(assign_df["barcode"], assign_df["cluster"]))

    # Compute ARI if truth available
    ari_score = None
    pred_labels = []
    true_labels_matched = []
    indices_fit = []

    for bc, clust in assignments.items():
        if bc.startswith("bc_"):
            idx = int(bc.split("_")[1])
            indices_fit.append(idx)
            pred_labels.append(clust)

            if true_labels is not None:
                if idx < len(true_labels):
                    true_labels_matched.append(true_labels[idx])
                else:
                    raise ValueError("Invalid sorc index", idx)

    if true_labels is not None:
        ari_score = adjusted_rand_score(true_labels_matched, pred_labels)

    pred_is_doublet = None
    doublet_probs = None
    doublet_indices = None

    # Now we can predict doublets if specified
    if detect_doublets:
        cmd = [
            "souporcell/troublet/target/release/troublet",
            "-a", alt_mtx,
            "-r", ref_mtx,
            "-d", str(doublet_prior),
            '--clusters', clusters_file,
        ]

        doublet_file = os.path.join(temp_dir_loc, f'doublets_tmp_{uid}.tsv')
        with open(doublet_file, 'w') as fout:
            subprocess.run(cmd, check=True, stdout=fout)

        # Parse troublet output
        doublet_df = pd.read_csv(doublet_file, sep=r'\s+', engine='python')

        pred_is_doublet = []
        doublet_probs = []
        doublet_indices = []

        for _, row in doublet_df.iterrows():
            bc = row['barcode']

            if not bc.startswith("bc_"):
                continue

            # Save indices to keep the ordering
            idx = int(bc.split('_')[1])
            doublet_indices.append(idx)

            # Extract doublet status and associated probability
            pred_is_doublet.append(row['status'] == 'doublet')
            doublet_probs.append(row['doublet_posterior'])

        doublet_probs = np.array(doublet_probs)

    time_elapsed = (datetime.now() - start).total_seconds()

    return (true_labels, pred_labels, ari_score, time_elapsed, indices_fit,
            pred_is_doublet, doublet_probs, doublet_indices)

def count_transitions(chrom):
    """
    Finds the number of state transitions on each allele of a chromosome.
    Args:
        chrom: The chromosome to analyze
    Returns: number of transitions on allele 1 and number of transitions on allele 2

    """
    num_transitions = [0, 0]

    for c in range(chrom.shape[0]):
        for j in range(1, chrom.shape[1]):
            if chrom[c, j - 1] != chrom[c, j]:
                num_transitions[c] += 1

    return num_transitions[0], num_transitions[1]

def best_cluster_label_mapping(cluster_assignments, ground_truth_labels):
    """
    Find the best mapping between cluster assignments and ground truth labels.

    Args:
        cluster_assignments (array-like): predicted cluster IDs (ints)
        ground_truth_labels (array-like): true class labels (ints)

    Returns:
        dict: mapping from cluster ID to ground truth label
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

    # Hungarian algorithm maximizes match, so we negate counts to minimize
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
            gt_states = organism_list[mapped_org_idx]['chromosomes'][chrom_name]
            gt_states_joined = gt_states.sum(axis=0)

            # Compensate for haploid X
            if gt_states.shape[0] == 1:
                gt_states_joined *= 2

            # Get the inferred states
            inferred_states = inferred[chrom_idx][org_idx]
            num_not_matching = np.sum(inferred_states != gt_states_joined)
            chrom_errors.append(num_not_matching)

        # Report error as a fraction of sites
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
    final_gen = experiment.run_experiment(params['num_generations'],
                                            n_proc=params['num_threads'])

    # Sample cells from the final generation
    final_gen.organism_list = final_gen.organism_list[:params['offspring_per_generation']]
    adata_sample, organism_list = final_gen.sample_cells(params['avg_UMI'],
                                             params['variance_UMI'],
                                             params['num_cells_per_org'],
                                             adata,
                                             n_processes=params['num_threads'],
                                             doublet_rate=params['doublet_rate'])

    print('Done cell sampling')
    print(datetime.now())

    return adata_sample, organism_list

def get_transitions(num_generations):
    """
    Loads a pre-computed T matrix for a simulated dataset.
    Args:
        num_generations: The number of generations simulated

    Returns:T

    """
    transitions = pickle.load(open('data_dir/transitions/transitions_{' + str(num_generations) + '}_gen.pickle', 'rb'))
    return transitions

def count_snps_in_genes(adata, valid_chromosomes, gene_to_snp_maps):
    """
    Debug helper that counts the number of snps seen in expressed genes.
    :return: None
    """
    # Count SNPs for genes that have >0 expression in the adata used for sampling
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

def print_stats_on_sim_data(A, D, valid_chromosomes, organisms):
    """
    Given a simulated dataset, prints statistics about it
    :return:
    """
    for i in range(len(A)):
        print('A shape: ', A[i].shape)

    # A and D are lists per chromosome, with A[i] shape (n_cells, n_snps_chr)
    # valid_chromosomes order is the same as A/D
    chr_index = valid_chromosomes.index('X')
    A_X = A[chr_index]   # shape (n_cells, n_snps)
    D_X = D[chr_index]

    alt_per_cell = np.asarray(A_X.sum(axis=1)).ravel()
    depth_per_cell = np.asarray(D_X.sum(axis=1)).ravel()
    alt_per_site = np.asarray(A_X.sum(axis=0)).ravel()

    # fraction of cells with zero alt reads on X
    zero_alt_per_cell = (alt_per_cell == 0)
    print("Cells with zero alt on X:", zero_alt_per_cell.sum(), "/", A_X.shape[0],
          f"({100*zero_alt_per_cell.mean():.2f}%)")

    # distribution of total coverage on X per cell
    print("X coverage per cell: mean, median, 10th, 90th:",
          np.mean(depth_per_cell), np.median(depth_per_cell),
          np.percentile(depth_per_cell, 10), np.percentile(depth_per_cell, 90))

    print("Sites with zero alt reads (never observed):", (alt_per_site == 0).sum(), "/", len(alt_per_site))
    print("Top 10 alt counts per site:", np.sort(alt_per_site)[-10:])

    # final_gen organism_list contains organisms; each org.chromosomes['X'].states is 1xN or 2xN
    for i, org in enumerate(organisms[:20]):
        states = org['chromosomes']['X']
        print("org", i, "gender", org['gender'], "states shape", states.shape, "mean alt per chromatid:",
              [s.mean() for s in states])

def organism_to_dict(organism):
    """
    Given an organism as defined in data_simulator.py, convert to a dict containing
    only the key properties.

    :param organism:
    :return: organism_dict
    """
    organism_dict = {'chr_names': organism.chr_names,
                     'gender': organism.gender,
                     'chromosomes': {chrom_name: organism.chromosomes[chrom_name].states
                                     for chrom_name in organism.chromosomes}}
    return organism_dict

def shuffle_snps_to_Z(adata, shuffle_percent=1.0,
                      chromosome_key="chromosome", seed=None):
    """
    Move a fraction of SNP data onto a synthetic chromosome Z and
    randomly shuffle the data among those SNPs.

    The selected SNPs have their chromosome annotation changed to "Z".
    Their X and layers["depth"] columns are then jointly permuted.
    All other SNP data and metadata remain unchanged.
    :param adata: AnnData object containing:
                    - X: alternate allele counts
                    - layers["depth"]: depth counts
                    - var[chromosome_key]: chromosome labels
    :param shuffle_percent: Fraction of all SNPs to move to Z, in [0, 1].
    :param chromosome_key: adata.var column containing chromosome labels.
    :param seed: Optional random seed.
    :return: adata_shuffled: Modified copy of adata.
    """
    if not 0 <= shuffle_percent <= 1:
        raise ValueError("shuffle_percent must be in [0, 1]")

    adata_shuffled = adata.copy()

    n_snps = adata.n_vars
    n_shuffle = int(round(shuffle_percent * n_snps))

    if n_shuffle < 2:
        return adata_shuffled

    rng = np.random.default_rng(seed)

    # Select SNPs globally.
    selected = rng.choice(n_snps, size=n_shuffle, replace=False)

    # Add Z to categories if chromosome column is categorical.
    chrom = adata_shuffled.var[chromosome_key]
    if hasattr(chrom.dtype, "categories"):
        if "Z" not in chrom.cat.categories:
            adata_shuffled.var[chromosome_key] = chrom.cat.add_categories(["Z"])

    # Change their chromosome annotation to Z.
    adata_shuffled.var.loc[adata_shuffled.var.index[selected], chromosome_key] = "Z"

    # Randomly permute the selected SNP data.
    permutation = selected.copy()
    rng.shuffle(permutation)

    # Shuffle X.
    X = adata_shuffled.X.copy()
    X[:, selected] = X[:, permutation]
    adata_shuffled.X = X

    # Shuffle depth using exactly the same permutation.
    depth = adata_shuffled.layers["depth"].copy()
    depth[:, selected] = depth[:, permutation]
    adata_shuffled.layers["depth"] = depth

    return adata_shuffled

def vireo_genotype_mat_to_hmm(arr, A_ref):
    """
    Converts a vireo np.array to HMM format. Indices in the vireo array are assumed
    to be ordered the same as A_ref. Just concatenated. Specifically for use for genotype
    arrays

    :param arr: The vireo np.array to be converted
    :param A_ref: A reference array in HMM format
    :return: hmm_arr
    """

    converted_arr = []

    # We need to align the indices
    cur_idx = 0

    for chrom_arr in A_ref:
        num_snps = chrom_arr.shape[1]
        converted_arr.append(arr[:, cur_idx:cur_idx + num_snps])
        cur_idx += num_snps

    return converted_arr