import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode

def prepare_sparse_counts(A, D):
    """
    Convert sparse allele-count matrices into a GPU-friendly,
    cell-major compressed representation.

    Parameters
    ----------
    A : scipy sparse matrix, shape (num_cells, num_variants)
        Alternate allele counts.
    D : scipy sparse matrix, shape (num_cells, num_variants)
        Total depths.

    Returns
    -------
    positions : uint32 array
        Global variant index for each observed cell-variant pair.
    A_values : uint16 array
        Alternate allele counts.
    D_values : uint16 array
        Total depths.
    offsets : uint64 array
        offsets[i]:offsets[i+1] gives observations for cell i.
    """

    # Convert to CSR for efficient row/cell access
    A_csr = sparse.csr_matrix(A)
    D_csr = sparse.csr_matrix(D)

    # D defines which SNPs are observed.
    D_csr.eliminate_zeros()
    D_csr.sort_indices()

    # Get A values at exactly the positions where D > 0.
    rows = np.repeat(np.arange(D_csr.shape[0]), np.diff(D_csr.indptr))

    positions = D_csr.indices.astype(np.uint32, copy=False)
    D_values = D_csr.data.astype(np.uint16, copy=False)

    # A may contain implicit zeros, which are biologically meaningful:
    # observed REF-only SNPs have A=0, D>0.
    A_values = np.asarray(A_csr[rows, positions]).ravel().astype(np.uint16)

    offsets = D_csr.indptr.astype(np.uint64, copy=False)

    return positions, A_values, D_values, offsets, A_csr, D_csr

def adata_to_HMM(adata, valid_chromosomes, chromosome_key='chromosome'):
    """
    Convert an anndata into the DemuxHMM format of a list of matrices by chromosome.

    :param adata: An anndata object where X contains alt counts, and layers['depth'] is depth.
    :param valid_chromosomes: The order of chromosomes. Should match adata.
    :param chromosome_key: The var key under which chromosome identities for snps are stored.
    :return: A (list of csr matrices for alt counts, by chromosome),
            D (list of csr matrices for depth, by chromosome),
            ground_truth (ground truth identity for each cell)
    """
    A = []
    D = []

    for chrom in valid_chromosomes:
        mask = (adata.var[chromosome_key] == chrom).to_numpy()

        A.append(sparse.csr_matrix(adata.X[:, mask]))
        D.append(sparse.csr_matrix(adata.layers["depth"][:, mask]))

    if 'identity' in adata.obs.columns:
        ground_truth = adata.obs["identity"].to_numpy(dtype=np.int32)
        return A, D, ground_truth
    else:
        return A, D

def filter_doublets(A, D, ground_truth, doublet_mask):
    """
    Remove predicted doublets from all chromosome matrices.
    """
    keep_idx = np.flatnonzero(~doublet_mask)

    A_filtered = [A_chr[keep_idx] for A_chr in A]
    D_filtered = [D_chr[keep_idx] for D_chr in D]
    ground_truth_filtered = ground_truth[keep_idx]

    return A_filtered, D_filtered, ground_truth_filtered, keep_idx

def consensus_clustering_jaccard(all_assignments, reference):
    """
    Align repeated clustering attempts to a reference clustering using maximum total Jaccard similarity,
    then compute a Jaccard-weighted consensus assignment.

    :param all_assignments: np.ndarray  Shape (n_cells, n_attempts).
    :param reference: np.ndarray Shape (n_cells,). The best clustering run.
    :return consensus: np.ndarray Consensus cluster assignment for each cell.
    :return aligned_assignments: np.ndarray Cluster assignments aligned to the reference.
    """
    all_assignments = np.asarray(all_assignments)
    reference = np.asarray(reference)

    if all_assignments.ndim != 2:
        raise ValueError("all_assignments must have shape (cells, attempts)")
    if reference.ndim != 1:
        raise ValueError("reference must have shape (cells,)")
    if all_assignments.shape[0] != len(reference):
        raise ValueError("all_assignments and reference must contain the same cells")

    ref_clusters = np.unique(reference)
    aligned_assignments = np.empty_like(all_assignments)

    for j in range(all_assignments.shape[1]):
        labels = all_assignments[:, j]
        clusters = np.unique(labels)

        # Compute Jaccard similarity
        jaccard = np.zeros((len(ref_clusters), len(clusters)), dtype=float)

        for i, ref_cluster in enumerate(ref_clusters):
            ref_mask = reference == ref_cluster

            for k, cluster in enumerate(clusters):
                cluster_mask = labels == cluster

                intersection = np.sum(ref_mask & cluster_mask)
                union = np.sum(ref_mask | cluster_mask)

                if union > 0:
                    jaccard[i, k] = intersection / union

        # Maximum-weight one-to-one matching
        row_ind, col_ind = linear_sum_assignment(-jaccard)
        mapping = {int(clusters[col]): int(ref_clusters[row]) for row, col in zip(row_ind, col_ind)}

        # Record the Jaccard weight associated with each mapping
        cluster_weights = {int(clusters[col]): jaccard[row, col] for row, col in zip(row_ind, col_ind)}

        # Handle unmatched clusters
        matched_clusters = set(mapping.keys())

        for k, cluster in enumerate(clusters):
            cluster = int(cluster)

            if cluster not in matched_clusters:
                best_reference = np.argmax(jaccard[:, k])
                mapping[cluster] = int(ref_clusters[best_reference])
                cluster_weights[cluster] = jaccard[best_reference, k]

        # Align this attempt
        aligned_assignments[:, j] = np.array([mapping[int(x)] for x in labels], dtype=all_assignments.dtype)

    # Select the reference cluster receiving the largest total weight
    consensus = mode(aligned_assignments, axis=1, keepdims=False).mode

    return consensus, aligned_assignments