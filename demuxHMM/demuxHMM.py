from numba import njit, cuda
from math import lgamma

from scipy.special import gammaln
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math
from . import demuxHMM_utils as utils

# Gpu math constant for doublet detection
MAX_K = 250
MAX_CLUSTER_PAIRS = (MAX_K * (MAX_K - 1)) // 2

@njit
def xlogy(x, y):
    """
    Computes xlog(y) so that it is 0 if x is 0.
    Args:
        x: float >= 0
        y: float >= 0
    Returns: xlog(y)
    """
    if x == 0:
        return 0

    return x*np.log(y)

@njit
def xlog1py(x, y):
    """
    Calculates xlog1p(y) so that it is 0 if x is 0.
    Args:
        x: float >= 0
        y: float >= 0
    Returns: xlog(1 + y)
    """
    if x == 0:
        return 0

    return x*np.log1p(y)

@njit
def binom_logpmf(k=1, n=1, p=0.5):
    """
    Overall log pmf function for the binomial distribution.
    Args:
        k: Number of successes.
        n: Total number of draws.
        p: Probability of success.

    Returns: The binomial log-pmf.
    """
    combiln = lgamma(n + 1) - (lgamma(k + 1) + lgamma(n - k + 1))
    r = combiln + xlogy(k, p) + xlog1py(n - k, -p)
    return r

@njit
def make_lgamma_lookup(max_depth):
    # Creates a lookup table for lgamma up to max_depth + 1
    table = np.empty((max_depth + 2), dtype=np.float32)

    for i in range(1, max_depth + 2):
        table[i] = lgamma(i)

    return table

@cuda.jit(device=True)
def _lse3(a0, a1, a2):
    """
    Stabilized log-sum-exp for three floats.
    Args:
        a0: float
        a1: float
        a2: float
    Returns: LSE([a0, a1, a2])
    """
    m = max(a0, a1) if max(a0, a1) > a2 else a2
    return math.log(math.exp(a0 - m) + math.exp(a1 - m) + math.exp(a2 - m)) + m

@cuda.jit(device=True)
def _lse(lse_terms, num_pairs):
    m = lse_terms[0]
    for i in range(1, num_pairs):
        if lse_terms[i] > m:
            m = lse_terms[i]

    s = 0.0
    for i in range(num_pairs):
        s += math.exp(lse_terms[i] - m)

    return  m + math.log(s)

@njit
def betaln(a, b):
    return np.log(math.gamma(a)) + np.log(math.gamma(b)) - np.log(math.gamma(a + b))

@njit
def beta_logpdf(x, a, b):
    lPx = xlog1py(b - 1.0, -x) + xlogy(a - 1.0, x)
    lPx -= betaln(a, b)
    return lPx

@njit
def _logsumexp(a):
    a_max = a.max()
    b = a[a != a_max] - a_max
    m = len(a) - len(b)
    return np.log1p(np.exp(b).sum()/m) + np.log(m) + a_max

def stable_softmax(log_probs):
    a_max = np.max(log_probs)
    exp_vals = np.exp(log_probs - a_max)
    sum_exp = np.sum(exp_vals)

    if sum_exp == 0 or np.isnan(sum_exp):
        # fallback to uniform
        return np.ones_like(log_probs) / len(log_probs)
    return exp_vals / sum_exp

@cuda.jit(device=True)
def xlogy_gpu(x, y):
    """
    Computes xlog(y) so that it is 0 if x is 0. This is the gpu code version.
    Args:
        x: float >= 0
        y: float >= 0
    Returns: xlog(y)
    """
    return 0.0 if x == 0 else x * math.log(y)

@cuda.jit(device=True)
def xlog1py_gpu(x, y):
    """
    Calculates xlog1p(y) so that it is 0 if x is 0. This is the gpu code version.
    Args:
        x: float >= 0
        y: float >= 0
    Returns: xlog(1 + y)
    """
    return 0.0 if x == 0 else x * math.log1p(y)

@cuda.jit(device=True)
def binom_logpmf_gpu(k, n, log_p, log_1p, lgamma_lookup):
    """
    The log pmf function for the binomial distribution. Compatible with gpu.
    Args:
        k: Number of successes.
        n: Total number of draws.
        log_p: log probability of success.
        log_1p: log 1 - probability of success.
        lgamma_lookup: A lookup table for the lgamma function

    Returns: The binomial log-pmf.
    """
    log_pmf = lgamma_lookup[n + 1] - lgamma_lookup[k + 1] - lgamma_lookup[n - k + 1]
    log_pmf += k * log_p
    log_pmf += (n - k) * log_1p

    return log_pmf

@cuda.jit
def z_probs_kernel(positions, A_values, D_values, cell_offsets, V,
                   log_theta, log1_theta, log_kappas, num_chr,
                   log_probs, lgamma_lookup):
    """
    Each thread computes the log-probability for one (cell, cluster) pair.

    positions[p]: Variant index for observation p
    A_values[p]: Alternate allele count
    D_values[p]: Total depth
    cell_offsets[c, i]: cell_offsets[c, i+1]: absolute observation range for cell i on chromosome c
    V: [num_chr, num_clusters, max_len]
    """
    cell_idx, cluster_idx = cuda.grid(2)
    num_cells, num_clusters = log_probs.shape

    if cell_idx >= num_cells or cluster_idx >= num_clusters:
        return

    log_prob = 0.0

    for c in range(num_chr):
        start = cell_offsets[c, cell_idx]
        end = cell_offsets[c, cell_idx + 1]

        for p in range(start, end):
            j = positions[p]
            a = A_values[p]
            d = D_values[p]
            genotype = V[c, cluster_idx, j]

            log_prob += binom_logpmf_gpu(a, d, log_theta[genotype], log1_theta[genotype], lgamma_lookup)

    log_prob += log_kappas[cluster_idx]
    log_probs[cell_idx, cluster_idx] = log_prob

def get_z_probs_gpu(num_cells, num_clusters, num_chr, chain_lengths,
                    positions, A_values, D_values, cell_offsets,
                    V, thetas, kappas, lgamma_lookup):
    # Pad chains so we can transfer to GPU
    max_len = max(chain_lengths)
    V_pad = np.zeros((num_chr, num_clusters, max_len), dtype=np.uint8)

    for c in range(num_chr):
        V_pad[c, :, :chain_lengths[c]] = V[c]

    # Precompute log kappas since we use it for each z_i
    log_kappas = np.log(kappas, dtype=np.float32)

    # Clamp thetas to a safe range (in case of numerical errors)
    thetas_clipped = np.clip(thetas, a_min=1e-6, a_max=1 - 1e-6)

    # Pre-compute common logs of theta
    log_theta = np.log(thetas_clipped, dtype=np.float32)
    log1_theta = np.log(1.0 - thetas_clipped, dtype=np.float32)

    # Transfer sparse observations
    positions_d = cuda.to_device(positions.astype(np.uint32, copy=False))
    A_values_d = cuda.to_device(A_values.astype(np.uint16, copy=False))
    D_values_d = cuda.to_device(D_values.astype(np.uint16, copy=False))
    cell_offsets_d = cuda.to_device(cell_offsets.astype(np.uint64, copy=False))

    # Transfer model data
    V_d = cuda.to_device(V_pad)
    log_theta_d = cuda.to_device(log_theta)
    log1_theta_d = cuda.to_device(log1_theta)
    log_kappas_d = cuda.to_device(log_kappas)
    lgamma_lookup_d = cuda.to_device(lgamma_lookup.astype(np.float32, copy=False))

    log_probs_d = cuda.device_array((num_cells, num_clusters), dtype=np.float32)

    threadsperblock = (16, 16)  # 256 threads/block
    blockspergrid = ((num_cells + threadsperblock[0] - 1) // threadsperblock[0],
                     (num_clusters + threadsperblock[1] - 1) // threadsperblock[1])

    z_probs_kernel[blockspergrid, threadsperblock](positions_d, A_values_d, D_values_d, cell_offsets_d, V_d,
                                                   log_theta_d, log1_theta_d, log_kappas_d, num_chr,
                                                   log_probs_d, lgamma_lookup_d)

    # Copy back and normalize on CPU
    log_probs = log_probs_d.copy_to_host()
    probs = np.empty_like(log_probs)

    for i in range(num_cells):
        probs[i] = stable_softmax(log_probs[i])

    return probs

def precompute_cluster_log_likelihoods_chr(c, ZT, A_sparse_c, D_sparse_c,
                                           logcomb_sparse_c, chain_lengths, log_theta,
                                           log1_theta, num_states, num_clusters):
    """
    Precompute sum(log_binom_pmf) for all (cluster, pos, state) in a chromosome.

    :param c: The current chromosome.
    :param ZT: (num_cells, num_clusters) where Z_onehot[i][k] == 1 if the cluster of cell i is k
    :param A_sparse_c: A csr matrix of alternate allele counts for chromosome c.
    :param D_sparse_c: A csr matrix of depth counts for chromosome c.
    :param logcomb_sparse_c: Precomputed values for D[c][i][j] choose A[c][i][j]. Stored as csr.
    :param chain_lengths: A list containing number of SNPs by chromosome.
    :param log_theta: log(emission probabilities)
    :param log1_theta: log( 1 - emission probabilities)
    :param num_states: Number of states (always 3).
    :param num_clusters: Number of clusters.
    :return: c, log_likelihoods_c
    """
    L = chain_lengths[c]
    log_likelihoods_c = np.empty((num_clusters, L, num_states), dtype=np.float32)

    # Sum of all observations over cells. Shape (num_clusters, L)
    sum_A = (ZT @ A_sparse_c).toarray()
    sum_D = (ZT @ D_sparse_c).toarray()
    sum_logcomb = (ZT @ logcomb_sparse_c).toarray()

    for t in range(num_states):
        log_likelihoods_c[:, :, t] = sum_logcomb + sum_A * log_theta[t] + (sum_D - sum_A) * log1_theta[t]

    return c, log_likelihoods_c

def precompute_cluster_log_likelihoods(A_sparse, D_sparse, Z,
                                       logcomb_sparse,thetas,
                                       num_chr, num_clusters,
                                       chain_lengths, num_states=3, threads=8):
    """
    Precomputes log(binompdf) for each cluster.

    :param A_sparse: A list of csr matrices of alternate allele counts by chromosome.
    :param D_sparse: A list of csr matrices of depth counts by chromosome.
    :param Z: Cluster assignments (Z[i] = k means cell i is in cluster k).
    :param logcomb_sparse: Precomputed values for D[c][i][j] choose A[c][i][j]. Stored as csr.
    :param thetas: The current emission probabilities.
    :param num_chr: Number of chromosomes.
    :param num_clusters: The number of clusters.
    :param chain_lengths: The number of snps on each chromosome (indexed by c).
    :param num_states: The number of states (always 3).
    :param threads: The number of threads to use.
    :return: log_likelihoods
    """
    max_len = np.max(chain_lengths)
    log_likelihoods = np.zeros((num_chr, num_clusters, max_len, num_states), dtype=np.float32)

    # Make a one-hot matrix identifying the clustering
    # (num_cells, num_clusters) where Z_onehot[i][k] == 1 if the cluster of cell i is k
    num_cells = len(Z)
    Z_onehot = csr_matrix((np.ones(num_cells, dtype=np.float32), (np.arange(num_cells), Z)),
                          shape=(num_cells, num_clusters))
    ZT = Z_onehot.T

    # We will use log thetas throughout the log binompmf, so pre-compute once
    log_theta = np.log(thetas)
    log1_theta = np.log1p(-thetas)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(precompute_cluster_log_likelihoods_chr,c, ZT, A_sparse[c], D_sparse[c],
                            logcomb_sparse[c], chain_lengths, log_theta, log1_theta,
                            num_states, num_clusters) for c in range(num_chr)]

        for future in as_completed(futures):
            c, result = future.result()
            L = chain_lengths[c]
            log_likelihoods[c, :, :L, :] = result

    return log_likelihoods

@cuda.jit
def forward_backward_likelihood_kernel(log_likelihoods, beta_ln_prior, log_T, log_pi, log_kappas,
                                       chain_lengths, log_alphas, log_betas,
                                         num_chr, num_clusters):

    c_idx, k_idx = cuda.grid(2)
    if c_idx >= num_chr or k_idx >= num_clusters:
        return

    # Basic constants
    L = chain_lengths[c_idx]
    NUM_STATES = 3

    vals = cuda.local.array(3, dtype=np.float32)

    # Add priors
    for t in range(NUM_STATES):
        log_alphas[c_idx, k_idx, 0, t] = (log_likelihoods[c_idx, k_idx, 0, t] +
                                          log_kappas[k_idx] + log_pi[t] + beta_ln_prior[t])

    # Forward pass
    for l in range(1, L):
        for t in range(NUM_STATES):
            for t_prev in range(NUM_STATES):
                vals[t_prev] = log_alphas[c_idx, k_idx, l - 1, t_prev] + log_T[c_idx, t_prev, t]
            max_val = max(vals[0], max(vals[1], vals[2]))
            lse = math.log(math.exp(vals[0] - max_val) +
                           math.exp(vals[1] - max_val) +
                           math.exp(vals[2] - max_val)) + max_val
            log_alphas[c_idx, k_idx, l, t] = log_likelihoods[c_idx, k_idx, l, t] + lse

    # Backward pass
    for t in range(NUM_STATES):
        # Re-initialize
        log_betas[c_idx, k_idx, L - 1, t] = 0.0

    for l in range(L - 2, -1, -1):
        for t in range(NUM_STATES):
            for t_next in range(NUM_STATES):
                vals[t_next] = (
                    log_likelihoods[c_idx, k_idx, l + 1, t_next] +
                    log_betas[c_idx, k_idx, l + 1, t_next] +
                    log_T[c_idx, t, t_next]
                )
            max_val = max(vals[0], max(vals[1], vals[2]))
            lse = math.log(math.exp(vals[0] - max_val) +
                           math.exp(vals[1] - max_val) +
                           math.exp(vals[2] - max_val)) + max_val
            log_betas[c_idx, k_idx, l, t] = lse

@njit
def compute_betaln_prior(thetas, alphas, betas):
    """
    Computes the betaln pdf function.

    :param thetas: Emission probabilities.
    :param alphas: The alpha priors for each emission probability.
    :param betas: The beta priors for each emission probability.
    :return: betaln_pdf (len(3) np.array)
    """
    prior = np.zeros_like(thetas)

    for t in range(thetas.shape[0]):
        prior[t] += (alphas[t] - 1) * math.log(thetas[t]) + (betas[t] - 1) * math.log1p(-thetas[t])
        prior[t] -= lgamma(alphas[t]) + lgamma(betas[t]) - lgamma(alphas[t] + betas[t])

    return prior

def forward_backward_gpu_optimized(log_likelihoods, thetas, T, pi, kappas, alphas, betas,
                                   chain_lengths, num_chr, num_clusters, max_len):
    num_states = 3

    # Precompute commonly used logs
    log_T = np.log(T, dtype=np.float32)
    log_pi = np.log(pi, dtype=np.float32)
    log_kappas = np.log(kappas, dtype=np.float32)
    beta_ln_prior = compute_betaln_prior(thetas, alphas, betas)

    # Move everything to device
    log_T_d = cuda.to_device(log_T)
    log_pi_d = cuda.to_device(log_pi)
    log_kappas_d = cuda.to_device(log_kappas)
    lens_d = cuda.to_device(np.array(chain_lengths, dtype=np.int32))
    beta_ln_prior_d = cuda.to_device(beta_ln_prior)

    log_likelihoods_d = cuda.to_device(log_likelihoods)
    log_alphas_d = cuda.device_array((num_chr, num_clusters, max_len, num_states), dtype=np.float32)
    log_betas_d = cuda.device_array_like(log_alphas_d)

    threadsperblock = (16, 16)
    blockspergrid = (
        (num_chr + threadsperblock[0] - 1) // threadsperblock[0],
        (num_clusters + threadsperblock[1] - 1) // threadsperblock[1]
    )

    forward_backward_likelihood_kernel[blockspergrid, threadsperblock](
        log_likelihoods_d, beta_ln_prior_d, log_T_d, log_pi_d, log_kappas_d,
        lens_d, log_alphas_d, log_betas_d,
        num_chr, num_clusters)

    return log_alphas_d.copy_to_host(), log_betas_d.copy_to_host()

@cuda.jit
def chain_probs_kernel(log_alphas, log_betas, chain_lens, out_probs,
                       num_chr, num_clusters, max_len):
    """
    Computes probs for each (chr=c, cluster=k, pos=l, state=t):
      p_t ∝ exp(log_alpha + log_beta - log p(data))
    Normalizes across t=0..2 for each (c,k,l).
    Shapes:
      log_alphas, log_betas: [num_chr, num_clusters, max_len, 3]
      chain_lens: [num_chr]
      out_probs:  [num_chr, num_clusters, max_len, 3]
    """
    c, k, l = cuda.grid(3)
    if c >= num_chr or k >= num_clusters:
        return
    L = chain_lens[c]
    if l >= L:
        return

    # log p(data) = logsumexp(log_alpha at final state position)
    la0 = log_alphas[c, k, L - 1, 0]
    la1 = log_alphas[c, k, L - 1, 1]
    la2 = log_alphas[c, k, L - 1, 2]
    log_joint = _lse3(la0, la1, la2)

    # Unnormalized posteriors in linear space
    s0 = math.exp(log_alphas[c, k, l, 0] + log_betas[c, k, l, 0] - log_joint)
    s1 = math.exp(log_alphas[c, k, l, 1] + log_betas[c, k, l, 1] - log_joint)
    s2 = math.exp(log_alphas[c, k, l, 2] + log_betas[c, k, l, 2] - log_joint)

    s = s0 + s1 + s2
    # Numerical guard + normalization
    if s <= 0.0 or math.isnan(s) or math.isinf(s):
        out_probs[c, k, l, 0] = 1.0 / 3.0
        out_probs[c, k, l, 1] = 1.0 / 3.0
        out_probs[c, k, l, 2] = 1.0 / 3.0
    else:
        invs = 1.0 / s
        out_probs[c, k, l, 0] = s0 * invs
        out_probs[c, k, l, 1] = s1 * invs
        out_probs[c, k, l, 2] = s2 * invs

def pack_logs_to_padded(num_chr, K, chain_lengths,
                         log_alphas_V, log_betas_V):
    """
    Pads values of log-alpha and log-betas into GPU-friendly arrays
    Args:
        num_chr: The number of chromosomes
        K: The number of clusters
        chain_lengths: An array of lengths of the chains
        log_alphas_V: log_alphas as produced by the forward step
        log_betas_V: log_betas as produced by the backward step
    Returns: la, lb (padded versions of log_alphas and log_betas)
    """
    S = 3 # The number of states
    max_len = max(chain_lengths)

    la_pad = np.zeros((num_chr, K, max_len, S), dtype=np.float32)
    lb_pad = np.zeros_like(la_pad)

    for c in range(num_chr):
        L = chain_lengths[c]

        for k in range(K):
            # each is (L,3)
            la = log_alphas_V[c][k].astype(np.float32)
            lb = log_betas_V[c][k].astype(np.float32)

            la_pad[c, k, :L, :] = la[:L, :]
            lb_pad[c, k, :L, :] = lb[:L, :]

    return la_pad, lb_pad, max_len

def chain_probs_gpu(num_chr, num_clusters,
                    chain_lengths, log_alphas_V, log_betas_V):
    """
    Prepares data for the GPU kernel to get probabilities for each possible state in the chain.
    Args:
        num_chr: The number of chromosomes
        num_clusters: The number of clusters
        chain_lengths: An array of lengths of the chains
        log_alphas_V: log_alphas as produced by the forward step
        log_betas_V: log_betas as produced by the backward step
    Returns: The probabilities for each state in each chain.

    """
    # Pack lists into padded tensors for GPU
    la_pad, lb_pad, max_len = pack_logs_to_padded(num_chr, num_clusters,
                                                  chain_lengths, log_alphas_V, log_betas_V)

    # Move to device
    la_d = cuda.to_device(la_pad)
    lb_d = cuda.to_device(lb_pad)
    lens_d = cuda.to_device(np.array(chain_lengths, dtype=np.int32))
    out_d = cuda.device_array_like(la_d)  # same shape

    # Launch config: 3D grid over (c,k,l)
    threadsperblock = (4, 8, 16)  # tweak as needed
    blockspergrid = (
        (num_chr + threadsperblock[0] - 1) // threadsperblock[0],
        (num_clusters + threadsperblock[1] - 1) // threadsperblock[1],
        (max_len + threadsperblock[2] - 1) // threadsperblock[2],
    )

    chain_probs_kernel[blockspergrid, threadsperblock](
        la_d, lb_d, lens_d, out_d,
        num_chr, num_clusters, max_len
    )

    out = out_d.copy_to_host()
    return out

def normalize_rows(mat):
    """
    Normalizes a matrix containing log probabilities (unnormalized) and bring it into normal space.
    Args:
        mat: A matrix of unnomormalized log probabilities.

    Returns: normed_mat
    """
    normed_mat = mat.copy()

    for i in range(mat.shape[0]):
        # Normalize probabilities in log space (log sum exp trick)
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        lse_row = _logsumexp(mat[i]) # ! If there is instability, look here first!!!

        probs_row = np.zeros(mat.shape[1])
        for k in range(mat.shape[1]):
            probs_row[k] = np.exp(mat[i, k] - lse_row)

        normed_mat[i] = probs_row

    return normed_mat

def get_theta_chr(z_probs, A_c, D_c, probs_V_c):
    """
    A worker function to compute values of theta on a chromosome-by-chromosome basis.
    :param z_probs: Cluster probabilities
    :param A_c: Alternate allele counts for chromosome c (csr matrix)
    :param D_c: Depth counts for chromosome c (csr matrix)
    :param probs_V_c: The chain probabilities for chromosome c
    :return:
    """
    # (clusters, cells) x (cells, snps) equiv to weighted_A[k][j] = sum_i z[i][k]*A[c][i][j]
    weighted_A = (A_c.T @ z_probs).T
    weighted_D = (D_c.T @ z_probs).T

    # For numerators equiv to sum_k sum_j (sum_i z[i][k]*A[c][i][j]) * probs_v[c][k][j]
    numerators = np.einsum('kj,kjt->t', weighted_A, probs_V_c)
    denominators = np.einsum('kj,kjt->t', weighted_D, probs_V_c)

    return numerators, denominators

@cuda.jit
def doublet_probs_kernel(positions, A_values, D_values,
                         cell_offsets, V, top_clusters, pair_log_weights,
                         log_thetas, log1_thetas, num_chr,
                         log_probs, lgamma_lookup, k):
    """
    Each thread computes the binomial data log-probability for one cell assuming doublet status.
    We cycle though the top k candidates for the doublet

    :param positions: Variant index for observation p.
    :param A_values: Alternate allele count.
    :param D_values: Total depth.
    :param cell_offsets: cell_offsets[c, i+1]: absolute observation range for cell i on chromosome c.
    :param V: [num_chr, num_clusters, max_len].
    :param top_clusters: The top k cluster indices for each cell.
    :param pair_log_weights: log of cluster probabilities for each cell.
    :param log_thetas: log(thetas_grid). thetas_grid is a lookup for paired thetas.
    :param log1_thetas: log(1 - thetas_grid).
    :param num_chr: Number of chromosomes.
    :param log_probs: A cell-wise array to store binomial log-prob output.
    :param lgamma_lookup: A lookup table for the lgamma function.
    :param k: How many clusters to consider.
    """
    cell_idx = cuda.grid(1)
    num_cells = log_probs.shape[0]
    if cell_idx >= num_cells:
        return

    # Compute the top clusters and their weights
    num_cluster_pairs = (k * (k - 1)) // 2 # k choose 2
    lse_terms = cuda.local.array(MAX_CLUSTER_PAIRS, np.float32)
    i = 0

    for k1 in range(k):
        for k2 in range(k1):
            cluster1 = top_clusters[cell_idx, k1]
            cluster2 = top_clusters[cell_idx, k2]

            # Compute the log prob for one cluster pair
            log_prob = np.float32(0.0)

            for c in range(num_chr):
                start = cell_offsets[c, cell_idx]
                end = cell_offsets[c, cell_idx + 1]

                for p in range(start, end):
                    j = positions[p]
                    a = A_values[p]
                    d = D_values[p]
                    genotype1 = V[c, cluster1, j]
                    genotype2 = V[c, cluster2, j]

                    log_prob += binom_logpmf_gpu(a, d, log_thetas[genotype1, genotype2],
                                                 log1_thetas[genotype1, genotype2], lgamma_lookup)

            # Add prior
            log_prob += pair_log_weights[cell_idx, i]

            # We will have to sum over all the pairings
            lse_terms[i] = log_prob
            i += 1

    log_probs[cell_idx] = _lse(lse_terms, num_cluster_pairs)

def _get_data_prob_doublet(z_probs, positions_flat, A_values_flat, D_values_flat,
                           cell_offsets_flat, V, thetas, lgamma_lookup, k):
    """
    Computes the data likelihood for each cell given the current model and clustering.
    Assumes the cell is a singlet.

    :param z_probs: The probability of membership for each cell x cluster.
    :param positions_flat: A flattened array of indices corresponding to SNPs.
    :param A_values_flat: A flattened array of alt reads.
    :param D_values_flat:  A flattened array of depths.
    :param cell_offsets_flat: Cell offsets indexed by (chromosome, cell_idx)
    :param V: The fitted chains.
    :param thetas: Current theta values.
    :param lgamma_lookup: A lookup table for the lgamma function.
    :param k: The number of top cluster pairings we want to explore.
    :return data_prob_singlet:
    """
    # Define constants
    num_chr = len(V) # Number of chromosomes
    num_clusters = V[0].shape[0] # Number of clusters
    chain_lengths = [V[c].shape[1] for c in range(num_chr)]
    max_len = max(chain_lengths)
    num_cells = z_probs.shape[0]

    # Pad chains so we can transfer to GPU
    V_pad = np.zeros((num_chr, num_clusters, max_len), dtype=np.uint8)

    for c in range(num_chr):
        V_pad[c, :, :chain_lengths[c]] = V[c]

    # Clamp thetas to a safe range (in case of numerical errors)
    thetas_clipped = np.clip(thetas, a_min=1e-6, a_max=1 - 1e-6)
    num_states = len(thetas_clipped)

    # Pre-compute common logs of theta pairings
    theta_pairs_lookup = np.empty(shape=(num_states, num_states), dtype=np.float32)

    for s1 in range(num_states):
        for s2 in range(num_states):
            theta_pairs_lookup[s1, s2] = (thetas_clipped[s1] + thetas_clipped[s2]) / 2

    log_theta = np.log(theta_pairs_lookup, dtype=np.float32)
    log1_theta = np.log(1.0 - theta_pairs_lookup, dtype=np.float32)

    # Compute top-k clusters for every cell
    top_clusters = np.argsort(z_probs, axis=1)[:, -k:].astype(np.uint16)
    top_probs = np.empty((num_cells, k), dtype=np.float32)

    for i in range(num_cells):
        for j in range(k):
            cluster = top_clusters[i, j]
            top_probs[i, j] = z_probs[i, cluster]

    # Compute a normalized cluster weighting
    num_pairs = k * (k - 1) // 2
    pair_log_weights = np.empty((num_cells, num_pairs), dtype=np.float32)

    for i in range(num_cells):
        pair_weights = np.empty(num_pairs, dtype=np.float32)

        p = 0
        for k1 in range(k):
            for k2 in range(k1):
                pair_weights[p] = top_probs[i, k1] * top_probs[i, k2]
                p += 1

        pair_weights /= pair_weights.sum()
        pair_log_weights[i] = np.log(np.maximum(pair_weights, 1e-30))

    # Transfer sparse observations
    positions_d = cuda.to_device(positions_flat.astype(np.uint32, copy=False))
    A_values_d = cuda.to_device(A_values_flat.astype(np.uint16, copy=False))
    D_values_d = cuda.to_device(D_values_flat.astype(np.uint16, copy=False))
    cell_offsets_d = cuda.to_device(cell_offsets_flat.astype(np.uint64, copy=False))

    # Transfer model data
    V_d = cuda.to_device(V_pad)
    log_theta_d = cuda.to_device(log_theta)
    log1_theta_d = cuda.to_device(log1_theta)
    lgamma_lookup_d = cuda.to_device(lgamma_lookup.astype(np.float32, copy=False))
    top_clusters_d = cuda.to_device(top_clusters)
    pair_log_weights_d = cuda.to_device(pair_log_weights)
    log_probs_d = cuda.device_array(num_cells, dtype=np.float32)

    threadsperblock = 256  # 256 threads/block
    blockspergrid = (num_cells + threadsperblock - 1) // threadsperblock

    doublet_probs_kernel[blockspergrid, threadsperblock](positions_d, A_values_d, D_values_d,
                                                         cell_offsets_d, V_d, top_clusters_d, pair_log_weights_d,
                                                         log_theta_d, log1_theta_d, num_chr,
                                                         log_probs_d, lgamma_lookup_d, k)

    # Copy back and bring back to normal space on CPU
    log_probs = log_probs_d.copy_to_host()

    return log_probs

@cuda.jit
def singlet_probs_kernel(positions, A_values, D_values,
                         cell_offsets, V, Z,
                         log_theta, log1_theta, num_chr,
                         log_probs, lgamma_lookup):
    """
    Each thread computes the binomial data log-probability for one (cell) pair.

    :param positions: Variant index for observation p.
    :param A_values: Alternate allele count.
    :param D_values: Total depth.
    :param cell_offsets: cell_offsets[c, i+1]: absolute observation range for cell i on chromosome c.
    :param V: [num_chr, num_clusters, max_len].
    :param Z: The current best clustering.
    :param log_theta: log(thetas).
    :param log1_theta: log(1 - thetas).
    :param num_chr: Number of chromosomes.
    :param log_probs: A cell-wise array to store binomial log-prob output.
    :param lgamma_lookup: A lookup table for the lgamma function.
    """
    cell_idx = cuda.grid(1)
    num_cells = log_probs.shape[0]
    if cell_idx >= num_cells:
        return

    cluster_idx = Z[cell_idx]
    log_prob = np.float32(0.0)

    for c in range(num_chr):
        start = cell_offsets[c, cell_idx]
        end = cell_offsets[c, cell_idx + 1]

        for p in range(start, end):
            j = positions[p]
            a = A_values[p]
            d = D_values[p]
            genotype = V[c, cluster_idx, j]

            log_prob += binom_logpmf_gpu(a, d, log_theta[genotype], log1_theta[genotype], lgamma_lookup)

    log_probs[cell_idx] = log_prob

def _get_data_prob_singlet(Z, positions_flat, A_values_flat, D_values_flat,
                           cell_offsets_flat, V, thetas,lgamma_lookup):
    """
    Computes the data likelihood for each cell given the current model and clustering.
    Assumes the cell is a singlet.

    :param Z: A np.array containing a index for each cell corresponding to its cluster.
    :param positions_flat: A flattened array of indices corresponding to SNPs.
    :param A_values_flat: A flattened array of alt reads.
    :param D_values_flat:  A flattened array of depths.
    :param cell_offsets_flat: Cell offsets indexed by (chromosome, cell_idx)
    :param V: The fitted chains.
    :param thetas: Current theta values.
    :param lgamma_lookup: A lookup table for the lgamma function.
    :return data_prob_singlet:
    """
    # Pad chains so we can transfer to GPU
    num_chr = len(V) # Number of chromosomes
    num_clusters = V[0].shape[0] # Number of clusters
    chain_lengths = [V[c].shape[1] for c in range(num_chr)]
    max_len = max(chain_lengths)
    V_pad = np.zeros((num_chr, num_clusters, max_len), dtype=np.uint8)

    for c in range(num_chr):
        V_pad[c, :, :chain_lengths[c]] = V[c]

    # Clamp thetas to a safe range (in case of numerical errors)
    thetas_clipped = np.clip(thetas, a_min=1e-6, a_max=1 - 1e-6)

    # Pre-compute common logs of theta
    log_theta = np.log(thetas_clipped, dtype=np.float32)
    log1_theta = np.log(1.0 - thetas_clipped, dtype=np.float32)

    # Transfer sparse observations
    positions_d = cuda.to_device(positions_flat.astype(np.uint32, copy=False))
    A_values_d = cuda.to_device(A_values_flat.astype(np.uint16, copy=False))
    D_values_d = cuda.to_device(D_values_flat.astype(np.uint16, copy=False))
    cell_offsets_d = cuda.to_device(cell_offsets_flat.astype(np.uint64, copy=False))

    # Transfer model data
    V_d = cuda.to_device(V_pad)
    log_theta_d = cuda.to_device(log_theta)
    log1_theta_d = cuda.to_device(log1_theta)
    lgamma_lookup_d = cuda.to_device(lgamma_lookup.astype(np.float32, copy=False))
    Z_d = cuda.to_device(Z.astype(np.uint16, copy=False))

    num_cells = len(Z)
    log_probs_d = cuda.device_array(num_cells, dtype=np.float32)

    threadsperblock = 256  # 256 threads/block
    blockspergrid = (num_cells + threadsperblock - 1) // threadsperblock

    singlet_probs_kernel[blockspergrid, threadsperblock](positions_d, A_values_d, D_values_d,
                                                         cell_offsets_d, V_d, Z_d,
                                                         log_theta_d, log1_theta_d, num_chr,
                                                         log_probs_d, lgamma_lookup_d)

    # Copy back and bring back to normal space on CPU
    log_probs = log_probs_d.copy_to_host()

    return log_probs

class HMMModel:
    def __init__(self, A, D, num_clusters, expected_transitions=None,
                 z_init=None, threads=80, T_init=None, T_eps=10e-10, theta_init=None, pi_init=None,
                 do_theta_update=False, do_pi_update=True, use_gpu=True):
        """
        Args:
            A: list of alternative allele counts (n_cells x n_snps).
            D: List of total counts depth at each snp (n_cells x n_snps).
            num_clusters: The expected number of clusters.
            expected_transitions: An array of expected transitions per chromosome.
            z_init: An initial clustering.
            threads: The number of threads to use.
        """
        self.num_cells = A[0].shape[0]
        self.num_chr = len(A)
        self.num_clusters = num_clusters
        self.expected_transitions = expected_transitions
        self.num_states = 3  # The number of genotypes we can detect
        self.chain_lengths = [A[i].shape[1] for i in range(self.num_chr)]
        self.num_threads = threads
        self.do_theta_update = do_theta_update
        self.do_pi_update = do_pi_update
        self.model_prob = None
        self.use_gpu = use_gpu
        self.z_init = z_init

        # Make a sparse-friendly representation of A and D indexed by chromosomes (c)
        self.positions = []
        self.A_values = []
        self.D_values = []
        self.cell_offsets = []
        self.A_sparse = []
        self.D_sparse = []

        for c in range(self.num_chr):
            positions, A_values, D_values, offsets, A_c, D_c = utils.prepare_sparse_counts(A[c], D[c])
            self.positions.append(positions)
            self.A_values.append(A_values)
            self.D_values.append(D_values)
            self.cell_offsets.append(offsets)
            self.A_sparse.append(A_c)
            self.D_sparse.append(D_c)

        # Flatten sparse A/D representations across chromosomes for GPU
        self.obs_chr_offsets = np.zeros(self.num_chr + 1, dtype=np.uint64)

        for c in range(self.num_chr):
            self.obs_chr_offsets[c + 1] = self.obs_chr_offsets[c] + len(self.positions[c])

        total_obs = int(self.obs_chr_offsets[-1])

        self.positions_flat = np.empty(total_obs, dtype=np.uint32)
        self.A_values_flat = np.empty(total_obs, dtype=np.uint16)
        self.D_values_flat = np.empty(total_obs, dtype=np.uint16)
        self.cell_offsets_flat = np.empty((self.num_chr, self.num_cells + 1), dtype=np.uint64)

        for c in range(self.num_chr):
            start = int(self.obs_chr_offsets[c])
            end = int(self.obs_chr_offsets[c + 1])

            self.positions_flat[start:end] = self.positions[c]
            self.A_values_flat[start:end] = self.A_values[c]
            self.D_values_flat[start:end] = self.D_values[c]

            # Convert chromosome-local offsets to absolute flattened offsets
            self.cell_offsets_flat[c] = self.cell_offsets[c] + start

        self.logcomb_sparse = None
        self.__init_logcomb__() # D choose A values that are pre-computed

        # Set hyperparameters for theta based on VireoSNP paper and then initialize theta
        self.alphas = np.array([0.3, 3, 29.7])
        self.betas = np.array([29.7, 3, 0.3])

        if theta_init is not None:
            self.thetas = theta_init
        else:
            self.__init_theta__()

        # Set pi. We note that a mixed genotype start is more likely (see notes)
        if pi_init is not None:
            self.pi = pi_init
            self.pi_init = pi_init
        else:
            self.pi = np.array([0.25, 0.5, 0.25])
            self.pi_init = np.array([0.25, 0.5, 0.25])

        if expected_transitions is None and T_init is None:
            raise ValueError("Either T_init or expected_transitions must be given.")

        # Initialize T
        self.T_eps = T_eps
        if T_init is not None:
            self.T = T_init
        else:
            self.__init_T__()

        # Initialize V
        self.__init_chains__()

        # Initialize clustering
        self.kappas = np.array([1/self.num_clusters for _ in range(num_clusters)])

        if z_init is not None:
            self.Z = z_init
            self.Z_init = z_init
        else:
            self.Z = np.random.choice(self.num_clusters,
                                      size=self.num_cells,
                                      replace=True,
                                      p=self.kappas)
            self.Z_init = None

        # Create a lookup table for the lgamma function up to the maximum seen depth to save time in iterations
        max_depth = int(np.max([np.max(D[c]) for c in range(self.num_chr)]))
        self.lgamma_lookup = make_lgamma_lookup(max_depth)

    def __init_logcomb__(self):
        """
        Pre-computes the values of D[c][i][j] choose A[c][i][j] in sparse format.
        :return:
        """
        self.logcomb_sparse = []

        for c in range(self.num_chr):
            # Based on the gamma function definition of D choose A
            logcomb_c =  gammaln(self.D_values[c].astype(np.float32) + 1) \
                         - gammaln(self.A_values[c].astype(np.float32) + 1)\
                         - gammaln(self.D_values[c].astype(np.float32) - self.A_values[c].astype(np.float32) + 1)

            logcomb_c_sparse = csr_matrix((logcomb_c, self.D_sparse[c].indices, self.D_sparse[c].indptr),
                                         shape=self.D_sparse[c].shape)

            self.logcomb_sparse.append(logcomb_c_sparse)

    def __init_T__(self):
        """
        Initializes the global transition matrix based on the data and
        the amount of transitions we expect to see per chromosome.
        """
        T = np.zeros(shape=(self.num_chr, self.num_states, self.num_states))

        for c in range(self.num_chr):
            p = self.expected_transitions[c] / self.chain_lengths[c]
            T[c, :, :] = np.array([[1-p, p, 0],
                                  [p/2, 1-p, p/2],
                                  [0, p, 1-p]])
            T[c, :, :] = (T[c, :, :] + self.T_eps) / (1 + 3*self.T_eps)
        self.T = T

    def __init_theta__(self):
        # Sample thetas from the prior
        self.thetas = np.array([beta_dist.rvs(alpha, beta) for alpha, beta in zip(self.alphas, self.betas)])

    def __init_chains__(self):
        # Initialize V as a list of K x n_variants matrices for each chromosome
        self.V = [np.zeros(shape=(self.num_clusters, self.chain_lengths[c]), dtype=int) for c in range(self.num_chr)]

        # Initialize V[c][i][0] from pi and then sample transitions for the chain
        for c in range(self.num_chr):
            for i in range(self.num_clusters):
                # Initialize
                self.V[c][i][0] = int(np.random.choice(3, p=self.pi))

                for j in range(1, len(self.V[c][i])):
                    # Select the row of the transition matrix to sample based on the previous value
                    T_row = self.T[c][self.V[c][i][j-1]]
                    self.V[c][i][j] = int(np.random.choice(3, p=T_row))

    def get_z_probs(self):
        """
        Computes z_probs using CPU or GPU.
        """
        if self.use_gpu:
            self.z_probs = get_z_probs_gpu(
                self.num_cells, self.num_clusters, self.num_chr, self.chain_lengths,
                self.positions_flat, self.A_values_flat, self.D_values_flat, self.cell_offsets_flat,
                self.V, self.thetas, self.kappas, self.lgamma_lookup
            )
        else:
            raise NotImplementedError("CPU only not currently implemented for z probabilities.")

        self.z_probs = np.clip(self.z_probs, 1e-12, 1.0)

        for i in range(self.num_cells):
            self.z_probs[i] = self.z_probs[i] / np.sum(self.z_probs[i])  # re-normalize

    def set_z(self):
        """
        Based on precomputed probabilities for z_i = k, sets z_i
        to its maximum likelihood value.
        """
        self.Z = np.argmax(self.z_probs, axis=1)

    def forward_backward_pass(self):
        if self.use_gpu:
            log_likelihoods = precompute_cluster_log_likelihoods(
                self.A_sparse, self.D_sparse, self.Z, self.logcomb_sparse, self.thetas,
                self.num_chr, self.num_clusters, self.chain_lengths, threads=self.num_threads
            )

            log_alphas, log_betas = forward_backward_gpu_optimized(
                log_likelihoods, self.thetas, self.T, self.pi, self.kappas,
                self.alphas, self.betas, self.chain_lengths,
                self.num_chr, self.num_clusters, max(self.chain_lengths)
            )

            self.log_alphas_V = [[log_alphas[c, k] for k in range(self.num_clusters)] for c in range(self.num_chr)]
            self.log_betas_V = [[log_betas[c, k] for k in range(self.num_clusters)] for c in range(self.num_chr)]
        else:
            raise NotImplementedError("CPU only not currently implemented for forward-backward pass.")


    def get_chain_probabilities(self):
        """
        Uses computed alphas and betas for V to get the probability of each possible
        state in each chain position. GPU-first with CPU fallback.
        """
        # Initialize the prob matrix (list of arrays per chromosome to match existing API)
        probs_V = []
        for c in range(self.num_chr):
            chain_length = self.chain_lengths[c]
            probs_V_c = np.zeros(shape=(self.num_clusters, chain_length, 3), dtype=np.float32)
            probs_V.append(probs_V_c)

        if self.use_gpu:
            try:
                out = chain_probs_gpu(self.num_chr, self.num_clusters,
                                      self.chain_lengths, self.log_alphas_V, self.log_betas_V)

                # Unpack back into the expected list-of-arrays shape
                for c in range(self.num_chr):
                    L = self.chain_lengths[c]
                    probs_V[c][:, :L, :] = out[c, :, :L, :]

                self.probs_V = probs_V
                return
            except Exception as e:
                print(f"[GPU Fallback] Chain probs GPU failed: {e}. Falling back to CPU.")

        # CPU fallback (original implementation)
        for c in tqdm(range(self.num_chr), desc='chain probs'):
            chain_length = len(self.V[c][0])

            for k in range(self.num_clusters):
                # Compute denominator
                log_prob_joint = logsumexp(self.log_alphas_V[c][k][chain_length - 1])

                if np.isnan(log_prob_joint):
                    print('log_prob_joint is nan', self.log_alphas_V[c][k][chain_length - 1])

                for l in range(chain_length):
                    # Vectorized over t=0..2
                    log_vec = self.log_alphas_V[c][k][l] + self.log_betas_V[c][k][l] - log_prob_joint
                    if np.any(np.isnan(log_vec)):
                        print('log_prob is nan', l, self.log_alphas_V[c][k][l], self.log_betas_V[c][k][l])

                    row = np.exp(log_vec)
                    s = row.sum()
                    if s <= 0 or np.isnan(s) or np.isinf(s):
                        probs_V[c][k][l] = np.array([1/3, 1/3, 1/3], dtype=np.float32)
                    else:
                        probs_V[c][k][l] = (row / s).astype(np.float32)

        self.probs_V = probs_V

    def set_V(self):
        """
        Assumes that the probabilities of each genotype state have already been computed.
        Sets the chain to their maximum likelihood state.
        """
        for c in range(self.num_chr):
            for k in range(self.num_clusters):
                probs = self.probs_V[c][k]
                max_likelihood_chain = np.argmax(probs, axis=1)
                self.V[c][k] = max_likelihood_chain

    def update_pi(self):
        """
        Updates pi_t according to the loss function.
        """
        # For each genotype sum the probabilities for the chain to start at that genotype
        start_probs = np.zeros(self.num_states)

        for t in range(self.num_states):
            for c in range(self.num_chr):
                for k in range(self.num_clusters):
                    start_probs[t] += self.probs_V[c][k][0][t]

        # pi is the normalization of this vector
        self.pi = start_probs / np.sum(start_probs)
        self.pi = np.clip(self.pi, 0.001, 0.999)
        self.pi = self.pi / np.sum(self.pi)

    def update_theta(self):
        """
        Update theta according to the loss function.
        """
        numerators = np.zeros(self.num_states)
        denominators = np.zeros(self.num_states)

        # Add the prior contributions
        numerators += self.alphas - 1
        denominators += self.alphas + self.betas - 2

        # Allow threads to share memory and split up the computation by chromosome
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(get_theta_chr,self.z_probs,
                                       self.A_sparse[c],self.D_sparse[c], self.probs_V[c]) for c in range(self.num_chr)]

            for future in futures:
                numerator, denominator = future.result()
                numerators += numerator
                denominators += denominator

        # Set thetas and make sure they are in (0, 1)
        self.thetas = np.array([np.clip(numerators[t] / denominators[t], 0.001, 0.999) for t in range(self.num_states)])
        print(f'New thetas are {self.thetas}')

    def get_model_prob(self):
        """
        Approximates the log probability of the model using alphas
        """
        log_joint = 0

        # Fast approximation
        for c in range(self.num_chr):
            for k in range(self.num_clusters):
                cur_v_val = self.V[c][k][self.chain_lengths[c] - 1]
                log_joint += self.log_alphas_V[c][k][self.chain_lengths[c] - 1][cur_v_val]

        return log_joint

    def get_per_cluster_model_prob(self, k):
        """
        Approximates the log probability of the model using alphas
        """
        log_joint = 0

        # Fast approximation
        for c in range(self.num_chr):
            cur_v_val = self.V[c][k][self.chain_lengths[c] - 1]
            log_joint += self.log_alphas_V[c][k][self.chain_lengths[c] - 1][cur_v_val]

        return log_joint

    def identify_doublets(self, k=5, prior_doublet=0.5):
        """
        Runs the doublet detection method on the current model.
        Fit the model before running doublet detection.

        :param k: (int) The number of best candidates to consider for each cell.
        :param prior_doublet: (float) The prior probability that a randomly chosen cell is a doublet.
        :return p_doublet: A np.array of doublet probabilities for each cell.
        """
        # Check that k is small enough to reasonably compute
        if k <= 1 or k > MAX_K:
            raise ValueError(f'k must be 1< k <= {MAX_K}')

        log_p_data_given_doublet = _get_data_prob_doublet(self.z_probs, self.positions_flat,
                                                      self.A_values_flat, self.D_values_flat,
                                                      self.cell_offsets_flat, self.V,
                                                      self.thetas, self.lgamma_lookup, k)
        log_p_data_given_singlet = _get_data_prob_singlet(self.Z, self.positions_flat,
                                                      self.A_values_flat, self.D_values_flat,
                                                      self.cell_offsets_flat, self.V,
                                                      self.thetas,self.lgamma_lookup)

        print(f'Prob doublet: {log_p_data_given_doublet}')
        print(f'Prob singlet: {log_p_data_given_singlet}')

        # Compute the doublet likelihood in log-space
        log_prior_doublet = np.log(prior_doublet)
        log_prior_singlet = np.log(1.0 - prior_doublet)

        log_num = log_p_data_given_doublet + log_prior_doublet
        log_den = np.logaddexp(log_num, log_p_data_given_singlet + log_prior_singlet)

        p_doublet = np.exp(log_num - log_den)

        print('Nan values in doublet detection', np.any(np.isnan(p_doublet)))

        return p_doublet

    def update(self):
        # Update z
        start_time = time.time()
        self.get_z_probs()
        self.set_z()
        end_time = time.time()
        print(f"Z update completed in {end_time - start_time:.4f} seconds (GPU: {self.use_gpu})")

        # Update V
        start_time = time.time()
        self.forward_backward_pass()
        self.get_chain_probabilities()
        self.set_V()
        end_time = time.time()
        print(f"Forward-backward completed in {end_time - start_time:.4f} seconds (GPU: {self.use_gpu})")


        if self.do_theta_update:
            start_time = time.time()
            self.update_theta()
            end_time = time.time()
            print(f"theta update completed in {end_time - start_time:.4f} seconds (GPU: {self.use_gpu})")

        if self.do_pi_update:
            start_time = time.time()
            self.update_pi()
            end_time = time.time()
            print(f"pi update completed in {end_time - start_time:.4f} seconds (GPU: {self.use_gpu})")

    def solve(self, tol, iters=25):
        """
        Performs EM iterations until convergence.
        Returns:
        """
        prev_prob = np.inf

        # If we are warm starting from a cluster initialization
        if self.z_init is not None:
            self.forward_backward_pass()
            self.get_chain_probabilities()
            self.set_V()

        for i in range(iters):
            # Do EM iterations
            t0 = time.time()
            self.update()
            self.model_prob = self.get_model_prob()
            t1 = time.time()
            time_per_iteration = t1 - t0

            print(f'log fit probability {self.model_prob}', f'Took {time_per_iteration} s')
            if np.abs(self.model_prob - prev_prob) < tol:
                print('Converged!')
                break

            prev_prob = self.model_prob

    def get_output(self):
        per_cluster_joints = np.array([self.get_per_cluster_model_prob(i) for i in range(self.num_clusters)])

        model_out = {'z_probs': self.z_probs, 'V': self.V, 'pi': self.pi,
                     'theta': self.thetas, 'probability': self.model_prob,
                     'cluster_joints': per_cluster_joints}
        return model_out