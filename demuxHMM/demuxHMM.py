from numba import njit, cuda, float32, int32
from math import lgamma

import scipy
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import time
import math

@njit
def xlogy(x, y):
    """
    Computes xlog(y) so that it is 0 if x is 0.
    Args:
        x:
        y:
    Returns:
    """
    if x == 0:
        return 0

    return x*np.log(y)

@njit
def xlog1py(x, y):
    """
    Calculates xlog1p(y) so that it is 0 if x is 0.
    Args:
        x:
        y:
    Returns:
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

@cuda.jit(device=True)
def xlogy_gpu(x, y):
    return 0.0 if x == 0 else x * math.log(y)

@cuda.jit(device=True)
def xlog1py_gpu(x, y):
    return 0.0 if x == 0 else x * math.log1p(y)

@cuda.jit(device=True)
def binom_logpmf_gpu(k, n, p):
    p = min(max(p, 1e-6), 1 - 1e-6)  # clamp probability away from 0 or 1
    combiln = lgamma(n + 1) - (lgamma(k + 1) + lgamma(n - k + 1))
    return combiln + xlogy_gpu(k, p) + xlog1py_gpu(n - k, -p)

@cuda.jit(device=True)
def _lse3(a0, a1, a2):
    # logsumexp for 3 floats (stable)
    m = max(a0, a1) if max(a0, a1) > a2 else a2
    return math.log(math.exp(a0 - m) + math.exp(a1 - m) + math.exp(a2 - m)) + m

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

@njit
def numb_cell_z_prob(k, num_cells, num_chr, chain_lengths, A, V, D, thetas, kappas):
    log_probs_col = np.zeros(num_cells)

    for i in range(num_cells):
        for c in range(num_chr):
            for j in range(chain_lengths[c]):
                genotype = V[c][k][j]
                log_probs_col[i] += binom_logpmf(A[c][i][j], D[c][i][j], thetas[genotype])

    log_probs_col += np.log(kappas[k])

    return log_probs_col

def stable_softmax(log_probs):
    a_max = np.max(log_probs)
    exp_vals = np.exp(log_probs - a_max)
    sum_exp = np.sum(exp_vals)

    if sum_exp == 0 or np.isnan(sum_exp):
        # fallback to uniform
        return np.ones_like(log_probs) / len(log_probs)
    return exp_vals / sum_exp


@cuda.jit
def z_probs_kernel(A, D, V, thetas, kappas, chain_lens, num_chr, log_probs):
    """
    Each thread computes the log-probability for a single (cell, cluster) pair across all chromosomes.
    A, D: [num_chr, num_cells, max_len]
    V:    [num_chr, num_clusters, max_len]
    thetas: (3,)
    kappas: (num_clusters,)
    chain_lens: (num_chr,)
    log_probs: [num_cells, num_clusters] (output)
    """
    cell_idx, cluster_idx = cuda.grid(2)
    num_cells, num_clusters = log_probs.shape

    if cell_idx >= num_cells or cluster_idx >= num_clusters:
        return

    log_prob = 0.0

    for c in range(num_chr):
        chain_len = chain_lens[c]

        for j in range(chain_len):
            genotype = V[c, cluster_idx, j]
            a = A[c, cell_idx, j]
            d = D[c, cell_idx, j]
            if d > 0:
                log_prob += binom_logpmf_gpu(a, d, thetas[genotype])

    log_prob += math.log(kappas[cluster_idx])
    log_probs[cell_idx, cluster_idx] = log_prob

def get_z_probs_gpu(num_cells, num_clusters, num_chr, chain_lengths, A, D, V, thetas, kappas):
    max_len = max(chain_lengths)

    A_pad = np.zeros((num_chr, num_cells, max_len), dtype=np.int32)
    D_pad = np.zeros((num_chr, num_cells, max_len), dtype=np.int32)
    V_pad = np.zeros((num_chr, num_clusters, max_len), dtype=np.int32)

    for c in range(num_chr):
        A_pad[c, :, :chain_lengths[c]] = A[c]
        D_pad[c, :, :chain_lengths[c]] = D[c]
        V_pad[c, :, :chain_lengths[c]] = V[c]

    # Transfer to device
    A_d = cuda.to_device(A_pad)
    D_d = cuda.to_device(D_pad)
    V_d = cuda.to_device(V_pad)
    thetas_d = cuda.to_device(thetas.astype(np.float32))
    kappas_d = cuda.to_device(kappas.astype(np.float32))
    chain_lens_d = cuda.to_device(np.array(chain_lengths, dtype=np.int32))

    log_probs_d = cuda.device_array((num_cells, num_clusters), dtype=np.float32)

    threadsperblock = (16, 16)  # 256 threads/block
    blockspergrid = (
        (num_cells + threadsperblock[0] - 1) // threadsperblock[0],
        (num_clusters + threadsperblock[1] - 1) // threadsperblock[1]
    )

    z_probs_kernel[blockspergrid, threadsperblock](
        A_d, D_d, V_d, thetas_d, kappas_d, chain_lens_d, num_chr, log_probs_d
    )

    # Copy back and normalize on CPU
    log_probs = log_probs_d.copy_to_host()
    probs = np.zeros_like(log_probs)
    for i in range(num_cells):
        probs[i] = stable_softmax(log_probs[i])

    return probs

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

def get_cell_z_prob(args):
    k, num_cells, num_chr, chain_lengths, A, V, D, thetas, kappas = args
    return numb_cell_z_prob(k, num_cells, num_chr, chain_lengths, A, V, D, thetas, kappas)

@njit
def numb_get_forward_prob(c, k, Z, chain_lengths, alphas, betas, kappas, pi, A_c, D_c, thetas, T_c):
    # Get the indices of cells in cluster k
    cluster_inds = np.where(Z == k)[0]

    log_alpha_V_ck = np.zeros(shape=(chain_lengths[c], 3))

    # Set the initial conditions for each possible genotype
    for t in range(3):
        # Work in log space
        log_alpha_t = 0

        # Add the contribution from the data and clustering
        for i in cluster_inds:
            logpmf = binom_logpmf(A_c[i][0], D_c[i][0], thetas[t])
            if np.isnan(logpmf) or np.isinf(logpmf):
                print('logpmf isnan', A_c[i][0], D_c[i][0], thetas[t])

            log_alpha_t += logpmf
            log_alpha_t += np.log(kappas[k])

        # Add the contribution from theta
        theta_cont = beta_logpdf(thetas[t], alphas[t], betas[t])
        if np.isnan(theta_cont) or np.isinf(theta_cont):
            print('theta cont nan', thetas[t], alphas[t], betas[t])

        log_alpha_t += theta_cont

        # Add the prior on v_c,k,0
        log_alpha_t += np.log(pi[t])
        log_alpha_V_ck[0][t] = log_alpha_t

    for l in range(1, chain_lengths[c]):
        for t in range(3):
            # We will do a sum over the values of alpha from the previous step using the lse trick
            lse_terms = np.zeros(3)

            # Precompute the data contribution term
            data_term = 0
            for i in cluster_inds:
                pmf = binom_logpmf(A_c[i][l], D_c[i][l], thetas[t])
                data_term += pmf

            lse_terms += data_term

            for t_prev in range(3):
                lse_terms[t_prev] += log_alpha_V_ck[l-1][t_prev]  # Previous alpha contribution
                lse_terms[t_prev] += np.log(T_c[t_prev, t])  # Transition probability

            if np.any(np.isinf(lse_terms)):
                print(data_term, log_alpha_V_ck[l-1][t_prev], l)
                raise ValueError('inf')

            lse = _logsumexp(lse_terms)
            # if np.isnan(lse) or np.isinf(lse):
            #     print('lse is nan', lse_terms)
            log_alpha_V_ck[l][t] = lse

            if np.isinf(log_alpha_V_ck[l][t]) or np.isnan(log_alpha_V_ck[l][t]):
                print(data_term, log_alpha_V_ck[l-1], l)
                raise ValueError('inf or nan')

    return log_alpha_V_ck

@njit
def numb_get_backward_prob(c, k, Z, chain_lengths, A_c, D_c, thetas, T_c):
    # Get the indices of cells in cluster k
    cluster_inds = np.where(Z == k)[0]

    # We do not need to do any initialization because beta(c, k, n-1, :) = 1 and log(1) = 0
    log_beta_V_ck = np.zeros(shape=(chain_lengths[c], 3))

    for l in range(chain_lengths[c] - 2, -1, -1):
        for t in range(3):
            lse_terms = np.zeros(3)

            for t_next in range(3):
                # Add data term contribution
                for i in cluster_inds:
                    lse_terms[t_next] += binom_logpmf(A_c[i][l + 1], D_c[i][l + 1], thetas[t_next])

                # Add the higher indexed Beta term contribution
                lse_terms[t_next] += log_beta_V_ck[l + 1][t_next]

                # Add the transition probability
                lse_terms[t_next] += np.log(T_c[t_next, t])

            log_beta_V_ck[l][t] = _logsumexp(lse_terms)

            if np.isinf(log_beta_V_ck[l][t]) or np.isnan(log_beta_V_ck[l][t]):
                print(lse_terms, log_beta_V_ck[l][t], l, 'beta')
                raise ValueError('inf or nan')

    return log_beta_V_ck

@njit
def precompute_cluster_log_likelihoods(A, D, Z, thetas, num_chr, num_clusters, chain_lengths, num_states=3):
    """Precompute ∑ log_binom_pmf for all (chr, cluster, pos, state)."""
    max_len = max(chain_lengths)
    log_likelihoods = np.zeros((num_chr, num_clusters, max_len, num_states), dtype=np.float32)

    for c in range(num_chr):
        L = chain_lengths[c]
        for k in range(num_clusters):
            cell_inds = np.where(Z == k)[0]
            for l in range(L):
                for t in range(num_states):
                    p = thetas[t]
                    total = 0.0
                    for i in cell_inds:
                        d = D[c][i][l]
                        a = A[c][i][l]
                        if d > 0:
                            total += binom_logpmf(a, d, p)
                    log_likelihoods[c, k, l, t] = total

    return log_likelihoods

@cuda.jit
def forward_backward_likelihood_kernel(log_likelihoods, thetas, T, pi, kappas,
                                       alphas, betas,
                                       chain_lengths, log_alphas, log_betas,
                                       num_chr, num_clusters, num_states, max_len):

    c_idx, k_idx = cuda.grid(2)
    if c_idx >= num_chr or k_idx >= num_clusters:
        return

    L = chain_lengths[c_idx]
    # TODO: The first dimension should be the max chain length
    log_alpha = cuda.local.array((4500, 3), dtype=float32)
    log_beta = cuda.local.array((4500, 3), dtype=float32)

    # === Forward pass ===
    for t in range(num_states):
        # Theta prior (rough beta_logpdf)
        prior = xlog1py_gpu(betas[t] - 1.0, -thetas[t]) + xlogy_gpu(alphas[t] - 1.0, thetas[t])
        prior -= xlogy_gpu(alphas[t], 1.0) + xlogy_gpu(betas[t], 1.0) - xlogy_gpu(alphas[t] + betas[t], 1.0)

        log_alpha[0][t] = log_likelihoods[c_idx, k_idx, 0, t] + math.log(kappas[k_idx]) + math.log(pi[t]) + prior

    for l in range(1, L):
        for t in range(num_states):
            vals = cuda.local.array(3, dtype=float32)
            for t_prev in range(num_states):
                vals[t_prev] = log_alpha[l - 1][t_prev] + math.log(T[c_idx, t_prev, t])
            max_val = max(vals[0], max(vals[1], vals[2]))
            lse = math.log(math.exp(vals[0] - max_val) +
                           math.exp(vals[1] - max_val) +
                           math.exp(vals[2] - max_val)) + max_val
            log_alpha[l][t] = log_likelihoods[c_idx, k_idx, l, t] + lse

    # === Backward pass ===
    for t in range(num_states):
        log_beta[L - 1][t] = 0.0

    for l in range(L - 2, -1, -1):
        for t in range(num_states):
            vals = cuda.local.array(3, dtype=float32)
            for t_next in range(num_states):
                vals[t_next] = (
                    log_likelihoods[c_idx, k_idx, l + 1, t_next] +
                    log_beta[l + 1][t_next] +
                    math.log(T[c_idx, t_next, t])
                )
            max_val = max(vals[0], max(vals[1], vals[2]))
            lse = math.log(math.exp(vals[0] - max_val) +
                           math.exp(vals[1] - max_val) +
                           math.exp(vals[2] - max_val)) + max_val
            log_beta[l][t] = lse

    # === Write to global memory
    for l in range(L):
        for t in range(num_states):
            log_alphas[c_idx, k_idx, l, t] = log_alpha[l][t]
            log_betas[c_idx, k_idx, l, t] = log_beta[l][t]


def get_forward_backward(args):
    c, k, Z, chain_lengths, alphas, betas, kappas, pi, A_c, D_c, thetas, T_c = args

    log_alpha = numb_get_forward_prob(c, k, Z, chain_lengths, alphas, betas, kappas, pi, A_c, D_c, thetas, T_c)
    log_beta = numb_get_backward_prob(c, k, Z, chain_lengths, A_c, D_c, thetas, T_c)

    return c, k, log_alpha, log_beta

def forward_backward_gpu_optimized(log_likelihoods, thetas, T, pi, kappas, alphas, betas,
                                   chain_lengths, num_chr, num_clusters, max_len):
    num_states = 3

    # Move everything to device
    thetas_d = cuda.to_device(thetas.astype(np.float32))
    T_d = cuda.to_device(T.astype(np.float32))
    pi_d = cuda.to_device(pi.astype(np.float32))
    kappas_d = cuda.to_device(kappas.astype(np.float32))
    alphas_d = cuda.to_device(alphas.astype(np.float32))
    betas_d = cuda.to_device(betas.astype(np.float32))
    lens_d = cuda.to_device(np.array(chain_lengths, dtype=np.int32))

    log_likelihoods_d = cuda.to_device(log_likelihoods)
    log_alphas_d = cuda.device_array((num_chr, num_clusters, max_len, num_states), dtype=np.float32)
    log_betas_d = cuda.device_array_like(log_alphas_d)

    threadsperblock = (8, 8)
    blockspergrid = (
        (num_chr + threadsperblock[0] - 1) // threadsperblock[0],
        (num_clusters + threadsperblock[1] - 1) // threadsperblock[1]
    )

    forward_backward_likelihood_kernel[blockspergrid, threadsperblock](
        log_likelihoods_d, thetas_d, T_d, pi_d, kappas_d,
        alphas_d, betas_d,
        lens_d, log_alphas_d, log_betas_d,
        num_chr, num_clusters, num_states, max_len
    )

    return log_alphas_d.copy_to_host(), log_betas_d.copy_to_host()

@njit
def numb_get_thetas(t, k, num_states, num_chr, num_cells, chain_lengths, z_probs, probs_V, A, D):
    numerators = np.zeros(num_states)
    denominators = np.zeros(num_states)

    # Add the data contributions
    for i in range(num_cells):
        for c in range(num_chr):
            for j in range(chain_lengths[c]):
                if A[c][i][j] != 0:
                    numerators[t] += z_probs[i][k] * probs_V[c][k][j][t] * A[c][i][j]
                if D[c][i][j] != 0:
                    denominators[t] += z_probs[i][k] * probs_V[c][k][j][t] * D[c][i][j]

    return [numerators, denominators]

def get_thetas(args):
    t, k, num_states, num_chr, num_cells, chain_lengths, z_probs, probs_V, A, D = args
    return numb_get_thetas(t, k, num_states, num_chr, num_cells, chain_lengths, z_probs, probs_V, A, D)


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

    # log p(data) = logsumexp(log_alpha at final position over states)
    la0 = log_alphas[c, k, L - 1, 0]
    la1 = log_alphas[c, k, L - 1, 1]
    la2 = log_alphas[c, k, L - 1, 2]
    log_joint = _lse3(la0, la1, la2)

    # unnormalized posteriors in linear space
    s0 = math.exp(log_alphas[c, k, l, 0] + log_betas[c, k, l, 0] - log_joint)
    s1 = math.exp(log_alphas[c, k, l, 1] + log_betas[c, k, l, 1] - log_joint)
    s2 = math.exp(log_alphas[c, k, l, 2] + log_betas[c, k, l, 2] - log_joint)

    s = s0 + s1 + s2
    # numerical guard + normalization
    if s <= 0.0 or math.isnan(s) or math.isinf(s):
        out_probs[c, k, l, 0] = 1.0 / 3.0
        out_probs[c, k, l, 1] = 1.0 / 3.0
        out_probs[c, k, l, 2] = 1.0 / 3.0
    else:
        invs = 1.0 / s
        out_probs[c, k, l, 0] = s0 * invs
        out_probs[c, k, l, 1] = s1 * invs
        out_probs[c, k, l, 2] = s2 * invs

def get_double_indicator(args):
    """
    Computes the double indicator needed to update the transition matrix.
    Args:
        args:

    Returns: log-prob of the double indicator given the data
    """
    (k, c, l, v_c_k_prev, v_c_k_cur, chain_lengths,
     log_alphas_c, log_betas_c, A_c, D_c, z, kappas, thetas, theta_alphas, theta_betas, T_c) = args

    log_probs = np.zeros(chain_lengths[c] - 1) # Record the transition probabilities for all pairs in the chain
    cluster_inds = np.where(z == k)[0]
    num_cells = len(cluster_inds)

    for l in range(1, chain_lengths[c]):
        prob = 0
        prob += log_alphas_c[k][l-1][v_c_k_prev] # Add the alpha contribution
        prob += log_betas_c[k][l][v_c_k_cur] # Add the beta contribution

        for i in cluster_inds:
            prob += binom_logpmf(A_c[i][0], D_c[i][0], thetas[v_c_k_cur]) # Add the data contribution

        prob += np.log(T_c[v_c_k_prev, v_c_k_cur]) # Transition contribution

        # Denominator
        prob -= num_cells * np.log(kappas[k]) # Cluster contribution
        prob -= beta_logpdf(thetas[v_c_k_prev],
                                        theta_alphas[v_c_k_prev], theta_betas[v_c_k_prev]) # Theta prior
        prob -= _logsumexp(log_alphas_c[k][chain_lengths[c] - 1]) # Probability of the data
        log_probs[l-1] = prob

    return log_probs

class HMMModel:
    def __init__(self, A, D, num_clusters, expected_transitions=None,
                 z_init=None, threads=80, T_init=None, theta_init=None, pi_init=None,
                 do_theta_update=False, do_pi_update=True):
        """

        Args:
            A: list of alternative allele counts (n_cells x n_snps).
            D: List of total counts depth at each snp (n_cells x n_snps).
            num_chr: The number of chromosomes being modelled.
            num_clusters: The expected number of clusters.
            expected_transitions: An array of expected transitions per chromosome.
            z_init: An initial clustering.
            threads: The number of threads to use.
        """
        self.A = A
        self.D = D
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
        if T_init is not None:
            self.T = T_init
        else:
            self.__init_T__()

        # Initialize V
        self.__init_chains__()

        # Initialize clustering
        self.kappas = np.array([1/self.num_clusters for i in range(num_clusters)])

        if z_init is not None:
            self.Z = z_init
            self.Z_init = z_init
        else:
            self.Z = np.random.choice(self.num_clusters,
                                      size=self.num_cells,
                                      replace=True,
                                      p=self.kappas)
            self.Z_init = None

    def __init_T__(self):
        """
        Initializes the global transition matrix based on the data and
        the amount of transitions we expect to see per chromosome.
        """
        
        # Find the average number of variants on a chromosome
        num_vars = [mat.shape[1] for mat in self.A]

        T = np.zeros(shape=(self.num_chr, self.num_states, self.num_states))

        for c in range(self.num_chr):
            p = self.expected_transitions[c] / num_vars[c]
            T[c, :, :] = np.array([[1-p, p, 10**(-10)],
                                  [p/2, 1-p, p/2],
                                  [10**(-10), p, 1-p]])
        self.T = T

    def __init_theta__(self):
        # Sample thetas from the prior
        self.thetas = np.array([scipy.stats.beta.rvs(alpha, beta) for alpha, beta in zip(self.alphas, self.betas)])

    def __init_chains__(self):
        # Initialize V as a list of K x n_variants matrices for each chromosome
        self.V = [np.zeros(shape=(self.num_clusters, self.A[i].shape[1]), dtype=int) for i in range(self.num_chr)]

        # Initialize V[c][i][0] from pi and then sample transitions for the chain
        for c in range(self.num_chr):
            for i in range(self.num_clusters):
                # Initialize
                self.V[c][i][0] = int(np.random.choice(3, p=self.pi))

                for j in range(1, len(self.V[c][i])):
                    # Select the row of the transition matrix to sample based on the previous value
                    T_row = self.T[c][self.V[c][i][j-1]]
                    self.V[c][i][j] = int(np.random.choice(3, p=T_row))

    def init_pool(self):
        self.pool = Pool(self.num_threads)

    def close_pool(self):
        self.pool.close()

    def get_output(self):
        model_out = {'z_probs': self.z_probs, 'V': self.V, 'pi': self.pi,
                     'theta': self.thetas, 'probability': self.model_prob}
        return model_out

    def update(self):
        # Update z
        self.get_z_probs()
        self.set_z()

        # Update V
        self.forward_backward_pass()
        self.get_chain_probabilities()
        self.set_V()

        if self.do_theta_update:
            self.update_theta()

        if self.do_pi_update:
            self.update_pi()


    def get_z_probs(self, use_gpu=True):
        """
        Computes z_probs using CPU or GPU.
        """
        start_time = time.time()
        if use_gpu:
            self.z_probs = get_z_probs_gpu(
                self.num_cells, self.num_clusters, self.num_chr,
                self.chain_lengths, self.A, self.D, self.V,
                self.thetas, self.kappas
            )
        else:
            probs = np.zeros((self.num_cells, self.num_clusters))
            args = [
                [k, self.num_cells, self.num_chr, self.chain_lengths,
                 self.A, self.V, self.D, self.thetas, self.kappas]
                for k in range(self.num_clusters)
            ]
            prob_cols = list(tqdm(self.pool.imap(get_cell_z_prob, args), total=self.num_clusters, desc='Z Probs'))

            for k, col in enumerate(prob_cols):
                probs[:, k] = col

            probs = normalize_rows(probs)
            self.z_probs = probs

        self.z_probs = np.clip(self.z_probs, 1e-12, 1.0)

        for i in range(self.num_cells):
            self.z_probs[i] = self.z_probs[i] / np.sum(self.z_probs[i])  # re-normalize

        end_time = time.time()
        print(f"get_z_probs completed in {end_time - start_time:.4f} seconds (GPU: {use_gpu})")

    def set_z(self):
        """
        Based on precomputed probabilities for z_i = k, sets z_i
        to its maximum likelihood value.
        """
        self.Z = np.argmax(self.z_probs, axis=1)

    def forward_backward_pass(self, use_gpu=True):
        if use_gpu:
            try:
                log_likelihoods = precompute_cluster_log_likelihoods(
                    self.A, self.D, self.Z, self.thetas,
                    self.num_chr, self.num_clusters, self.chain_lengths
                )

                log_alphas, log_betas = forward_backward_gpu_optimized(
                    log_likelihoods, self.thetas, self.T, self.pi, self.kappas,
                    self.alphas, self.betas, self.chain_lengths,
                    self.num_chr, self.num_clusters, max(self.chain_lengths)
                )

                self.log_alphas_V = [[log_alphas[c, k] for k in range(self.num_clusters)] for c in range(self.num_chr)]
                self.log_betas_V = [[log_betas[c, k] for k in range(self.num_clusters)] for c in range(self.num_chr)]

                return
            except Exception as e:
                print(f"[GPU Fallback] Forward-backward GPU failed: {e}. Falling back to CPU.")
                # Fallback to CPU below

        # CPU fallback
        log_alphas_V = [[] for _ in range(self.num_chr)]
        log_betas_V = [[] for _ in range(self.num_chr)]
        args = []
        for c in range(self.num_chr):
            for k in range(self.num_clusters):
                args.append([c, k, self.Z, self.chain_lengths, self.alphas,
                             self.betas, self.kappas, self.pi, self.A[c], self.D[c],
                             self.thetas, self.T[c]])

        results = list(tqdm(self.pool.imap(get_forward_backward, args),
                            total=len(args), desc='Forward+Backward'))

        for c, k, log_alpha, log_beta in results:
            log_alphas_V[c].append(log_alpha)
            log_betas_V[c].append(log_beta)

        self.log_alphas_V = log_alphas_V
        self.log_betas_V = log_betas_V

    def _pack_logs_to_padded(self):
        """
        Packs self.log_alphas_V / self.log_betas_V (lists) into
        padded arrays [num_chr, num_clusters, max_len, 3] for GPU.
        """
        num_chr = self.num_chr
        K = self.num_clusters
        S = self.num_states
        max_len = max(self.chain_lengths)

        la_pad = np.zeros((num_chr, K, max_len, S), dtype=np.float32)
        lb_pad = np.zeros_like(la_pad)

        for c in range(num_chr):
            L = self.chain_lengths[c]

            for k in range(K):
                # each is (L,3)
                la = self.log_alphas_V[c][k].astype(np.float32)
                lb = self.log_betas_V[c][k].astype(np.float32)

                la_pad[c, k, :L, :] = la[:L, :]
                lb_pad[c, k, :L, :] = lb[:L, :]

        return la_pad, lb_pad, max_len

    def get_chain_probabilities(self, use_gpu=True):
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

        if use_gpu:
            try:
                # Pack lists -> padded tensors for GPU
                la_pad, lb_pad, max_len = self._pack_logs_to_padded()

                # Move to device
                la_d = cuda.to_device(la_pad)
                lb_d = cuda.to_device(lb_pad)
                lens_d = cuda.to_device(np.array(self.chain_lengths, dtype=np.int32))
                out_d = cuda.device_array_like(la_d)  # same shape

                # Launch config: 3D grid over (c,k,l)
                threadsperblock = (4, 8, 16)  # tweak as needed
                blockspergrid = (
                    (self.num_chr + threadsperblock[0] - 1) // threadsperblock[0],
                    (self.num_clusters + threadsperblock[1] - 1) // threadsperblock[1],
                    (max_len + threadsperblock[2] - 1) // threadsperblock[2],
                )

                chain_probs_kernel[blockspergrid, threadsperblock](
                    la_d, lb_d, lens_d, out_d,
                    self.num_chr, self.num_clusters, max_len
                )

                out = out_d.copy_to_host()

                # Unpack back into the expected list-of-arrays shape
                for c in range(self.num_chr):
                    L = self.chain_lengths[c]
                    probs_V[c][:, :L, :] = out[c, :, :L, :]

                self.probs_V = probs_V
                return
            except Exception as e:
                print(f"[GPU Fallback] Chain probs GPU failed: {e}. Falling back to CPU.")

        # === CPU fallback (original implementation) ===
        for c in tqdm(range(self.num_chr), desc='chain probs'):
            chain_length = len(self.V[c][0])

            for k in range(self.num_clusters):
                # Compute denominator
                log_prob_joint = scipy.special.logsumexp(self.log_alphas_V[c][k][chain_length - 1])

                if np.isnan(log_prob_joint):
                    print('log_prob_joint is nan', self.log_alphas_V[c][k][chain_length - 1])

                for l in range(chain_length):
                    # vectorized over t=0..2
                    log_vec = self.log_alphas_V[c][k][l] + self.log_betas_V[c][k][l] - log_prob_joint
                    if np.any(np.isnan(log_vec)):
                        print('log_prob is nan', l, self.log_alphas_V[c][k][l], self.log_betas_V[c][k][l])

                    row = np.exp(log_vec)
                    s = row.sum()
                    if s <= 0 or np.isnan(s) or np.isinf(s):
                        probs_V[c][k][l] = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
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
        for t in tqdm(range(self.num_states), desc='Thetas'):
            numerators[t] += self.alphas[t] - 1
            denominators[t] += self.alphas[t] + self.betas[t] - 2

            args = [[t, k, self.num_states, self.num_chr, self.num_cells,
                     self.chain_lengths, self.z_probs, self.probs_V, self.A, self.D] for k in range(self.num_clusters)]
            terms = list(tqdm(self.pool.imap(get_thetas, args),
                              total=self.num_clusters, desc=f'Thetas for state {t}', disable=True))

            for term in terms:
                numerators += term[0]
                denominators += term[1]

            print(f'State {t} has numerator {numerators} denominator {denominators}')

        # Set thetas and make sure they are in (0, 1)
        self.thetas = np.array([np.clip(numerators[t] / denominators[t], 0.001, 0.999) for t in range(self.num_states)])


    def update_T(self):
        """
        Updates the transition matrix.
        Args:
            self:
        Returns:
        """
        for c in range(self.num_chr):
            # Compute entries for each state pair
            for t1 in range(self.num_states):
                for t2 in range(self.num_states):
                    return

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

        # for i in range(self.num_cells):
        #     k = self.Z[i]
        #
        #     for c in range(self.num_chr):
        #         for l in range(self.chain_lengths[c]):
        #             state = self.V[c][k][l]
        #             log_joint += binom_logpmf(self.A[c][i][l], self.D[c][i][l], self.thetas[state])

        return log_joint

    def solve(self, tol, iters=25, num_inits=15, inner_iters=3):
        """
        Performs EM iterations until convergence.
        Returns:
        """
        prev_prob = np.inf
        self.init_pool()

        # Try different initializations to get the best one
        best_prob = -np.inf
        best_Z = self.Z
        best_chain = self.V

        for _ in tqdm(range(num_inits), disable=True):
            print(f'Initialization iteration {_}')

            # If we already set Z, that is intentional and we don't want to change it
            if self.Z_init is None:
                self.Z = np.random.choice(self.num_clusters,
                                          size=self.num_cells,
                                          replace=True,
                                          p=self.kappas)
            self.__init_chains__()
            self.pi = self.pi_init

            for i in range(inner_iters):
                self.update()

            new_model_prob = self.get_model_prob()
            print(f'Initialization #{_} has probability {new_model_prob:.4f}')

            if new_model_prob > best_prob:
                best_Z = self.Z.copy()
                best_chain = [v.copy() for v in self.V]
                best_prob = new_model_prob

        if num_inits > 0:
            self.Z = best_Z.copy()
            self.V = best_chain.copy()

        for i in range(iters):
            # Do EM iterations
            t0 = time.time()
            self.update()
            model_prob = self.get_model_prob()
            t1 = time.time()
            time_per_iteration = t1 - t0

            print(f'log fit probability {model_prob}', f'Took {time_per_iteration} s')
            if np.abs(model_prob - prev_prob) < tol:
                print('Converged!')
                break

            prev_prob = model_prob

        self.model_prob = model_prob
        self.close_pool()

    def _nan_finder_(self, mat, shape):
        """
        Searches the given matrix for any nan values.
        """
        # Make a flat coordinate grid for traversing the matrix
        total_entries = np.prod(shape)
        dims = len(shape)
        coordinate_arr = np.zeros(shape=(total_entries, dims), dtype=int)
        
        # Generate coordinates for each element in the matrix
        for i in range(total_entries):
            index = i
            for j, dim in enumerate(reversed(shape)):
                coordinate_arr[i][dims - 1 - j] = index % dim
                index //= dim

        # Now traverse the coordinate list and return any coordinates with a nan
        nan_coords = []
        for coords in coordinate_arr:
            if np.isnan(mat[tuple(coords)]):
                nan_coords.append(coords)

        return nan_coords