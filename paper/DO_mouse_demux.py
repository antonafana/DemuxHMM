if __name__ == '__main__':
    import numpy as np
    import anndata
    import argparse
    from demuxHMM.demuxHMM_utils import adata_to_HMM, filter_doublets, consensus_clustering_jaccard
    from sklearn.metrics import homogeneity_score
    import simulation_utils
    import pandas as pd
    import json
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run HMM on DO mouse data.")
    parser.add_argument('--demuxHMM',action="store_true",
                        default=False, help='Whether or not to use demuxHMM.')
    parser.add_argument('--vireo',action="store_true",
                        default=False, help='Whether or not to use vireoSNP.')
    parser.add_argument('--souporcell',action="store_true",
                        default=False, help='Whether or not to use souporcell.')
    parser.add_argument('--run_num', type=int, default=0, help='Run number for tracking repeats.')
    parser.add_argument('--save_dir', type=str, default='sweeps/temp/', help='Directory to save results.')
    parser.add_argument('--data_path', type=str, default='data_dir/DO_mouse_cellsnp_001.h5ad',
                         help='Path to a previously generated datasets.')
    parser.add_argument('--num_threads', type=int, default=77, help='Number of threads for the HMM model.')
    parser.add_argument('--TOL', type=int, default=1000000, help='Tolerance value for the HMM model.')
    parser.add_argument('--num_repeats', type=int, default=30, help='Number of repeats for the HMM model.')
    parser.add_argument('--detect_doublets',action="store_true", default=False, help='Whether or not to detect doublets.')
    parser.add_argument('--doublet_rate', type=float, default=0.05, help='The doublet rate to use. Float in (0, 1)')
    parser.add_argument('--doublet_cutoff', type=float, default=0.9, help='If doublet probability is > the cutoff, the cell is a doublet.' )
    parser.add_argument('--num_clusters', type=int, default=77, help='Number of clusters to use.')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of clusters to use.')
    parser.add_argument('--snp_shuffle_percent', type=float, default=0, help='Percentage of SNPs to shuffle.')
    parser.add_argument('--sorc_seed', type=int, default=None, help='Seed for sorc.')

    params = vars(parser.parse_args())
    print('Running with params: ', params)

    # Paths and constants
    USE_GPU = True
    NUM_ORGANISMS = 77
    EXPECTED_TRANSITIONS = 39.05
    T_EPS = 0.01

    # Load data
    adata = anndata.read_h5ad(params['data_path'])
    print(f'Loaded adata with shape {adata.shape}')

    # Get info on number of chromosomes
    chromosomes = adata.var['chrom'].unique()
    num_chromosomes = len(chromosomes)
    chain_lengths = np.array([np.sum(adata.var['chrom'] == chrom) for chrom in chromosomes])
    e_transition_per_chrom = [EXPECTED_TRANSITIONS for chrom in chromosomes]

    # Set other parameters based on previous fits
    theta_init = np.array([0.01543684, 0.44519121, 0.96256561])
    pi_init = np.array([0.73456463, 0.19245602, 0.07297934])
    print(f"Running HMM with {NUM_ORGANISMS} organisms and {num_chromosomes} chromosomes")

    T_init = None

    if params['snp_shuffle_percent'] > 0:
        # Save original adata for later use in setup
        adata_original = adata.copy()
        chromosomes_orig = chromosomes.copy()
        A_orig, D_orig = adata_to_HMM(adata_original, chromosomes_orig, chromosome_key='chrom')

        # Shuffle SNPs onto a fake chromosome Z
        adata = simulation_utils.shuffle_snps_to_Z(adata, params['snp_shuffle_percent'], chromosome_key='chrom')

        # Remove chromosomes which no longer contain SNPs
        chroms_with_snps = set(adata.var['chrom'])
        chromosomes = [c for c in chromosomes if c in chroms_with_snps]
        print(chromosomes)
        if 'Z' in chroms_with_snps and 'Z' not in chromosomes:
            chromosomes.append('Z')

        # Correct the transition matrix to adjust for the shuffling
        num_chr = len(chromosomes)
        num_states = 3
        T_init = np.zeros(shape=(num_chr, num_states, num_states))

        for c in range(num_chr - 1):
            orig_c = list(chromosomes_orig).index(chromosomes[c])
            p = EXPECTED_TRANSITIONS / A_orig[orig_c].shape[1]
            T_init[c, :, :] = np.array([[1-p, p, 0],
                                  [p/2, 1-p, p/2],
                                  [0, p, 1-p]])
            T_init[c, :, :] = (T_init[c, :, :] + T_EPS) / (1 + 3*T_EPS)

        T_init[-1, :, :] = np.array([[1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3]])

        pi_init = None # A set pi_init no longer makes sense here

    # demuxHMM Input Matrices
    A_sparse, D_sparse = adata_to_HMM(adata, chromosomes, chromosome_key='chrom')
    print('A_sparse shape', len(A_sparse), 'D_sparse shape', len(D_sparse))

    # Make the save dir if needed
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    # Save params and results
    with open(f"{params['save_dir']}params_run_{params['run_num']}.json", 'w') as f:
        json.dump(params, f)

    results_dict = {}
    assignments_dict = {}

    # Run vireo
    if params['vireo']:
        (assignments_vireo, ari_score_vireo,
         time_vireo, prob_doublet, model) = simulation_utils.run_vireo(A_sparse, D_sparse,
                                                                params['num_clusters'], None,
                                                                num_threads=params['num_threads'],
                                                                detect_doublets=params['detect_doublets'],
                                                                doublet_prior=params['doublet_rate'])

        if params['detect_doublets']:
            vireo_doublets = prob_doublet > params['doublet_cutoff']
            assignments_dict['vireo_doublet_predicted'] = vireo_doublets
            print('Predicted # doublets:', np.sum(vireo_doublets))

            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_sparse, D_sparse,
                                                                np.array(adata.obs['pool']),
                                                                vireo_doublets)

            (assignments_vireo_filt, ari_score_vireo_filt,
             time_vireo_filt, prob_doublet_filt, model2) = simulation_utils.run_vireo(A_filt, D_filt,
                                                                    params['num_clusters'], None,
                                                                    num_threads=params['num_threads'],
                                                                    detect_doublets=False,
                                                                    doublet_prior=0)

            homog_score_filt = homogeneity_score(gt_filt, assignments_vireo_filt)
            print('Two pass homog:', homog_score_filt)

            results_dict['vireo_homog_doublets_removed'] = homog_score_filt
            results_dict['vireo_time_doublets_removed'] = time_vireo + time_vireo_filt
            results_dict['vireo_num_doublets_removed'] = np.sum(vireo_doublets)

            # Save doublet removed assignments
            assignments_vireo_dedub = np.full(len(adata), np.nan)
            assignments_vireo_dedub[keep_idx] = assignments_vireo_filt
            assignments_dict['assignments_vireo_dedub'] = assignments_vireo_dedub

        # Evaluate
        homog_score_vireo = homogeneity_score(adata.obs['pool'], assignments_vireo)
        print(f'Homogeneity vireo: {homog_score_vireo:.4f}')

        results_dict['vireo_homog_score'] = homog_score_vireo
        results_dict['vireo_time'] = time_vireo
        assignments_dict['assignments_vireo'] = assignments_vireo
        assignments_dict['gt_vireo'] = adata.obs['pool']

    # Run HMM
    if params['demuxHMM']:
        (assignments_hmm, ari_score_hmm,
         time_hmm, model_out, all_assignments) = simulation_utils.run_demuxHMM(A_sparse, D_sparse, params['num_clusters'],
                                                             params['TOL'], None,
                                                             mean_transitions=e_transition_per_chrom,
                                                             num_threads=params['num_threads'],
                                                             num_repeats=params['num_repeats'],
                                                             theta_init=theta_init,
                                                             pi_init=pi_init,
                                                             detect_doublets=params['detect_doublets'],
                                                             doublet_prior=params['doublet_rate'],
                                                             T_eps=T_EPS,
                                                             num_processes=params['num_processes'],
                                                             T_init=T_init,
                                                             do_theta_update=True)

        time_start = datetime.now()
        consensus_assignments_first, aligned_assignments_first = consensus_clustering_jaccard(all_assignments,
                                                                                            assignments_hmm)
        time_first_consensus = (datetime.now() - time_start).total_seconds()

        if params['detect_doublets']:
            hmm_doublets = model_out['doublet_probs'] > params['doublet_cutoff']
            assignments_dict['hmm_doublet_predicted'] = hmm_doublets
            prev_best_minus_doublets = consensus_assignments_first[~hmm_doublets]

            print('Predicted # doublets:', np.sum(hmm_doublets))
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_sparse, D_sparse,
                                                                np.array(adata.obs['pool']), hmm_doublets)

            (assignments_hmm_filt, ari_score_hmm_filt,
             time_hmm_filt, model_filt,
             all_assignments_filt) = simulation_utils.run_demuxHMM(A_filt, D_filt, params['num_clusters'],
                                                                  params['TOL'], None,
                                                                  mean_transitions=e_transition_per_chrom,
                                                                  num_threads=params['num_threads'],
                                                                  num_repeats=1,
                                                                  theta_init=theta_init,
                                                                  pi_init=pi_init,
                                                                  z_init=prev_best_minus_doublets,
                                                                  detect_doublets=False,
                                                                  doublet_prior=0,
                                                                  T_eps=T_EPS,
                                                                  num_processes=params['num_processes'],
                                                                  T_init=T_init,
                                                                  do_theta_update=True)

            homog_score_filt = homogeneity_score(gt_filt, assignments_hmm_filt)
            print('Two pass homog:', homog_score_filt)

            results_dict['hmm_homog_doublets_removed'] = homog_score_filt
            results_dict['hmm_num_doublets_removed'] = np.sum(hmm_doublets)
            results_dict['hmm_model_prob_dedub'] = model_filt['probability']

            # Save doublet removed assignments
            assignments_hmm_dedub = np.full(len(adata), np.nan)
            assignment_probs_hmm_dedub = np.full(len(adata), np.nan)
            assignments_hmm_dedub[keep_idx] = assignments_hmm_filt
            assignment_probs_hmm_dedub[keep_idx] = np.max(model_filt['z_probs'], axis=1)
            assignments_dict['assignments_hmm_dedub'] = assignments_hmm_dedub
            assignments_dict['assignment_probs_hmm_dedub'] = assignment_probs_hmm_dedub

            # Save per-cluster-likelihoods
            np.savetxt(f'{params["save_dir"]}/cluster_joints_run_{params["run_num"]}.npy',
                       model_filt['cluster_joints'])

            # Do consensus clustering
            # time_start = datetime.now()
            # consensus_assignments, aligned_assignments = consensus_clustering_jaccard(all_assignments_filt,
            #                                                                             assignments_hmm_filt)
            # assignments_consensus = np.full(len(adata), np.nan)
            # assignments_consensus[keep_idx] = consensus_assignments
            # assignments_dict['consensus_assignments_hmm_dedub'] = assignments_consensus
            # time_second_consensus = (datetime.now() - time_start).total_seconds()

            #homog_consensus = homogeneity_score(gt_filt, consensus_assignments)
            results_dict['hmm_time_doublets_removed'] = time_hmm + time_hmm_filt + time_first_consensus
            # results_dict['hmm_score_consensus_dedub'] = homog_consensus
            # print('Consensus homogeneity is', homog_consensus)

        # Evaluate
        homog_score = homogeneity_score(adata.obs['pool'], assignments_hmm)
        print(f'Model pi is {model_out["pi"]}')
        print(f'Model theta is {model_out["theta"]}')

        homog_score_cons = homogeneity_score(adata.obs['pool'], consensus_assignments_first)
        print(f'Homogeneity HMM: {homog_score_cons:.4f} with model prob {model_out["probability"]:.4f}')
        assignments_dict['consensus_assignments_hmm'] = consensus_assignments_first

        # Store results
        results_dict['hmm_score'] = homog_score
        results_dict['hmm_score_consensus'] = homog_score_cons
        results_dict['hmm_time'] = time_hmm
        results_dict['hmm_model_prob'] = model_out['probability']
        assignments_dict['assignments_hmm'] = assignments_hmm
        assignments_dict['gt_hmm'] = adata.obs['pool']
        assignments_dict['assignment_probs_hmm'] = np.max(model_out['z_probs'], axis=1)

    # Run souporcell
    if params['souporcell']:
        (sorc_true_labels, sorc_pred_labels,
         ari_score_souporcell, time_souporcell, indices_fit,
         doublets_sorc, doublet_probs, indices_doublet) = simulation_utils.run_souporcell(A_sparse, D_sparse, chromosomes,
                                                    'temp_dir', params['num_clusters'], true_labels=None,
                                                                threads=params['num_threads'],
                                                                detect_doublets=params['detect_doublets'],
                                                                doublet_prior=params['doublet_rate'],
                                                                random_seed=params['sorc_seed'])

        sorc_true_labels = adata.obs['pool'].iloc[indices_fit]
        homog_score_sorc = homogeneity_score(sorc_true_labels, sorc_pred_labels)
        print(f'Homogeneity souporcell: {homog_score_sorc:.4f}')

        results_dict['souporcell_score'] = homog_score_sorc
        results_dict['souporcell_time'] = time_souporcell
        assignments_dict['assignments_souporcell'] = sorc_pred_labels
        assignments_dict['gt_souporcell'] = sorc_true_labels

        if params['detect_doublets']:
            sorc_doublets = np.zeros(len(adata), dtype=bool)
            sorc_doublets[indices_doublet] = doublets_sorc
            assignments_dict['souporcell_doublet_predicted'] = sorc_doublets
            print('Predicted # doublets:', np.sum(sorc_doublets))
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_sparse, D_sparse,
                                                                np.array(adata.obs['pool']), sorc_doublets)

            (sorc_true_labels_filt, assignments_souporcell_filt,
             ari_score_souporcell_filt, time_souporcell_filt, indices_fit_filt,
             doublets_sorc_filt, doublet_probs_filt, indices_doublet_filt) = simulation_utils.run_souporcell(
                                                    A_filt, D_filt, chromosomes,
                                                    'temp_dir', params['num_clusters'], true_labels=None,
                                                    threads=params['num_threads'], detect_doublets=False,
                                                    doublet_prior=0,
                                                    random_seed=params['sorc_seed'])

            homog_score_filt = homogeneity_score(gt_filt, assignments_souporcell_filt)
            print('Two pass homog:', homog_score_filt)

            results_dict['souporcell_homog_doublets_removed'] = homog_score_filt
            results_dict['souporcell_time_doublets_removed'] = time_souporcell + time_souporcell_filt
            results_dict['souporcell_num_doublets_removed'] = np.sum(sorc_doublets)

            assignments_souporcell_dedub = np.full(len(adata), np.nan)
            assignments_souporcell_dedub[keep_idx] = assignments_souporcell_filt
            assignments_dict['assignments_souporcell_dedub'] = assignments_souporcell_dedub

    # Save results to disk
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(f'{params["save_dir"]}/results_run_{params["run_num"]}.csv')

    print('Results df is', results_df)

    assignments_df = pd.DataFrame(assignments_dict)
    assignments_df.to_csv(f'{params["save_dir"]}/assignments_run_{params["run_num"]}.csv')
