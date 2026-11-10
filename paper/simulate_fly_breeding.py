if __name__ == '__main__':
    import numpy as np
    import argparse
    import json
    import pandas as pd
    import os
    import simulation_utils as utils
    import pickle
    import anndata
    import gzip
    from demuxHMM.demuxHMM_utils import adata_to_HMM, consensus_clustering_jaccard, filter_doublets
    from sklearn.metrics import precision_score, recall_score, average_precision_score
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Script Parameters")

    # Add arguments with default values specified directly
    parser.add_argument('--VCF_df_path', type=str, default='data_dir/adata_genes_map.csv', help='Path to the VCF dataframe.')
    parser.add_argument('--VCF_FILENAME', type=str, default='data_dir/KSA2_snps_on_ref_Wild5B_filtered.vcf', help='Path to the VCF file.')
    parser.add_argument('--scrnaseq_adata', type=str, default='data_dir/head_body_lifted_genes.h5ad', help='Path to the adata file containing scRNA-seq reads.')
    parser.add_argument('--data_path', type=str, default=None, help='Path to a previously generated datasets.')
    parser.add_argument('--transition_save_path', type=str, default=None, help='Path to where to save a generated dataset.')
    parser.add_argument('--cutoff_UMI', type=int, default=1500, help='Minimum UMI for cells in our sc dataset.')
    parser.add_argument('--valid_contigs', type=str, nargs='+', default=['CM010569.1', 'CM010570.1', 'CM010571.1', 'CM010572.1', 'CM010573.1', 'CM010574.1'], help='List of valid contigs.') # Chromosome 4 is mostly conserved and does more harm than good
    parser.add_argument('--num_generations', type=int, default=2, help='Number of generations in the breeding experiment.')
    parser.add_argument('--offspring_per_generation', type=int, default=100, help='Number of offspring per generation.')
    parser.add_argument('--num_clusters', type=int, default=None, help='Number clusters to use (otherwise uses the actual number of individuals)')
    parser.add_argument('--num_cells_per_org', type=int, default=10, help='Number of cells per organism.')
    parser.add_argument('--avg_UMI', type=int, default=1500, help='Average UMI.')
    parser.add_argument('--variance_UMI', type=int, default=0, help='Variance in UMI.')
    parser.add_argument('--num_threads', type=int, default=80, help='Number of threads for the HMM model.')
    parser.add_argument('--TOL', type=int, default=10, help='Tolerance value for the HMM model.')
    parser.add_argument('--num_repeats', type=int, default=1, help='Number of repeats for the HMM model.')
    parser.add_argument('--save_dir', type=str, default='sweeps/temp/', help='Directory to save results.')
    parser.add_argument('--run_num', type=int, default=0, help='Run number.')
    parser.add_argument('--males_cross_over', type=bool, default=False, help='Whether or not to use Males cross over.')
    parser.add_argument('--vireo', action="store_true", default=False, help='Whether or not to use vireoSNP.')
    parser.add_argument('--scsplit', action="store_true", default=False, help='Whether or not to use scSplit.')
    parser.add_argument('--souporcell', action="store_true", default=False, help='Whether or not to use souporcell.')
    parser.add_argument('--demuxHMM', action="store_true", default=False, help='Whether or not to use demuxHMM.')
    parser.add_argument("--snp_usage_percent", type=float, default=1, help="Percentage of SNPs to use.")
    parser.add_argument('--doublet_rate', type=float, default=0, help='The doublet rate to use. Float in (0, 1)')
    parser.add_argument('--detect_doublets', action="store_true", default=False, help='Whether or not to detect doublets.')
    parser.add_argument('--doublet_cutoff', type=float, default=0.9, help='If doublet probability is > the cutoff, the cell is a doublet.' )
    parser.add_argument("--snp_shuffle_percent", type=float, default=0, help="Percentage of SNPs to use.")
    parser.add_argument('--no_report_genotype_error', action="store_true", default=False, help='Whether or not to report genotype error.')

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
        snp_dfs_chrom[chrom].reset_index(drop=True, inplace=True)  # reset indices to 0...n-1

    # Either load or generate the desired dataset
    if params['data_path'] is not None:
        print('Loading data...')
        try:
            adata_sim = anndata.read_h5ad(params['data_path'])

            # Assume the ground truth organisms are pickled in the same directory
            base, _ = os.path.splitext(params['data_path'])
            organisms_path = base + '_organisms.pkl.gz'
            with gzip.open(organisms_path, 'rb') as f:
                organisms = pickle.load(f)

            print('Loaded data')
        except (FileNotFoundError, pickle.UnpicklingError, KeyError):
            # The data doesn't exist yet. Generate and save.
            adata_sim, organisms = utils.simulate_dataset(params, snp_dfs_chrom, adata)
            organisms = [utils.organism_to_dict(organism) for organism in organisms]

            directory = os.path.dirname(params['data_path'])
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            adata_sim.write_h5ad(params['data_path'], compression='gzip')

            base, _ = os.path.splitext(params['data_path'])
            organisms_path = base + '_organisms.pkl.gz'
            with gzip.open(organisms_path, 'wb') as f:
                pickle.dump(organisms, f)
    else:
       adata_sim, organisms = utils.simulate_dataset(params, snp_dfs_chrom, adata)
       organisms = [utils.organism_to_dict(organism) for organism in organisms]

    # Save some memory
    del adata

    if params['snp_shuffle_percent'] > 0:
        # Shuffle SNPs onto a fake chromosome Z
        adata_sim_original = adata_sim.copy()
        adata_sim = utils.shuffle_snps_to_Z(adata_sim, params['snp_shuffle_percent'])
        A_orig, D_orig, gt_orig = adata_to_HMM(adata_sim_original, valid_chromosomes)

        # Remove chromosomes which no longer contain SNPs
        chroms_with_snps = set(adata_sim.var['chromosome'])
        print(chroms_with_snps)
        valid_chromosomes = [c for c in valid_chromosomes if c in chroms_with_snps]
        print(valid_chromosomes)
        if 'Z' in chroms_with_snps and 'Z' not in valid_chromosomes:
            valid_chromosomes.append('Z')

    # Convert to HMM friendly data format and downsample snps if needed
    A, D, ground_truth = adata_to_HMM(adata_sim, valid_chromosomes)
    A, D = utils.downsample_snps_even(A, D, params)

    if params['snp_shuffle_percent'] == 0:
        utils.print_stats_on_sim_data(A, D, valid_chromosomes, organisms)

    num_organisms = len(organisms)

    if params['transition_save_path'] is not None:
        mean_transitions = np.zeros(num_chr)

        # Count mean transitions
        for i, chr_name in enumerate(valid_chromosomes):
            for organism in organisms:
                chromosome = organism['chromosomes'][chr_name]
                num_transitions1, num_transitions2 = utils.count_transitions(chromosome)

                # Adjust for over-expression
                if organism.gender == 'F' and chr_name == 'X':
                    mean_transitions[i] += (num_transitions1 + num_transitions2) / 2
                else:
                    mean_transitions[i] += num_transitions1 + num_transitions2

        mean_transitions /= num_organisms
        with open(params['transition_save_path'], 'wb') as f:
            pickle.dump(mean_transitions, f)
    else:
        mean_transitions = utils.get_transitions(params['num_generations'])
        print(f'Saw {mean_transitions} transitions on average')

    T_init = None
    if params['snp_shuffle_percent'] > 0:
        num_chr = len(valid_chromosomes)
        num_states = 3
        T_eps = 1e-10
        T_init = np.zeros(shape=(num_chr, num_states, num_states))

        for c in range(num_chr - 1):
            orig_c = ['X', '2', '3'].index(valid_chromosomes[c])
            p = mean_transitions[orig_c] / A_orig[orig_c].shape[1]
            T_init[c, :, :] = np.array([[1-p, p, 0],
                                  [p/2, 1-p, p/2],
                                  [0, p, 1-p]])
            T_init[c, :, :] = (T_init[c, :, :] + T_eps) / (1 + 3*T_eps)

        T_init[-1, :, :] = np.array([[1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3]])

    # Filter out cells with no reads in one of the chromosomes
    # For the HMM model to run you technically only have to filter out the ones with D reads = 0, but performance
    # is better this way for all models, so this represents a reasonable filtering step a group could take
    total_cells = A[0].shape[0]
    mask = np.full(total_cells, True)

    if params['snp_shuffle_percent'] > 0:
        for Ac, Dc in zip(A_orig, D_orig):
            count_sums_D = np.asarray(Dc.sum(axis=1)).ravel()
            count_sums_A = np.asarray(Ac.sum(axis=1)).ravel()
            mask &= count_sums_D != 0
            mask &= count_sums_A != 0
    else:
        for Ac, Dc in zip(A, D):
            count_sums_D = np.asarray(Dc.sum(axis=1)).ravel()
            count_sums_A = np.asarray(Ac.sum(axis=1)).ravel()
            mask &= count_sums_D != 0
            mask &= count_sums_A != 0

    A_filtered = [Ac[mask] for Ac in A]
    D_filtered = [Dc[mask] for Dc in D]
    ground_truth_filtered = ground_truth[mask]
    doublet_gt_filtered = adata_sim.obs['is_doublet'][mask]
    doublet_gt_filtered = np.asarray(doublet_gt_filtered)

    emb_remaining = len(np.unique(ground_truth_filtered))
    cells_remaining = np.sum(mask)
    print('Unique emb remaining', emb_remaining, 'Original', num_organisms)
    print(f'The number of total cells is {total_cells}, the number of filtered cells is {cells_remaining}')

    # Store results in these
    results_dict = {}
    assignments_dict = {}

    # Make the save dir if needed
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    if params['num_clusters'] is not None:
        num_clusters = params['num_clusters']
    else:
        num_clusters = num_organisms

    # Adjust paralellism for high embryo counts
    if num_clusters > 500:
        num_processes = 10
    else:
        num_processes = 20

    if params['demuxHMM']:
        theta_init = np.array([0.006, 0.469, 0.988])
        pi_init = None

        # First pass
        (assignments_hmm, ari_score_hmm,
         time_hmm, model, all_assignments) = utils.run_demuxHMM(A_filtered, D_filtered, num_clusters,
                                                                params['TOL'], ground_truth_filtered,
                                                                mean_transitions=mean_transitions,
                                                                T_init=T_init,
                                                                num_threads=params['num_threads'],
                                                                num_repeats=params['num_repeats'],
                                                                theta_init=theta_init,
                                                                pi_init=pi_init,
                                                                detect_doublets=params['detect_doublets'],
                                                                doublet_prior=params['doublet_rate'],
                                                                num_processes=num_processes)

        if params['detect_doublets']:
            hmm_doublets = model['doublet_probs'] > params['doublet_cutoff']

            precision = precision_score(doublet_gt_filtered, hmm_doublets)
            recall = recall_score(doublet_gt_filtered, hmm_doublets)
            aupr = average_precision_score(doublet_gt_filtered, model['doublet_probs'])
            tnr = recall_score(doublet_gt_filtered, hmm_doublets, pos_label=0)

            results_dict['hmm_doublet_precision'] = precision
            results_dict['hmm_doublet_recall'] = recall
            results_dict['hmm_aupr'] = aupr
            results_dict['hmm_tnr'] = tnr

            assignments_dict['hmm_doublet_gt'] = doublet_gt_filtered
            assignments_dict['hmm_doublet_predicted'] = hmm_doublets

            print('Doublet precision: ', precision)
            print('Doublet recall: ', recall)
            print('AUPR: ', aupr)
            print('TNR: ', tnr)

            # First-pass consensus clustering
            time_start = datetime.now()
            consensus_assignments, aligned_assignments = consensus_clustering_jaccard(all_assignments,
                                                                                      assignments_hmm)
            time_consensus = (datetime.now() - time_start).total_seconds()

            ari_consensus = utils.adjusted_rand_score(ground_truth_filtered, consensus_assignments)
            results_dict['hmm_ari_consensus'] = [ari_consensus]
            results_dict['hmm_time'] = [time_hmm + time_consensus]

            assignments_dict['assignments_hmm_consensus'] = consensus_assignments

            # Second pass: remove predicted doublets
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_filtered, D_filtered,
                                                                ground_truth_filtered, hmm_doublets)

            (assignments_hmm_filt, ari_score_hmm_filt,
             time_hmm_filt, model_filt,
             all_assignments_filt) = utils.run_demuxHMM(A_filt, D_filt, num_clusters,
                                                        params['TOL'], gt_filt,
                                                        mean_transitions=mean_transitions,
                                                        T_init=T_init,
                                                        num_threads=params['num_threads'],
                                                        num_repeats=params['num_repeats'],
                                                        theta_init=theta_init,
                                                        pi_init=pi_init,
                                                        detect_doublets=False,
                                                        doublet_prior=0,
                                                        num_processes=num_processes)

            # Second-pass consensus clustering
            time_start = datetime.now()
            consensus_assignments_filt, aligned_assignments_filt = consensus_clustering_jaccard(all_assignments_filt,
                                                                                                assignments_hmm_filt)
            time_consensus_filt = (datetime.now() - time_start).total_seconds()

            ari_consensus_filt = utils.adjusted_rand_score(gt_filt, consensus_assignments_filt)
            results_dict['hmm_ari_doublets_removed'] = [ari_score_hmm_filt]
            results_dict['hmm_ari_consensus_doublets_removed'] = [ari_consensus_filt]
            results_dict['hmm_time_consensus_doublets_removed'] = [time_hmm + time_hmm_filt +
                                                                   time_consensus + time_consensus_filt]
            results_dict['hmm_num_doublets_removed'] = [np.sum(hmm_doublets)]

            print('Two pass ARI: ', ari_score_hmm_filt)
            print('Two pass consensus ARI: ', ari_consensus_filt)

            # Restore second-pass assignments to original cell indexing
            assignments_hmm_dedub = np.full(A_filtered[0].shape[0], np.nan)
            assignments_hmm_dedub[~hmm_doublets] = assignments_hmm_filt
            assignments_dict['assignments_hmm_dedub'] = assignments_hmm_dedub

            assignments_hmm_consensus_dedub = np.full(A_filtered[0].shape[0], np.nan)
            assignments_hmm_consensus_dedub[~hmm_doublets] = consensus_assignments_filt
            assignments_dict['assignments_hmm_consensus_dedub'] = assignments_hmm_consensus_dedub

        else:
            # No doublet detection. First-pass consensus only.
            time_start = datetime.now()
            consensus_assignments, aligned_assignments = consensus_clustering_jaccard(all_assignments, assignments_hmm)
            time_consensus = (datetime.now() - time_start).total_seconds()
            ari_consensus = utils.adjusted_rand_score(ground_truth_filtered, consensus_assignments)

            results_dict['hmm_time'] = [time_hmm + time_consensus]
            results_dict['hmm_ari_consensus'] = [ari_consensus]

            assignments_dict['assignments_hmm_consensus'] = consensus_assignments

        print(f' HMM score is {ari_score_hmm} and took {time_hmm} seconds to run')

        results_dict['hmm_score'] = [ari_score_hmm]
        results_dict['hmm_model_prob'] = [model['probability']]
        results_dict['emb_remaining'] = emb_remaining
        results_dict['cells_remaining'] = cells_remaining
        assignments_dict['assignments_hmm'] = assignments_hmm
        assignments_dict['gt_hmm'] = ground_truth_filtered

        if not params['no_report_genotype_error']:
            # Match HMM clusters to ground truth organisms
            cluster_map = utils.best_cluster_label_mapping(consensus_assignments, ground_truth_filtered)

            # Compute genotype error using that mapping
            genotype_errors = utils.genotype_error_across_organisms(organisms, model['V'],
                                                                    valid_chromosomes, cluster_map)
            results_dict['hmm_genotype_error'] = genotype_errors['mean_error']

    if params['scsplit']:
        # First pass
        (assignments_scsplit, doublets_scsplit,
         gt_filt_scsplit, ari_score_scsplit,
         time_scsplit, indices) = utils.run_scsplit(A_filtered, D_filtered,
                                                    valid_chromosomes, 'temp_dir',
                                                    num_organisms,
                                                    true_labels=ground_truth_filtered,
                                                    detect_doublets=params['detect_doublets'],
                                                    doublet_prior=params['doublet_rate'])

        print(f'The scSplit ARI is {ari_score_scsplit} \n Took {time_scsplit} seconds')

        if params['detect_doublets']:
            indices = np.asarray(indices)

            # Compute doublet detection metrics
            doublet_gt_scsplit_order = doublet_gt_filtered[indices]
            precision = precision_score(doublet_gt_scsplit_order, doublets_scsplit)
            recall = recall_score(doublet_gt_scsplit_order, doublets_scsplit)
            tnr = recall_score(doublet_gt_scsplit_order, doublets_scsplit, pos_label=0)

            # Restore first-pass results to A_filtered indexing
            assignments_scsplit_full = np.full(A_filtered[0].shape[0], np.nan)
            gt_scsplit_full = np.full(A_filtered[0].shape[0], np.nan)
            scsplit_doublets_full = np.full(A_filtered[0].shape[0], False)

            assignments_scsplit_full[indices] = assignments_scsplit
            gt_scsplit_full[indices] = gt_filt_scsplit
            scsplit_doublets_full[indices] = doublets_scsplit

            assignments_dict_scsplit = {
                'assignments_scsplit': assignments_scsplit_full,
                'gt_scsplit': gt_scsplit_full,
                'scSplit_doublet_gt': doublet_gt_filtered,
                'scSplit_doublet_predicted': scsplit_doublets_full
            }

            results_dict['scSplit_doublet_precision'] = precision
            results_dict['scSplit_doublet_recall'] = recall
            results_dict['scSplit_doublet_tnr'] = tnr
            print('Doublet precision: ', precision)
            print('Doublet recall: ', recall)
            print('TNR: ', tnr)

            # Second pass: remove predicted doublets
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_filtered, D_filtered,
                                                                ground_truth_filtered, scsplit_doublets_full)

            (assignments_scsplit_filt, doublets_scsplit_filt,
             gt_filt_scsplit_filt, ari_score_scsplit_filt,
             time_scsplit_filt, indices_filt) = utils.run_scsplit(A_filt, D_filt,
                                                                    valid_chromosomes, 'temp_dir',
                                                                    num_organisms,
                                                                    true_labels=gt_filt,
                                                                    detect_doublets=False,
                                                                    doublet_prior=0)

            print('Two pass ARI: ', ari_score_scsplit_filt)

            results_dict['scSplit_ari_doublets_removed'] = [ari_score_scsplit_filt]
            results_dict['scSplit_time_doublets_removed'] = [time_scsplit + time_scsplit_filt]
            results_dict['scSplit_num_doublets_removed'] = [np.sum(scsplit_doublets_full)]

            # Restore second-pass assignments to A_filtered indexing
            assignments_scsplit_dedub = np.full(A_filtered[0].shape[0], np.nan)
            original_indices = np.asarray(keep_idx)[np.asarray(indices_filt)]
            assignments_scsplit_dedub[original_indices] = assignments_scsplit_filt
            assignments_dict_scsplit['assignments_scsplit_dedub'] = assignments_scsplit_dedub
        else:
            # Restore first-pass results to A_filtered indexing
            assignments_scsplit_full = np.full(A_filtered[0].shape[0], np.nan)
            gt_scsplit_full = np.full(A_filtered[0].shape[0], np.nan)

            indices = np.asarray(indices)
            assignments_scsplit_full[indices] = assignments_scsplit
            gt_scsplit_full[indices] = gt_filt_scsplit

            assignments_dict_scsplit = {
                'assignments_scsplit': assignments_scsplit_full,
                'gt_scsplit': gt_scsplit_full
            }

        results_dict['scSplit_ARI'] = [ari_score_scsplit]
        results_dict['scSplit_time'] = [time_scsplit]

        scsplit_df = pd.DataFrame(assignments_dict_scsplit)
        scsplit_df.to_csv(f'{params["save_dir"]}assignments_scsplit_run_{params["run_num"]}.csv')

    if params['souporcell']:
        # First pass
        (sorc_true_labels, sorc_pred_labels,
         ari_score_souporcell, time_souporcell, indices_fit,
         doublets_sorc, doublet_probs, indices_doublet) = utils.run_souporcell(A_filtered, D_filtered,
                                                                        valid_chromosomes, 'temp_dir',
                                                                        num_organisms,
                                                                        true_labels=ground_truth_filtered,
                                                                        threads=params['num_threads'],
                                                                        detect_doublets=params['detect_doublets'],
                                                                        doublet_prior=params['doublet_rate'])

        print(f'The souporcell ARI is {ari_score_souporcell} \n Took {time_souporcell} seconds')

        # Restore first-pass assignments to A_filtered indexing
        # (this shouldn't be necessary as they are in order in all scenarios I've seen)
        indices_fit = np.asarray(indices_fit)
        assignments_souporcell_full = np.full(A_filtered[0].shape[0], np.nan)
        assignments_souporcell_full[indices_fit] = sorc_pred_labels

        results_dict['souporcell_ARI'] = [ari_score_souporcell]
        results_dict['souporcell_time'] = [time_souporcell]
        assignments_dict['assignments_souporcell'] = assignments_souporcell_full
        assignments_dict['gt_souporcell'] = sorc_true_labels

        if params['detect_doublets']:
            indices_doublet = np.asarray(indices_doublet)

            # Compute doublet detection stats
            doublet_gt_sorc_order = doublet_gt_filtered[indices_doublet]
            precision = precision_score(doublet_gt_sorc_order, doublets_sorc)
            recall = recall_score(doublet_gt_sorc_order, doublets_sorc)
            aupr = average_precision_score(doublet_gt_sorc_order, doublet_probs)
            tnr = recall_score(doublet_gt_sorc_order, doublets_sorc, pos_label=0)

            results_dict['souporcell_doublet_precision'] = precision
            results_dict['souporcell_doublet_recall'] = recall
            results_dict['souporcell_aupr'] = aupr
            results_dict['souporcell_tnr'] = tnr
            print('Doublet precision: ', precision)
            print('Doublet recall: ', recall)
            print('AUPR: ', aupr)
            print('TNR: ', tnr)

            # Restore doublet predictions to A_filtered indexing (also should be unnecessary)
            souporcell_doublets_full = np.zeros(A_filtered[0].shape[0], dtype=bool)
            souporcell_doublets_full[indices_doublet] = doublets_sorc

            assignments_dict['souporcell_doublet_gt'] = doublet_gt_filtered
            assignments_dict['souporcell_doublet_predicted'] = souporcell_doublets_full

            # Second pass: remove predicted doublets
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_filtered, D_filtered,
                                                                ground_truth_filtered, souporcell_doublets_full)

            # Rerun Souporcell without doublet detection
            (sorc_true_labels_filt, sorc_pred_labels_filt,
             ari_score_souporcell_filt, time_souporcell_filt,
             indices_fit_filt, _, _, _) = utils.run_souporcell(A_filt, D_filt,
                                                                valid_chromosomes, 'temp_dir',
                                                                num_organisms,
                                                                true_labels=gt_filt,
                                                                threads=params['num_threads'],
                                                                detect_doublets=False,
                                                                doublet_prior=0)

            print('Two pass ARI: ', ari_score_souporcell_filt)

            results_dict['souporcell_ari_doublets_removed'] = [ari_score_souporcell_filt]
            results_dict['souporcell_time_doublets_removed'] = [time_souporcell + time_souporcell_filt]
            results_dict['souporcell_num_doublets_removed'] = [np.sum(souporcell_doublets_full)]

            # Restore second-pass assignments to original cell indexing
            assignments_souporcell_dedub = np.full(A_filtered[0].shape[0], np.nan)
            original_indices = np.asarray(keep_idx)[np.asarray(indices_fit_filt)]
            assignments_souporcell_dedub[original_indices] = sorc_pred_labels_filt
            assignments_dict['assignments_souporcell_dedub'] = assignments_souporcell_dedub

    if params['vireo']:
        # First pass
        (assignments_vireo, ari_score_vireo,
         time_vireo, doublet_prob_vireo, vireo_model) = utils.run_vireo(A_filtered, D_filtered,
                                                            num_organisms,
                                                            ground_truth_filtered,
                                                            detect_doublets=params['detect_doublets'],
                                                            doublet_prior=params['doublet_rate'],
                                                            num_threads=params['num_threads'])

        print(f'The vireo ARI score is {ari_score_vireo} \n Took {time_vireo} seconds')

        if params['detect_doublets']:
            vireo_doublets = doublet_prob_vireo > params['doublet_cutoff']

            precision = precision_score(doublet_gt_filtered, vireo_doublets)
            recall = recall_score(doublet_gt_filtered, vireo_doublets)
            aupr = average_precision_score(doublet_gt_filtered, doublet_prob_vireo)
            tnr = recall_score(doublet_gt_filtered, vireo_doublets, pos_label=0)

            results_dict['vireo_doublet_precision'] = precision
            results_dict['vireo_doublet_recall'] = recall
            results_dict['vireo_aupr'] = aupr
            results_dict['vireo_tnr'] = tnr

            assignments_dict['vireo_doublet_gt'] = doublet_gt_filtered
            assignments_dict['vireo_doublet_predicted'] = vireo_doublets

            print('Doublet precision: ', precision)
            print('Doublet recall: ', recall)
            print('AUPR: ', aupr)
            print('TNR: ', tnr)

            # Second pass: remove predicted doublets
            A_filt, D_filt, gt_filt, keep_idx = filter_doublets(A_filtered, D_filtered,
                                                                ground_truth_filtered, vireo_doublets)

            (assignments_vireo_filt, ari_score_vireo_filt,
             time_vireo_filt, _, vireo_model2) = utils.run_vireo(A_filt, D_filt,
                                                    num_organisms,
                                                    gt_filt,
                                                    detect_doublets=False,
                                                    doublet_prior=0,
                                                    num_threads=params['num_threads'])

            print('Two pass ARI: ', ari_score_vireo_filt)

            results_dict['vireo_ari_doublets_removed'] = [ari_score_vireo_filt]
            results_dict['vireo_time_doublets_removed'] = [time_vireo + time_vireo_filt]
            results_dict['vireo_num_doublets_removed'] = [np.sum(vireo_doublets)]

            # Restore second-pass assignments to original cell indexing
            assignments_vireo_dedub = np.full(A_filtered[0].shape[0], np.nan)
            assignments_vireo_dedub[~vireo_doublets] = assignments_vireo_filt
            assignments_dict['assignments_vireo_dedub'] = assignments_vireo_dedub

        if not params['no_report_genotype_error']:
            # Match clusters to ground truth organisms
            cluster_map = utils.best_cluster_label_mapping(assignments_vireo, ground_truth_filtered)

            # Vireo may permute categories sometimes (Ex. 1 may not always mean alt/ref).
            # ARI not affected, but genotypes need to be fixed. We can identify using beta mu
            # homo ref = 0, hetero = 0.5, homo alt = 1 (in the ideal case, so we can use the ordering of beta mu)
            beta_mu = np.asarray(vireo_model.beta_mu).squeeze()
            category_to_dosage = np.argsort(np.argsort(beta_mu))

            raw_categories = np.argmax(vireo_model.GT_prob, axis=2)
            predicted_genotypes = category_to_dosage[raw_categories].T
            predicted_genotypes = utils.vireo_genotype_mat_to_hmm(predicted_genotypes, A_filtered)

            # Compute genotype error using that mapping
            genotype_errors = utils.genotype_error_across_organisms(organisms, predicted_genotypes,
                                                                    valid_chromosomes, cluster_map)
            results_dict['vireo_genotype_error'] = genotype_errors['mean_error']
            print(genotype_errors['mean_error'])

        results_dict['vireo_ARI'] = [ari_score_vireo]
        results_dict['vireo_time'] = [time_vireo]
        assignments_dict['assignments_vireo'] = assignments_vireo
        assignments_dict['gt_vireo'] = ground_truth_filtered

    # Save params and results
    with open(f"{params['save_dir']}params_run_{params['run_num']}.json", 'w') as f:
        json.dump(params, f)

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'{params["save_dir"]}results_run_{params["run_num"]}.csv')

    assignments_df = pd.DataFrame(assignments_dict)
    assignments_df.to_csv(f'{params["save_dir"]}assignments_run_{params["run_num"]}.csv')
