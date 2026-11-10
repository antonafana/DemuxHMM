import numpy as np
import multiprocessing as mp
import anndata
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse

class Chromosome:
    def __init__(self, name, length, snp_locations, states, gene_to_snp_map, recombination_rate=None,
                 expected_cross_count=1, interval_size=1):
        """
        Represents a diploid chromosome.

        Args:
            name: A string representing the chromosome name.
            length: The length of the chromosome.
            snp_locations: An ordered list of snp locations
            states: A 2 x n_snps boolean ndarray. 0 represents the wild type base, 1 the mutant type.
                    In the case of the X chromosome it may be 1 x n_snps.
            gene_to_snp_map: A dictionary mapping gene names to snp indices.
            recombination_rate: A float representing the rate of recombination for each base.
            expected_cross_count: A float representing the expected cross number of crossing over events
                                  on each chromosome
            interval_size: If sampling from the whole chromosome is too slow, recombination rates can be given in
                           intervals of this size.
        """
        self.name = name
        self.length = length
        self.snp_locations = snp_locations
        self.states = states
        self.gene_to_snp_map = gene_to_snp_map
        self.recombination_rate = recombination_rate
        self.expected_cross_count = expected_cross_count
        self.interval_size = interval_size

        # Basic data validation
        if not np.all(self.snp_locations[:-1] <= self.snp_locations[1:]):
            raise ValueError('SNP locations must be sorted in ascending order.')

        # It is possible to have no snps on a chromosome
        if len(self.snp_locations) > 0:
            if self.snp_locations[-1] > self.length:
                raise ValueError('Cannot have snps at locations greater than the length of the chromosome.')

            if self.states.shape[1] != len(self.snp_locations) or \
                    (self.states.shape[0] != 2 and self.name != 'X'):
                raise ValueError('States must be 2 x n_snps boolean ndarray.')

            if self.name == 'X' and not (self.states.shape[0] == 1 or self.states.shape[0] == 2):
                print(self.states.shape)
                raise ValueError('X chromosome states must be 1 x n_snps boolean ndarray for males and 2 x n_snps boolean ndarray for females')


    def simulate_meiosis(self, crossing_over=True):
        """
        Simulates a new set of snp states after a crossing over occurs in meiosis.
        The returned result will be a haploid chromosome. It is assumed that exactly
        one crossing over event occurs.

        Returns:
        new_states: A haploid states array after crossing over (1 x n_snps) boolean ndarray.
        crossing_over: A boolean indicating whether crossing over occurs.
        """
        # Males will have a 1 x n_snps set of states and will have essentially no recombination
        # We do not simulate the Y chromosome, so simply return the only X state, and let gender be
        # decided downstream (ie this state may go unused)
        if self.states.shape[0] == 1:
            return self.states[0]

        # First choose which chromatid will be inherited (maternal or paternal)
        chromatid_selection = np.random.choice([0, 1])
        selected_chromatid = self.states[chromatid_selection, :].copy()

        # If crossing over doesn't occur, just choose one of the chromatids for inheritance
        if not crossing_over or self.expected_cross_count == 0:
            return selected_chromatid

        # The number of crossing over events is Poisson distributed
        num_crosses = np.random.poisson(lam=self.expected_cross_count)

        for _ in range(num_crosses):
            # Select a start and end index for the crossover
            num_locations = len(self.recombination_rate)
            cross_start_ind = np.random.choice(num_locations, p=self.recombination_rate)
            cross_start_ind = cross_start_ind * self.interval_size

            # Check if we have an empty chromosome
            if len(self.snp_locations) == 0:
                return self.states

            # Find the snps nearest the start and end
            dist_to_start = self.snp_locations - cross_start_ind
            max_int = np.iinfo(np.int32).max
            dist_to_start[dist_to_start < 0] = max_int
            location_start_ind = np.argmin(dist_to_start)

            # Create a new chromatid with a crossover at the specified locations
            other_chromatid = self.states[1-chromatid_selection, :]

            # Check which arm the breakpoint is on
            if cross_start_ind > self.length / 2:
                selected_chromatid[location_start_ind:] = other_chromatid[location_start_ind:]
            else:
                selected_chromatid[:location_start_ind] = other_chromatid[:location_start_ind]

        return selected_chromatid

    def sample(self, gene_umi, x_inact):
        """
        Converts umi from genes into variant and wild type reads depending on the states of the chromosome
        Args:
            gene_umi: The gene umi to convert.
            x_inact: Whether we simulate X inactivation (ie only one of the chromatids is active)

        Returns: wild_type_arr, variant_arr. Two numpy arrays representing the wild type and
        variant reads from each variant.
        """
        num_snps = len(self.snp_locations)
        wild_type_arr = np.zeros(num_snps)
        variant_arr = np.zeros(num_snps)

        genes = gene_umi.keys()
        # If we are practicing X inactivation, choose a chromatid at the start
        if x_inact:
            chromatid_num = np.random.choice(self.states.shape[0])

        for gene in genes:
            # Get associated variants for the gene
            associated_variants = self.gene_to_snp_map[gene]


            # For each UMI, map the variants to the state of one chromatid or the other
            for i in range(gene_umi[gene]):
                # No x inactivation means each umi comes from a random chromatid
                if not x_inact:
                    chromatid_num = np.random.choice(self.states.shape[0]) # Randomly select which chromatid to read

                variant_states = self.states[chromatid_num, associated_variants].flatten()

                # See whether the umi is a variant or wild type
                for state, variant_ind in zip(variant_states, associated_variants):
                    if state == 0:
                        wild_type_arr[variant_ind] += 1
                    else:
                        variant_arr[variant_ind] += 1

        return wild_type_arr, variant_arr

class Organism:
    def __init__(self, chr_names, chromosomes, gender):
        """
        Simulates an organism having chromosomes and snps. An individual SNP can be
        not present (wild type), heterozygous (mixed), or homozygous (mutant type).

        Args:
            chr_names: The names of the chromosomes. Must correspond to snp files
            chromosomes: A dict of Chromosomes.
            gender: 'M' or 'F'. The gender of the organism.
        """
        self.chr_names = chr_names
        self.chromosomes = chromosomes
        self.gender = gender

    def breed(self, organism2, males_cross_over=True):
        """
        Simulates breeding with another organism. I.e. returns a new Organism with chromosomes from
        the mother and father than have undergone meiosis.

        Returns:
            new_organism: A new Organism with chromosomes from mother and father
        """
        new_chromosomes = {}

        # Choose a gender for the new offspring
        gender = np.random.choice(['M', 'F'])

        for chr_name in self.chr_names:
            snp_locations = self.chromosomes[chr_name].snp_locations
            chr_len = self.chromosomes[chr_name].length
            gene_to_snp_map = self.chromosomes[chr_name].gene_to_snp_map

            maternal_state = self.chromosomes[chr_name].simulate_meiosis()
            paternal_state = organism2.chromosomes[chr_name].simulate_meiosis(crossing_over=males_cross_over)

            if chr_name == 'X' and gender == 'M':
                new_snp_state = np.reshape(maternal_state, (1, len(maternal_state)))
            else:
                new_snp_state = np.vstack((maternal_state, paternal_state))

            new_chr = Chromosome(chr_name, chr_len, snp_locations, new_snp_state, gene_to_snp_map,
                                 recombination_rate=organism2.chromosomes[chr_name].recombination_rate,
                                 expected_cross_count=organism2.chromosomes[chr_name].expected_cross_count,
                                 interval_size=organism2.chromosomes[chr_name].interval_size)
            new_chromosomes[chr_name] = new_chr

        return Organism(self.chr_names, new_chromosomes, gender)

    def sample_cells(self, mean_var_UMI, UMI_variance, num_cells, adata):
        """
        Samples cell snp profiles with the number of UMI taken from a normal distribution.
        Args:
            mean_var_UMI: The mean of the UMI distribution
            UMI_variance: The variance of the UMI distribution
            num_cells: The number of cells to sample
            adata: An adata of cells, L1 normalized. var must have a 'chromosome' attribute labelling membership.

        Returns: A list of num_cells x n_variants np arrays corresponding to each chromosome
        """
        organism_samples_wt = [np.zeros(shape=(num_cells, len(self.chromosomes[chr_name].snp_locations)), dtype=int) \
                               for chr_name in self.chr_names]
        organism_samples_var = [np.zeros(shape=(num_cells, len(self.chromosomes[chr_name].snp_locations)), dtype=int) \
                               for chr_name in self.chr_names]
        num_umi = np.random.normal(loc=mean_var_UMI, scale=UMI_variance, size=num_cells).astype(int)

        # Get the per-chromosome gene info
        adata_genes = adata.var['genes']
        chromosome_memberships = adata.var['chromosome']

        for i, cell in enumerate(range(num_cells)):
            # Sample a cell profile
            cell_profile_ind = np.random.choice(adata.X.shape[0])
            cell_profile = adata.X[cell_profile_ind]

            # Sample genes by index for that cell
            cell_umi = num_umi[i]
            gene_indices = np.random.choice(len(adata_genes), p=cell_profile, replace=True, size=cell_umi).astype(int)

            # Sort these into our UMI storage
            gene_umi = [{gene: 0 for gene in adata_genes[chromosome_memberships == chr_name]} \
                        for chr_name in self.chr_names]

            for gene_idx in gene_indices:
                gene_name = adata_genes[gene_idx]
                chromosome = chromosome_memberships[gene_idx]
                chromosome_idx = self.chr_names.index(chromosome)
                gene_umi[chromosome_idx][gene_name] += 1

            # Collect variant profiles from each chromosome
            for j, chr_name in enumerate(self.chr_names):
                # Inactivate one X chromatid in females
                #x_inact = ((self.gender == 'F') and (chr_name == 'X'))
                x_inact = False  # disable X-inactivation for Drosophila simulation (they do not inactivate)
                wild_type_sample, variant_sample = self.chromosomes[chr_name].sample(gene_umi[j],
                                                                                     x_inact=x_inact)
                organism_samples_wt[j][i] = wild_type_sample
                organism_samples_var[j][i] = variant_sample

        return organism_samples_wt, organism_samples_var

# Globals for workers
_shared_organism_list = None
_shared_male_inds = None
_shared_female_inds = None
_shared_males_cross_over = None

def _init_breed_worker(organism_list, male_inds, female_inds, males_cross_over):
    """
    Initialize worker globals so we don't pickle large data for each task.
    """
    global _shared_organism_list, _shared_male_inds, _shared_female_inds, _shared_males_cross_over
    _shared_organism_list = organism_list
    _shared_male_inds = male_inds
    _shared_female_inds = female_inds
    _shared_males_cross_over = males_cross_over

def _breed_worker(_):
    """
    Breed a single offspring using shared globals.
    """
    np.random.seed(_)
    female_ind = np.random.choice(_shared_female_inds)
    male_ind = np.random.choice(_shared_male_inds)

    organism1 = _shared_organism_list[female_ind]
    organism2 = _shared_organism_list[male_ind]

    return organism1.breed(organism2, males_cross_over=_shared_males_cross_over)


# Globals for sample_cells workers
_shared_mean_var_UMI = None
_shared_UMI_variance = None
_shared_num_cells = None
_shared_adata_X = None
_shared_adata_genes = None
_shared_adata_chromosomes = None

def _init_sample_worker(mean_var_UMI, UMI_variance, num_cells, adata_X, adata_genes, adata_chromosomes):
    """
    Store big/static variables in each worker once.
    """
    global _shared_mean_var_UMI, _shared_UMI_variance, _shared_num_cells
    global _shared_adata_X, _shared_adata_genes, _shared_adata_chromosomes

    _shared_mean_var_UMI = mean_var_UMI
    _shared_UMI_variance = UMI_variance
    _shared_num_cells = num_cells
    _shared_adata_X = adata_X
    _shared_adata_genes = adata_genes
    _shared_adata_chromosomes = adata_chromosomes

def _sample_worker(organism_index, organism):
    """
    Worker function for parallelizable sampling from organisms.

    :param organism_index: The index of the organism (for record keeping later).
    :param organism: The organism object.
    :return: organism_index, wt_reads, var_reads
    """
    np.random.seed()

    # To reduce memory usage use a minimal fake adata
    class MiniAdata:
        pass
    adata = MiniAdata()
    adata.X = _shared_adata_X
    adata.var = {
        'genes': _shared_adata_genes,
        'chromosome': _shared_adata_chromosomes
    }

    # Sample from the given organism
    wt_reads, var_reads = organism.sample_cells(_shared_mean_var_UMI, _shared_UMI_variance,
                                                _shared_num_cells, adata)

    return organism_index, wt_reads, var_reads


class BreedingPool:
    def __init__(self, organism_list, males_cross_over=True, pool=None):
        """
        A pool of organisms that can breed with each other.

        Args:
            organism_list: The starting list of organisms
            males_cross_over: Whether males experience crossing over
        """
        self.organism_list = organism_list
        self.males_cross_over = males_cross_over

    def breed_generation(self, num_offspring, n_processes=None):
        """
        Uses multiprocessing to make a new breeding pool with the specified number of offspring.
        """
        if n_processes is None:
            n_processes = mp.cpu_count()

        # Identify male and female organisms
        male_inds = np.array([i for i, o in enumerate(self.organism_list) if o.gender == 'M'], dtype=int)
        female_inds = np.array([i for i, o in enumerate(self.organism_list) if o.gender != 'M'], dtype=int)

        with mp.Pool(
            processes=n_processes,
            initializer=_init_breed_worker,
            initargs=(self.organism_list, male_inds, female_inds, self.males_cross_over)
        ) as pool:
            new_organism_list = list(pool.map(_breed_worker, range(num_offspring)))

        return BreedingPool(new_organism_list, self.males_cross_over)

    def sample_cells(self, mean_var_UMI, UMI_variance,
                     num_cells, adata, doublet_rate=0, n_processes=None):
        """
        Samples cells from the pool of organisms with a specified doublet rate.

        :param mean_var_UMI: The mean number of read UMI's to sample (from a normal distribution).
        :param UMI_variance: The variance for read UMI's (sampled from a normal distribution).
        :param num_cells: The number of cells per organism to sample.
        :param adata: A starting (real) scRNA-seq dataset to sample reads from.
        :param doublet_rate: The doublet rate to use.
        :param n_processes: The number of threads to use.
        :return: (list of np.array of shapes (num_cells * len(self.organism_list), num_variants_chr))
        """
        if n_processes is None:
            n_processes = mp.cpu_count()

        # Determine how many cells we actually need to sample when doublets are factored in
        num_doublets = int(round(doublet_rate * num_cells))
        num_singlets = num_cells - num_doublets
        required_cells = num_singlets + 2 * num_doublets

        with mp.Pool(processes=n_processes,
                     initializer=_init_sample_worker,
                     initargs=(mean_var_UMI, UMI_variance, required_cells,
                               adata.X, adata.var['genes'].to_numpy(), adata.var['chromosome'].to_numpy())
        ) as pool:
            results = pool.starmap(_sample_worker, [(i, organism) for i, organism in enumerate(self.organism_list)])

        adata = self._results_to_anndata(results)

        # Combine cells to create doublets
        if doublet_rate > 0:
            adata = self._add_doublets(adata, num_doublets)

        return adata, self.organism_list

    def _add_doublets(self, adata, num_doublets):
        """
        Randomly combines cells into synthetic doublets.

        :param adata: Input anndata of cells x variants.
        :param num_doublets: Number of doublets per organism.
        :return: AnnData with doublets.
        """
        if num_doublets == 0:
            return adata

        total_doublets = num_doublets * len(self.organism_list) # The final number of doublets we'll have
        num_doublet_precursor = 2 * total_doublets # The number of precursor cells we need to combine to make them

        # Extract current layers
        X = adata.X.tocsr()
        depth = adata.layers["depth"].tocsr()

        # Randomly permute all cells so we can select singlets and doublets
        perm = np.random.permutation(adata.n_obs)
        doublet_cells = perm[:num_doublet_precursor]
        singlet_cells = np.sort(perm[num_doublet_precursor:])

        # Start creating new alt and depth, as well as obs for the anndata
        X_blocks = [X[singlet_cells]]
        depth_blocks = [depth[singlet_cells]]
        obs = adata.obs.iloc[singlet_cells].copy()

        for i in range(total_doublets):
            # Grab pairs of cells
            idx1 = doublet_cells[2 * i]
            idx2 = doublet_cells[2 * i + 1]

            # Combine their alt and depth values
            X_blocks.append(X[idx1] + X[idx2])
            depth_blocks.append(depth[idx1] + depth[idx2])

            # Set their identity and doublet status
            obs.loc[len(obs)] = {
                "identity": int(adata.obs.iloc[idx1]["identity"]),
                "identity_2": int(adata.obs.iloc[idx2]["identity"]),
                "is_doublet": True,
            }

        # Make a new matrix with our doublets added
        adata_out = anndata.AnnData(
            X=sparse.vstack(X_blocks, format="csr"),
            obs=obs.reset_index(drop=True),
            var=adata.var.copy(),
        )
        adata_out.layers["depth"] = sparse.vstack(depth_blocks, format="csr")

        return adata_out

    def _results_to_anndata(self, results):
        """
        Converts multiprocessing results into an anndata of cells.

        :param results: (list of tuples) (organism, wt_counts, variant_counts)
        :return: anndata with
                 X: variant counts
                 layers['depth']: depth counts.
                 var['chromosome']: The chromosome the variant is located on.
                 var['position']: The genomic position of the variant.
                 obs['identity']: The ground truth individual's index.
                 obs['identity_2']: If the cell is a doublet, the identity of the second individual of origin.
                 obs['is_doublet']: Whether the cell is a doublet.
        """
        chr_names = self.organism_list[0].chr_names

        alt_blocks = []
        depth_blocks = []

        obs_rows = []
        var_rows = []

        for chr_name in chr_names:
            # The var df will have the chromosome and snp location for each entry
            chromosome = self.organism_list[0].chromosomes[chr_name]
            positions = chromosome.snp_locations

            for pos in positions:
                var_rows.append({"chromosome": chr_name,
                                 "position": pos})

        # Construct the var dataframe
        var = pd.DataFrame(var_rows)

        # The obs dataframe needs to store the identity of each cell (and doublet identity)
        for organism_index, wt_list, var_list in results:
            alt = np.hstack(var_list)
            wt = np.hstack(wt_list)

            alt_blocks.append(alt)
            depth_blocks.append(alt + wt)

            # For each cell in the organism, make an entry
            n = alt.shape[0]
            obs_rows.extend([{"identity": organism_index, "identity_2": -1, "is_doublet": False} for _ in range(n)])

        # Create obs and then the data matrices
        obs = pd.DataFrame(obs_rows).astype({"identity": np.int32,
                                             "identity_2": np.int32,
                                             "is_doublet": bool})
        X = np.vstack(alt_blocks)
        depth = np.vstack(depth_blocks)

        adata = anndata.AnnData(X=csr_matrix(X), obs=obs, var=var)
        adata.layers["depth"] = csr_matrix(depth)

        return adata

class BreedingExperiment:
    def __init__(self, starting_organism_list, offspring_per_generation, males_cross_over=True):
        """
        A breeding experiment symbolizing the creation of n generations of breeding.
        Args:
            starting_organism_list: A list of organisms for generation 0
            offspring_per_generation: The number of offspring we produce at each generation
            males_cross_over: Whether crossing over events occur in males
        """

        self.organism_list = starting_organism_list
        self.offspring_per_generation = offspring_per_generation

        self.starting_pool = BreedingPool(starting_organism_list, males_cross_over)

    def run_experiment(self, num_generations, n_proc=None):
        """
        Runs a breeding experiment for n generations.

        :param num_generations: The number of generations to run.
        :param n_proc: The number of processes to use.
        :return: current_generation: The final generation of the breeding experiment.
        """
        current_generation = self.starting_pool

        for i in range(1, num_generations + 1):
            current_generation = current_generation.breed_generation(self.offspring_per_generation, n_processes=n_proc)

        return current_generation