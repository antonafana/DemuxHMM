import numpy as np
import multiprocessing as mp

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

def _init_breed_worker(organism_list, male_inds, female_inds, males_cross_over, backcross):
    """
    Initialize worker globals so we don't pickle large data for each task.
    """
    global _shared_organism_list, _shared_male_inds, _shared_female_inds, _shared_males_cross_over, _backcross
    _shared_organism_list = organism_list
    _shared_male_inds = male_inds
    _shared_female_inds = female_inds
    _shared_males_cross_over = males_cross_over
    _backcross = backcross

def _breed_worker(_):
    """
    Breed a single offspring using shared globals.
    """
    np.random.seed(_)
    female_ind = np.random.choice(_shared_female_inds)
    male_ind = np.random.choice(_shared_male_inds)

    if _backcross is None:
        organism1 = _shared_organism_list[female_ind]
        organism2 = _shared_organism_list[male_ind]
    else:
        backcross_gender = _backcross.gender

        if backcross_gender == 'M':
            organism1 = _shared_organism_list[female_ind]
            organism2 = _backcross
        else:
            organism1 = _backcross
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

def _sample_worker(organism):
    """
    Worker function that reconstructs a minimal adata-like object from globals.
    """
    np.random.seed()
    class MiniAdata:
        pass
    adata = MiniAdata()
    adata.X = _shared_adata_X
    adata.var = {
        'genes': _shared_adata_genes,
        'chromosome': _shared_adata_chromosomes
    }
    return organism.sample_cells(_shared_mean_var_UMI, _shared_UMI_variance, _shared_num_cells, adata)


class BreedingPool:
    def __init__(self, organism_list, males_cross_over=True):
        """
        A pool of organisms that can breed with each other.

        Args:
            organism_list: The starting list of organisms
            males_cross_over: Whether males experience crossing over
        """
        self.organism_list = organism_list
        self.males_cross_over = males_cross_over


    def breed_generation(self, num_offspring, n_processes=None, backcross=None):
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
            initargs=(self.organism_list, male_inds, female_inds, self.males_cross_over, backcross)
        ) as pool:
            new_organism_list = list(pool.map(_breed_worker, range(num_offspring)))

        return BreedingPool(new_organism_list, self.males_cross_over)

    def sample_cells(self, mean_var_UMI, UMI_variance, num_cells, adata, n_processes=None):
        """
        Multiprocessed sample_cells with adata variables stored in worker globals.
        """
        if n_processes is None:
            n_processes = mp.cpu_count()

        num_chromosomes = len(self.organism_list[0].chr_names)

        with mp.Pool(
                processes=n_processes,
                initializer=_init_sample_worker,
                initargs=(
                        mean_var_UMI,
                        UMI_variance,
                        num_cells,
                        adata.X,
                        adata.var['genes'].to_numpy(),
                        adata.var['chromosome'].to_numpy()
                )
        ) as pool:
            results = pool.map(_sample_worker, self.organism_list)

        # Initialize final matrices
        samples_mat_wt = []
        samples_mat_var = []
        for j in range(num_chromosomes):
            chr_name = self.organism_list[0].chr_names[j]
            num_variants_chr = len(self.organism_list[0].chromosomes[chr_name].snp_locations)
            wt_mat = np.zeros((num_cells * len(self.organism_list), num_variants_chr))
            var_mat = np.zeros_like(wt_mat)
            samples_mat_wt.append(wt_mat)
            samples_mat_var.append(var_mat)

        # Fill in data
        for i, (org_wt, org_var) in enumerate(results):
            for j in range(num_chromosomes):
                samples_mat_wt[j][num_cells * i:num_cells * (i + 1)] = org_wt[j]
                samples_mat_var[j][num_cells * i:num_cells * (i + 1)] = org_var[j]

        return samples_mat_wt, samples_mat_var


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

    def run_experiment(self, num_generations, n_proc=None, backcross=None):
        """
        Runs a breeding experiment to create num_generations of organisms.
        Args:
            num_generations: The number of generations to run. 0-indexed.
            backcross: If we want to breed with only a specific organism for each generation

        Returns: list of BreedingPools indexed by generation
        """

        current_generation = self.starting_pool

        for i in range(1, num_generations + 1):
            current_generation = current_generation.breed_generation(self.offspring_per_generation,
                                                                n_processes=n_proc, backcross=backcross)

        return current_generation


