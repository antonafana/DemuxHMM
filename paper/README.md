# DemuxHMM Paper Code
This folder contains the code needed to replicate the results in the DemuxHMM manuscript. The manuscript looks at two sets of data. The first is a datasets of 77 Diversity Outbred mouse embryos. The second is a set of simulated Drosophila Melanogaster datasets with 22 generations of breeding. Computing was done on a server with 190 CPU cores(AMD EPYC-Genoa), 390 GB of RAM, and an NVIDIA RTX 4500 Ada Generation GPU with 24 GB of VRAM. The scripts in this folder are currently tuned to such an environment. As such, users may have to adjust parameters to reflect their own compute environment. We assume your virtual environment is in the repo base as py314_venv (adjust as needed).

# Pre-processing
For DO mouse, we have to fetch the dataset, and then do the full mapping and variant calling workflow. You can follow the [README.md](DO_mouse_preprocessing/README.md) file in [DO_mouse_preprocessing](DO_mouse_preprocessing).

For fly, we must get and pre-process the pre-cursor data. The pre-processing creates mapping files between expressed genes and SNPs that could conservatively be seen using scRNA-seq technology. 

```bash
sh get_datasets.sh
python fly_variant_to_gene_mapping_fly_atlas.py
```
# Benchmarking and Simulation
## Installing Competing Methods
Vireo should already be installed if you followed the instructions on the main page, as it is a required package. The others take a touch more work.

For scSplit, I recommend using my [fork](https://github.com/antonafana/scSplit), which is more up to date package-wise and more numerically stable. Run the following:
```bash
git clone https://github.com/antonafana/scSplit
```

To install souporcell3, run the below. If you already have rust installed, skip that step.
```bash
git clone https://github.com/wheaton5/souporcell.git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd souporcell/souporcell && cargo build --release
```

## DO Mouse
### Figure 3
This figure looks at homogeneity as we downsample the number of counts seen in the dataset. We need to sweep over the competing methods, which is quite time-consuming. I recommend an HPC environment if you want to recreate it fully. Create it by running:
```bash
sh DO_mouse_downsampling_sweep.sh demuxHMM True
sh DO_mouse_downsampling_sweep.sh vireo True
sh DO_mouse_downsampling_sweep.sh souporcell True
python DO_mouse_cluster_error_dist.py
```

### Figure 4
We can also represent error with a set of cluster pi charts on the non-downsampled dataset.
```bash
python DO_mouse_cluster_pi_charts.py
```

## Simulated Drosophila Datasets
Simulation of fly datasets and evaluation, including by competing methods, happens through the entrypoint [simulate_fly_breeding.py](simulate_fly_breeding.py). This script has a lot of parameters to play with, so we recommend looking at the bash scripts we use to generate the figures. We detail these below.

### Figure 5
These figures are a methods comparison sweep over a number of datasets. We pre-generate the datasets and then run the methods on each one. This takes a very long time if you don't tune to your compute environment to run jobs concurrently.

```bash
mkdir figures
mkdir journal_figures
sh simulated_generate_datasets.sh 10000
sh simulated_generate_datasets.sh 2500
sh simulated_generate_transitions.sh
sh simulated_compare_methods.sh demuxHMM
sh simulated_compare_methods.sh scsplit
sh simulated_compare_methods.sh vireo
sh simulated_compare_methods.sh sorc3
sh simulated_compare_methods.sh demuxHMM2500
sh simulated_compare_methods.sh scsplit2500
sh simulated_snp_shuffling_sweep.sh
python simulated_comparison_fig.py
```
For the high embryo count test simply run

```bash
sh simulated_high_emb.sh
python simulated_high_emb_stats.py
```

### Figure 6
This figure focuses on examining the effects of demultiplexing error on trajectory inference using a [sea urchin snRNA-seq
dataset](https://journals.biologists.com/dev/article/148/19/dev198614/272307/Developmental-single-cell-transcriptomics-in-the). We use a version of the dataset processed for use in python. Users can download the [publicly posted dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184538) and convert it into h5ad format. To run the analysis:

```bash
python lv_urchin_ari_analysis.py
```

### Figure 7
This is a parameter sweep of demuxHMM over a range of parameters. The first is a grid sweep over UMI and breeding generations. The second looks at the effect of downsampling SNPs.
```bash
sh simulated_grid_sweep.sh
sh simulated_snp_downsampling_sweep.sh
python simulated_conditions_fig.py
```

### Figure 8 and Figure 10 (S1)
These figures compare doublet detection performance across the methods on a 50 individual dataset.

```bash
sh simulated_generate_doublets.sh
sh simulated_sweep_doublets.sh demuxHMM
sh simulated_sweep_doublets.sh vireo
sh simulated_sweep_doublets.sh scsplit
sh simulated_sweep_doublets.sh souporcell
python simulated_doublet_figure.py
```

### Figure 9
This figure looks at how to choose the optimal number of clusters when the number is unknown.

```bash
sh optimal_cluster_num_sweep.sh mouse
sh optimal_cluster_num_sweep.sh simulated
python optimal_cluster_num_plot.py
```