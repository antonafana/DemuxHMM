# DemuxHMM Paper Code
This folder contains the code needed to replicate the results in the DemuxHMM manuscript. Briefly, we simulate
SNP datasets on Drosophila Melanogaster using real SNP positions of two divergent strains, and real scRNA-seq reads.
The method is then benchmarked on this simulated data. We also include a benchmark for a common PBMC dataset from Demuxlet
which does not have the same structuring on SNPs. Computing was done on a server with 190 CPU cores 
(AMD EPYC-Genoa), 390 GB of RAM, and an NVIDIA RTX 4500 Ada Generation GPU with 24 GB of VRAM.
The scripts in this folder are currently tuned to such an environment. As such, users may have to adjust parameters
to reflect their own compute environment.

## Pre-processing
First, we must get and pre-process the pre-cursor data. The pre-processing creates mapping files between expressed
genes and SNPs that could conservatively be seen using scRNA-seq technology. 

```bash
sh get_datasets.sh
python fly_variant_to_gene_mapping_fly_atlas.py
```
## Benchmarking and Simulation

### Simulating and Evaluating Datasets
Simulation of datasets and evaluation, including by competing methods, happens through the entrypoint `simulate_fly_breeding.py`.
This script has a lot of parameters to play with, so we recommend looking at the bash scripts we use to generate the
figures. We detail these below. If you want to run souporcell or scSplit, you'll need to download these and pass
the path to their binaries when you run the script. For scSplit, we recommend using my [fork](https://github.com/antonafana/scSplit)
, which is more up to date package-wise and more numerically stable.

To install scSplit, run the following
```bash
git clone https://github.com/antonafana/scSplit
```

To install souporcell3, run the below. If you already have rust installed, skip that step.
```bash
git clone https://github.com/wheaton5/souporcell.git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cd souporcell/souporcell && cargo build --release
```

### Figures 4 and 5
These figures are a methods comparison sweep over a number of datasets. We pre-generate the datasets and then run the 
methods on each one. Note, this takes a while... You may have to adjust parameters to fit your system specs, or may
wish to run some of the parameter sweeps concurrently.

```bash
mkdir figures
sh generate_datasets.sh
sh comparison_demuxHMM.sh
sh comparison_scsplit.sh
sh comparison_vireo.sh
sh comparison_sorc3.sh
python comparison_fig.py
```
### Figure 6
This figure focuses on examining the effects of demultiplexing error on trajectory inference using a [sea urchin snRNA-seq
dataset](https://journals.biologists.com/dev/article/148/19/dev198614/272307/Developmental-single-cell-transcriptomics-in-the).
We use a version of the dataset processed for use in python. Users should be able to obtain similar results by converting
the [publicly posted dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184538). 
Please contact me if you have questions about how to do this. To run the analysis:

```bash
python lv_urchin_ari_analysis.py
```

### Figure B.1
This is a parameter sweep of DemuxHMM method on a grid of parameters. To run:

```bash
sh grid_sweep.sh
python grid_fig.py
```

### Figure B.2
This is a performance test where we fix all parameters but the number of breeding generations and evaluate performance.
To run:

```bash
sh generate_datasets_generations.sh
sh generations_sweep.sh
python ari_v_generations_fig.py
```

### Figure B.3
This figure evaluates performance with respect to SNP downsampling. It assumes the datasets from the grid sweep figure
have already been generated. To run:

```bash
sh downsampling_sweep.sh
python downsampling_fig.py
```

### PBMC Dataset
Running the pre-processing should have downloaded demuxlet's results from their paper, as well as an anndata of our
cellSNP-lite variant calls (`--minMAF 0.1 --minCOUNT 20`). The benchmark can be run with:

```bash
python PBMC_HMM.py
```
