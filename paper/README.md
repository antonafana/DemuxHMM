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
genes and SNPs that could conservatively be seen using scRNA-seq technology. Users may omit the python pre-processing
if desired as we have included the end result.

```console
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


### PBMC Dataset
Running the pre-processing should have downloaded demuxlet's results from their paper, as well as an anndata of our
cellSNP-lite variant calls (`--minMAF 0.1 --minCOUNT 20`). The benchmark can be run with:

```console
python PBMC_HMM.py
```
