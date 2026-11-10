# DemuxHMM
DemuxHMM is a highly performant demultiplexing method for scRNA-seq and other omics. Under the hood, it uses a Hidden
Markov Model to capture structure on SNPs when individuals are created with a breeding scheme. DemuxHMM is most powerful
with this structure, but it can also work on unstructured datasets. For fast computation, we leverage parallelization on
GPU to achieve significantly faster performance than competing methods on large datasets. The manuscript for the method
can be seen on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.23.703392v1). Please cite us if you use the code in your work.

## Installation
We offer two installation paths, depending on whether you want to run the benchmarks presented in the paper or not.
For simple installation, clone this repo and install it with pip:

```bash
git clone https://github.com/antonafana/DemuxHMM && cd DemuxHMM
pip install .
```

If you want to run benchmarks from the paper, install the optional dependencies with:

```bash
pip install .[paper]
```

## Usage

As input, the method expects an array of single-cell variant call matrices `A` and `D`. Currently, these should be 
formatted as an array of length corresponding to the number of chromosomes. `A[i]` and `D[i]` should numpy matrices of 
shape `(n_cells, n_snps_chromosome_i)` containing allele counts and overall depth for each cell and snp position 
combination on chromosome `i`. Cells should already be quality controlled and cleaned of doublets. 
We suggest [CellSNP-lite](https://github.com/single-cell-genetics/cellsnp-lite) or similar for variant calling.
The method also requires an upper bound on the number of individuals, and an estimate on the number of 
state changes per chromosome (ex. homozygous reference -> heterozygous). We suggest repeating inference multiple times
and using the run with the best model probability. We give an example for a single initialization:

```python
from demuxHMM import HMMModel
from demuxHMM.demuxHMM_utils import adata_to_HMM
import anndata

num_repeats = 30
tolerance = 1
num_individuals = 100 
mean_transitions = 10

# Cells x variants adata. X is variant calls, layers['depth'] is depth
adata = anndata.read_h5ad('variant_calls.h5ad')
A, D = adata_to_HMM(adata, adata.var['chromosome'].unique())

model = HMMModel(A, D, num_individuals, mean_transitions)
model.solve(tolerance)
best_model = model.get_output()
```

For more complex examples and practical applications, please see the "paper" directory.

## Requirements and Future Changes
Currently, the model's GPU mode requires a CUDA capable GPU with multiple GB of VRAM (scaling depending on number of cells and SNPs). The model can be run on CPU, but is much slower. We also plan to release several quality-of-life 
improvements to the API to simplify usage. If you have any suggestions, or notice any problems, please open an issue.

