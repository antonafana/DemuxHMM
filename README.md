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
and using the run with the best model probability, like the below example:

```python
import numpy as np
from demuxHMM import HMMModel

num_repeats = 30
tolerance = 1
num_individuals = 100 
mean_transitions = 10
best_model_score = -np.inf
best_model = None

for _ in range(num_repeats):
    model = HMMModel(A, D, num_individuals, mean_transitions)
    model.solve(tolerance, num_inits=0)
    model_out = model.get_output()

    if model_out['probability'] > best_model_score:
        best_model_score = model_out['probability']
        best_model = model_out
```

For more complex examples and practical applications, please see the "paper" directory.

## Requirements and Future Changes
Currently, the model's GPU mode requires a CUDA capable GPU with 9GB+ of VRAM. It is a high priority for us to release
a VRAM-scalable version. The model can be run on CPU, but is slower. We also plan to release several quality-of-life 
improvements to the API to simplify usage. If you have any suggestions, or notice any problems, please open an issue.

