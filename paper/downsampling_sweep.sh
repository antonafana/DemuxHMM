#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

num_emb=100
cells_per_emb=250
num_gen=22
umi=10000

i=0
MAX_JOBS=2

# This sweep uses a dataset from the grid sweep featured in the appendix. If this wasn't run
# beforehand, make sure to generate the dataset first.
for run_num in {6..9}; do
for snp_usage_percent in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --save_dir "sweeps/downsample_sweep/run_$run_num/" \
    --data_path "sim_data/grid_sweep/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$umi}_num_cells_{$cells_per_emb}.pickle" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI $umi \
    --num_cells_per_org $cells_per_emb \
    --offspring_per_generation $num_emb \
    --snp_usage_percent $snp_usage_percent \
    --num_threads 10 \
    --no_souporcell \
    --no_vireo \
    --no_scsplit &

  ((i++))
done
done

wait