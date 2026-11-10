#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

num_emb=200
cells_per_emb=100
num_gen=22
umi=10000

i=0
MAX_JOBS=1

# This sweep uses a dataset from the grid sweep featured in the appendix. If this wasn't run
# beforehand, make sure to generate the dataset first.
for run_num in {0..5}; do
  for snp_shuffle_percent in 0 0.01 0.05 0.1 0.25 0.5 0.75 0.9 1; do
      # Block until fewer than MAX_JOBS are running
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n
      done

      python simulate_fly_breeding.py \
        --save_dir "sweeps/simulated_snp_shuffle_Z_200/run_$run_num/" \
        --data_path "sim_data/G22_20k_runs/run_${run_num}/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$umi}_num_cells_{$cells_per_emb}.h5ad" \
        --run_num "$i" \
        --num_repeats 30 \
        --num_generations $num_gen \
        --avg_UMI "$umi" \
        --num_cells_per_org "$cells_per_emb" \
        --offspring_per_generation "$num_emb" \
        --snp_shuffle_percent "$snp_shuffle_percent" \
        --num_threads 25 \
        --demuxHMM \
        --vireo &

      ((i++))
  done
done

wait