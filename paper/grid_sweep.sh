#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
num_emb=100
cells_per_emb=250

for umi in 100 500 1000 2500 5000 10000 20000; do
for num_gen in 2 4 6 8 10 12 14 16 18 20 22 24; do
  python simulate_fly_breeding.py \
    --save_dir "sweeps/grid_sweep/" \
    --data_path "sim_data/grid_sweep/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$umi}_num_cells_{$cells_per_emb}.pickle" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI $umi \
    --num_cells_per_org $cells_per_emb \
    --offspring_per_generation $num_emb \
    --num_threads 80 \
    --no_souporcell \
    --no_vireo \
    --no_scsplit

  ((i++))
done
done
