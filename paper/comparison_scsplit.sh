#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
cell_budget=20000
num_gen=22
MAX_JOBS=100

for num_emb in 350 250 200 150 100 75 50 25 10; do
for run_num in {0..5}; do

  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --data_path "sim_data/G22_20k_runs/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{10000}_num_cells_{$((cell_budget / num_emb))}.pickle" \
    --save_dir "sweeps/scsplit/run_$run_num/" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI 10000 \
    --num_cells_per_org $((cell_budget / num_emb)) \
    --offspring_per_generation $num_emb \
    --num_threads 1 \
    --no_demuxHMM \
    --no_souporcell \
    --no_vireo &

  ((i++))
done
done

wait
