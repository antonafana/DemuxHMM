#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
num_gen=22
num_cells=250
num_emb=1000
run_num=0
MAX_JOBS=1

mkdir -p sim_data

for run in {0..2}; do
  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --data_path "sim_data/high_emb_counts/run_$run/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{10000}_num_cells_{$cells}.pickle" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI 10000 \
    --num_cells_per_org $cells \
    --offspring_per_generation $num_emb \
    --num_threads 40 \
    --no_demuxHMM \
    --no_scsplit \
    --no_souporcell \
    --no_vireo &

  ((i++))
done

wait

run_num=0

for run_num in {0..2}; do
  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --data_path "sim_data/high_emb_counts/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{10000}_num_cells_{$num_cells}.pickle" \
    --save_dir "sweeps/high_emb/run_$run_num/" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI 10000 \
    --num_cells_per_org $num_cells \
    --offspring_per_generation $num_emb \
    --num_threads 15 \
    --no_souporcell \
    --no_vireo \
    --no_scsplit &

  ((i++))
done

wait