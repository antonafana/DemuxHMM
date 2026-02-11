#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
MAX_JOBS=7

mkdir -p data_dir/transitions/

for num_gen in {1..25}; do
  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --transition_save_path "data_dir/transitions/transitions_{$num_gen}_gen.pickle" \
    --save_dir "sweeps/transitions/" \
    --run_num "$i" \
    --num_repeats 30 \
    --num_generations $num_gen \
    --avg_UMI 100 \
    --num_cells_per_org 1 \
    --offspring_per_generation 1000 \
    --num_threads 15 \
    --no_demuxHMM \
    --no_souporcell \
    --no_vireo \
    --no_scsplit &

  ((i++))
done

wait
