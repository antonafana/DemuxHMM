#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
num_emb=100
cells_per_emb=250
MAX_JOBS=7

if [ $# -ne 1 ]; then
    echo "Usage: $0 {generate data (True | False)?}"
    exit 1
fi

# Generating the datasets first is much more efficient
if [[ $1 == "True" ]]; then
  mkdir -p sim_data

  for run_num in {0..5}; do
    for umi in 100 500 1000 2500 5000 10000 20000; do
      for num_gen in 2 4 6 8 10 12 14 16 18 20 22 24; do

          # Block until fewer than MAX_JOBS are running
          while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            wait -n
          done

          python simulate_fly_breeding.py \
            --data_path "sim_data/grid_sweep/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$umi}_num_cells_{$cells_per_emb}.h5ad" \
            --run_num "$i" \
            --num_repeats 30 \
            --num_generations "$num_gen" \
            --avg_UMI "$umi" \
            --num_cells_per_org "$cells_per_emb" \
            --offspring_per_generation "$num_emb" \
            --num_threads 15 &

          ((i++))
      done
    done
  done

  wait
fi

MAX_JOBS=1
i=0

for run_num in {0..5}; do
  for umi in 100 500 1000 2500 5000 10000 20000; do
    for num_gen in 2 4 6 8 10 12 14 16 18 20 22 24; do

      # Block until fewer than MAX_JOBS are running
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n
      done

      python simulate_fly_breeding.py \
        --save_dir "sweeps/simulated_grid_sweep/run_$run_num/" \
        --data_path "sim_data/grid_sweep/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$umi}_num_cells_{$cells_per_emb}.h5ad" \
        --run_num "$i" \
        --num_repeats 30 \
        --num_generations "$num_gen" \
        --avg_UMI "$umi" \
        --num_cells_per_org "$cells_per_emb" \
        --offspring_per_generation "$num_emb" \
        --num_threads 3 \
        --demuxHMM &

      ((i++))
    done
  done
done

wait
