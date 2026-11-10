#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

i=0
num_gen=22
num_cells=250
num_emb=1000
run_num=0
MAX_JOBS=1

if [ $# -ne 1 ]; then
    echo "Usage: $0 {generate data (True | False)?}"
    exit 1
fi

mkdir -p sim_data

if [[ $1 == "True" ]]; then
  for run in {0..2}; do
    # Block until fewer than MAX_JOBS are running
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
      wait -n
    done

    python simulate_fly_breeding.py \
      --data_path "sim_data/high_emb_counts/run_$run/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{10000}_num_cells_{$num_cells}.h5ad" \
      --run_num "$i" \
      --num_repeats 30 \
      --num_generations "$num_gen" \
      --avg_UMI 10000 \
      --num_cells_per_org "$num_cells" \
      --offspring_per_generation "$num_emb" \
      --num_threads 40 &

    ((i++))
  done

  wait
fi

run_num=0

for run_num in {0..2}; do
  # Block until fewer than MAX_JOBS are running
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    wait -n
  done

  python simulate_fly_breeding.py \
    --data_path "sim_data/high_emb_counts/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{10000}_num_cells_{$num_cells}.h5ad" \
    --save_dir "sweeps/simulated_high_emb/run_$run_num/" \
    --run_num "$i" \
    --num_repeats 60 \
    --num_generations "$num_gen" \
    --avg_UMI 10000 \
    --num_cells_per_org "$num_cells" \
    --offspring_per_generation "$num_emb" \
    --num_threads 3 \
    --demuxHMM &

  ((i++))
done

wait