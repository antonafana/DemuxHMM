#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

UMI=10000
NUM_EMB=50
DATA_DIR="sim_data/simulated_doublets/"
cell_budget=20000
NUM_CELLS=$((cell_budget / NUM_EMB))
num_gen=22
MAX_JOBS=4
i=0

mkdir -p sim_data

for doublet_rate in 0.025 0.05 0.1 0.2; do
  for run in {0..5}; do

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n
    done

    python simulate_fly_breeding.py \
        --data_path "$DATA_DIR/run_$run/num_emb_${NUM_EMB}_num_gen_${num_gen}_avg_UMI_${UMI}_num_cells_${NUM_CELLS}_doublet_rate_${doublet_rate}.h5ad" \
        --run_num "$i" \
        --num_generations "$num_gen" \
        --avg_UMI "$UMI" \
        --num_cells_per_org "$NUM_CELLS" \
        --offspring_per_generation "$NUM_EMB" \
        --doublet_rate "$doublet_rate" \
        --num_threads 5 &

    ((i++))
  done
done

wait