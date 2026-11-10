#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

if [ $# -ne 1 ]; then
    echo "Usage: $0 {10000|2500}"
    exit 1
fi

case "$1" in
    10000)
        UMI=10000
        DATA_DIR="sim_data/G22_20k_runs"
        ;;
    2500)
        UMI=2500
        DATA_DIR="sim_data/G22_20k_runs_2500_UMI"
        ;;
    *)
        echo "Unknown UMI level: $1"
        exit 1
        ;;
esac

i=0
cell_budget=20000
num_gen=22
MAX_JOBS=7

mkdir -p sim_data

for run in {0..5}; do
    for num_emb in 10 25 50 75 100 150 200 250 350 500; do

        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            wait -n
        done

        python simulate_fly_breeding.py \
            --data_path "$DATA_DIR/run_$run/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$UMI}_num_cells_{$((cell_budget / num_emb))}.h5ad" \
            --run_num "$i" \
            --num_repeats 30 \
            --num_generations "$num_gen" \
            --avg_UMI "$UMI" \
            --num_cells_per_org $((cell_budget / num_emb)) \
            --offspring_per_generation "$num_emb" \
            --num_threads 15 &

        ((i++))

    done
done

wait