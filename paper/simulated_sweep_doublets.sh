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

if [ $# -ne 1 ]; then
    echo "Usage: $0 {demuxHMM|scsplit|souporcell|vireo}"
    exit 1
fi

METHOD=$1

case "$METHOD" in
    demuxHMM)
        SAVE_DIR="sweeps/simulated_doublets/demuxHMM/"
        METHOD_FLAG="--demuxHMM"
        NUM_THREADS=3
        MAX_JOBS=1
        ;;

    scsplit)
        SAVE_DIR="sweeps/simulated_doublets/scSplit/"
        METHOD_FLAG="--scsplit"
        NUM_THREADS=1
        MAX_JOBS=100
        ;;

    souporcell)
        SAVE_DIR="sweeps/simulated_doublets/sorc3/"
        METHOD_FLAG="--souporcell"
        NUM_THREADS=60
        MAX_JOBS=1
        ;;

    vireo)
        SAVE_DIR="sweeps/simulated_doublets/vireo/"
        METHOD_FLAG="--vireo"
        NUM_THREADS=50
        MAX_JOBS=2
        ;;

    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

for run in {0..5}; do
  for doublet_rate in 0.025 0.05 0.1 0.2; do

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        wait -n
    done

    python simulate_fly_breeding.py \
        --data_path "$DATA_DIR/run_$run/num_emb_${NUM_EMB}_num_gen_${num_gen}_avg_UMI_${UMI}_num_cells_${NUM_CELLS}_doublet_rate_${doublet_rate}.h5ad" \
        --save_dir "$SAVE_DIR/run_$run/" \
        --run_num "$i" \
        --num_generations "$num_gen" \
        --avg_UMI "$UMI" \
        --num_cells_per_org "$NUM_CELLS" \
        --offspring_per_generation "$NUM_EMB" \
        --doublet_rate "$doublet_rate" \
        --detect_doublets \
        --num_repeats 30 \
        --num_threads "$NUM_THREADS" \
        "$METHOD_FLAG" &

    ((i++))
  done
done

wait