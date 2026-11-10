#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

if [ $# -ne 1 ]; then
    echo "Usage: $0 {demuxHMM|scsplit|souporcell|vireo|demuxHMM2500|scsplit2500}"
    exit 1
fi

METHOD="$1"

cell_budget=20000
num_gen=22
i=0

case "$METHOD" in
    demuxHMM)
        SAVE_DIR="sweeps/simulated_demuxHMM/"
        METHOD_FLAG="--demuxHMM"
        DATA_DIR="sim_data/G22_20k_runs"
        UMI=10000
        NUM_THREADS=8
        MAX_JOBS=1
        ;;

    scsplit)
        SAVE_DIR="sweeps/simulated_scsplit"
        METHOD_FLAG="--scsplit"
        DATA_DIR="sim_data/G22_20k_runs"
        UMI=10000
        NUM_THREADS=1
        MAX_JOBS=100
        ;;

    souporcell)
        SAVE_DIR="sweeps/simulated_sorc3"
        METHOD_FLAG="--souporcell"
        DATA_DIR="sim_data/G22_20k_runs"
        UMI=10000
        NUM_THREADS=60
        MAX_JOBS=3
        ;;

    vireo)
        SAVE_DIR="sweeps/simulated_vireo"
        METHOD_FLAG="--vireo"
        DATA_DIR="sim_data/G22_20k_runs"
        UMI=10000
        NUM_THREADS=50
        MAX_JOBS=2
        ;;

    demuxHMM2500)
        SAVE_DIR="sweeps/simulated_demuxHMM_umi_2500/"
        METHOD_FLAG="--demuxHMM"
        DATA_DIR="sim_data/G22_20k_runs_2500_UMI"
        UMI=2500
        NUM_THREADS=3
        MAX_JOBS=1
        ;;

    scsplit2500)
        SAVE_DIR="sweeps/simulated_scsplit_umi_2500"
        METHOD_FLAG="--scsplit"
        DATA_DIR="sim_data/G22_20k_runs_2500_UMI"
        UMI=2500
        NUM_THREADS=1
        MAX_JOBS=100
        ;;

    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

for run_num in {0..5}; do
    for num_emb in 10 25 50 75 100 150 200 250 350 500; do

      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
          wait -n
      done

      python simulate_fly_breeding.py \
          --data_path "$DATA_DIR/run_$run_num/num_emb_{$num_emb}_num_gen_{$num_gen}_avg_UMI_{$UMI}_num_cells_{$((cell_budget / num_emb))}.h5ad" \
          --save_dir "$SAVE_DIR/run_$run_num/" \
          --run_num "$i" \
          --num_repeats 30 \
          --num_generations "$num_gen" \
          --avg_UMI "$UMI" \
          --num_cells_per_org $((cell_budget / num_emb)) \
          --offspring_per_generation "$num_emb" \
          --num_threads "$NUM_THREADS" \
          "$METHOD_FLAG" &

        ((i++))

    done
done

wait