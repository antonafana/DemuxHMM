#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

if [ $# -ne 1 ]; then
    echo "Usage: $0 {mouse|simulated}"
    exit 1
fi

DATASET="$1"

case "$DATASET" in
    simulated)
        cell_budget=20000
        num_gen=22
        i=0
        SAVE_DIR="sweeps/clusters/demuxHMM_simulated/"
        DATA_DIR="sim_data/G22_20k_runs"
        UMI=10000
        NUM_THREADS=3
        MAX_JOBS=1
        num_emb=50

        for run_num in {0..5}; do
          for clusters in 10 20 30 40 50 60 70 80 90 100; do

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
                --num_clusters "$clusters" \
                --demuxHMM &

              ((i++))

          done
        done

        wait
        ;;

    mouse)
      NUM_THREADS=20
      SAVE_DIR="sweeps/clusters/demuxHMM_mouse/run_0/"
      i=20

      # 10 20 30 40 50 60 70 77 80 90 100 110
      for clusters in  120 130 140 150 160; do

        python DO_mouse_demux.py \
          --save_dir "$SAVE_DIR" \
          --num_threads "$NUM_THREADS" \
          --TOL 250000 \
          --num_repeats 30 \
          --num_clusters "$clusters" \
          --run_num "$i" \
          --demuxHMM

        ((i++))
      done


      ;;

    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac