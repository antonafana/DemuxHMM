#!/usr/bin/env bash
source ../py314_venv/bin/activate
export PYTHONUNBUFFERED=1
export NUMBA_CUDA_USE_NVIDIA_BINDING=0

if [ $# -ne 2 ]; then
    echo "Usage: $0 {demuxHMM|souporcell|vireo} {True|False}"
    exit 1
fi

METHOD="$1"
DOUBLET_DETECT="$2"

case "$DOUBLET_DETECT" in
    True|False)
        ;;
    *)
        echo "Second argument must be True or False and indicated whether we run with doublets (and detection)."
        exit 1
        ;;
esac

case "$METHOD" in
    demuxHMM)
        SAVE_DIR="sweeps/DO_mouse_downsample/demuxHMM_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--demuxHMM"
        NUM_THREADS=20
        REPEATS=5
        MAX_JOBS=1
        ;;

    souporcell)
        SAVE_DIR="sweeps/DO_mouse_downsample/sorc3_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--souporcell"
        NUM_THREADS=60
        REPEATS=3
        MAX_JOBS=2
        ;;

    vireo)
        SAVE_DIR="sweeps/DO_mouse_downsample/vireo_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--vireo"
        NUM_THREADS=10
        REPEATS=5
        MAX_JOBS=1
        ;;

    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

ARGS=()

if [ "$DOUBLET_DETECT" = "True" ]; then
    ARGS+=(--detect_doublets --doublet_rate 0.05)
fi

i=0
MAX_REPEAT_IND=$((REPEATS - 1))
REPEAT_ARRAY=($(seq 0 $MAX_REPEAT_IND))

for data_repeat in "${REPEAT_ARRAY[@]}"; do
  for downsample_fraction in 0.1 0.25 0.5 1; do

      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
          wait -n
      done

    if [ "$downsample_fraction" = 1 ]; then
      data_path="data_dir/DO_mouse_cellsnp_001.h5ad"
    else
      data_path="data_dir/downsampled_mouse/do_mouse_${downsample_fraction}_reads_repeat_${data_repeat}.h5ad"
    fi

   # When we are not downsampling, we do not want sorc to be deterministic as neither the data nor parameters change
   SEED_ARGS=()
   if [ "$METHOD" = "souporcell" ] && [ "$downsample_fraction" = 1 ]; then
       SEED_ARGS+=(--sorc_seed "$data_repeat")
   fi

    python DO_mouse_demux.py \
        --save_dir "$SAVE_DIR/ds_frac_${downsample_fraction}/run_${data_repeat}/" \
        --data_path "$data_path" \
        --num_threads "$NUM_THREADS" \
        --TOL 250000 \
        --num_repeats 240 \
        --run_num "$i" \
        "$METHOD_FLAG" \
        "${ARGS[@]}"  \
       "${SEED_ARGS[@]}" &
    ((i++))

  done
done

wait