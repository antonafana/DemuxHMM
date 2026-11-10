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
        SAVE_DIR="sweeps/DO_mouse/demuxHMM_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--demuxHMM"
        NUM_THREADS=20
        ;;

    souporcell)
        SAVE_DIR="sweeps/DO_mouse/sorc3_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--souporcell"
        NUM_THREADS=60
        ;;

    vireo)
        SAVE_DIR="sweeps/DO_mouse/vireo_doublets_${DOUBLET_DETECT}/"
        METHOD_FLAG="--vireo"
        NUM_THREADS=25
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

python DO_mouse_demux.py \
    --save_dir "$SAVE_DIR" \
    --num_threads "$NUM_THREADS" \
    --TOL 250000 \
    --num_repeats 60 \
    "$METHOD_FLAG" \
    "${ARGS[@]}"