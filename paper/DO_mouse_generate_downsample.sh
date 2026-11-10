#!/usr/bin/env bash
source ../py314_venv/bin/activate

MAX_JOBS=1

for repeat in {0..4}; do
    for downsample_fraction in 0.1 0.25 0.5; do

        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            wait -n
        done

        python DO_mouse_downsample.py \
            --save_path "data_dir/downsampled_mouse/do_mouse_${downsample_fraction}_reads_repeat_${repeat}.h5ad" \
            --downsample_fraction "$downsample_fraction" &

    done
done

wait