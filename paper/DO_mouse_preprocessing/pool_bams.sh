#!/usr/bin/env bash
set -euo pipefail

INPUT=chromosomes

MAX_JOBS=20
THREADS=8

for CHRDIR in "$INPUT"/chr*; do

    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n
    done

    (
        # Merge all the bams in each chromosome folder
        samtools merge \
            -@ "$THREADS" \
            -f \
            "$CHRDIR/merged.bam" \
            "$CHRDIR"/*.bam

        samtools index \
            -@ "$THREADS" \
            "$CHRDIR/merged.bam"

        # Remove the individual sample bams
        find "$CHRDIR" -maxdepth 1 -type f \
              \( -name "*.bam" -o -name "*.bam.bai" \) \
              ! -name "merged.bam" \
              ! -name "merged.bam.bai" \
              -delete

    ) &
done

wait