#!/usr/bin/env bash
set -euo pipefail

INPUT=cellranger_unique
OUTPUT=chromosomes

MAX_JOBS=20
THREADS=4

CHROMS=(
    chr1 chr2 chr3 chr4 chr5
    chr6 chr7 chr8 chr9 chr10
    chr11 chr12 chr13 chr14 chr15
    chr16 chr17 chr18 chr19 chrX
)

mkdir -p "$OUTPUT"

# Split existing bams up by chromosome for parallelization
for BAM in "$INPUT"/*.bam; do
    SAMPLE=$(basename "$BAM" .bam)

    for CHR in "${CHROMS[@]}"; do

        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            wait -n
        done

        (
            mkdir -p "$OUTPUT/$CHR"

            samtools view \
                -@ "$THREADS" \
                -b \
                -o "$OUTPUT/$CHR/${SAMPLE}.bam" \
                "$BAM" \
                "$CHR"

            samtools index \
                -@ "$THREADS" \
                "$OUTPUT/$CHR/${SAMPLE}.bam"
        ) &
    done
done

wait