#!/usr/bin/env bash
set -euo pipefail

FASTQ_DIR="fastq"
MAP="SRR_pools_map.txt"

while read -r SRR SAMPLE LANE; do
    # Skip blank lines
    [[ -z "$SRR" ]] && continue

    R1_IN="${FASTQ_DIR}/${SRR}_2.fastq.gz"
    R2_IN="${FASTQ_DIR}/${SRR}_3.fastq.gz"

    R1_OUT="${FASTQ_DIR}/${SAMPLE}_S1_${LANE}_R1_001.fastq.gz"
    R2_OUT="${FASTQ_DIR}/${SAMPLE}_S1_${LANE}_R2_001.fastq.gz"

    if [[ ! -f "$R1_IN" ]]; then
        echo "Missing $R1_IN"
        continue
    fi

    if [[ ! -f "$R2_IN" ]]; then
        echo "Missing $R2_IN"
        continue
    fi

    echo "Renaming:"
    echo "  $R1_IN -> $R1_OUT"
    echo "  $R2_IN -> $R2_OUT"

    mv "$R1_IN" "$R1_OUT"
    mv "$R2_IN" "$R2_OUT"

done < "$MAP"