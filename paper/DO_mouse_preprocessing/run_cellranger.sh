#!/usr/bin/env bash
set -euo pipefail

FASTQ_DIR="fastq"
MAP="SRR_pools_map.txt"
REF="references/refdata-gex-GRCm39-2024-A"
OUTDIR="cellranger"

LOCAL_CORES=32
LOCAL_MEM=60
MAX_JOBS=4

mkdir -p "$OUTDIR"

# Extract unique sample names
cut -f2 "$MAP" | sort -u | while read -r SAMPLE; do

    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n
    done

    (
        cellranger count \
            --id="$SAMPLE" \
            --fastqs="$FASTQ_DIR" \
            --sample="$SAMPLE" \
            --transcriptome="$REF" \
            --localcores="$LOCAL_CORES" \
            --localmem="$LOCAL_MEM"

        mv "$SAMPLE/outs/possorted_genome_bam.bam" \
           "$OUTDIR/${SAMPLE}.bam"

        mv "$SAMPLE/outs/possorted_genome_bam.bam.bai" \
           "$OUTDIR/${SAMPLE}.bam.bai"

        mv "$SAMPLE/outs/filtered_feature_bc_matrix/barcodes.tsv.gz" \
           "$OUTDIR/${SAMPLE}.barcodes.tsv.gz"

        gzip -d "$OUTDIR/${SAMPLE}.barcodes.tsv.gz"

        rm -rf "$SAMPLE"

    ) &

done

wait