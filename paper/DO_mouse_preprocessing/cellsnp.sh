#!/usr/bin/env bash
set -euo pipefail

eval "$(/mnt/geofflab/anaconda3/bin/conda shell.bash hook)"
conda activate cellsnp;

BAM_DIR="chromosomes"
VCF_DIR="references"
OUT_DIR="var_calls"

CHROMS=(
    chr1 chr2 chr3 chr4 chr5
    chr6 chr7 chr8 chr9 chr10
    chr11 chr12 chr13 chr14 chr15
    chr16 chr17 chr18 chr19 chrX
)

mkdir -p "$OUT_DIR"

for chr in "${CHROMS[@]}";
do
        mkdir -p "${OUT_DIR}/${chr}"
        cellsnp-lite -s "${BAM_DIR}/${chr}/merged.bam" -b all_barcodes.tsv -O "${OUT_DIR}/${chr}" \
        --chrom "$chr" --cellTAG CB --UMItag UB -R "${VCF_DIR}/chromosome_${chr}_snps.vcf.gz" \
        -p 8 --minMAF 0.1 --minCOUNT 20 --gzip &
done

wait