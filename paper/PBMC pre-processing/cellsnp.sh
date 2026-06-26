#!/bin/bash
eval "$(anaconda3/bin/conda shell.bash hook)"
conda activate cellsnp;

vcf_dir=PBMC/chr_vcfs/

in_bam=$1
out_dir=$2
barcodes=$3

for chr in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X; do
        mkdir -p $out_dir/$chr
        nohup cellsnp-lite -s $in_bam -b $barcodes -O $out_dir/$chr --chrom $chr --cellTAG CB --UMItag UB -R $vcf_dir/${chr}.vcf.gz -p 8 --minMAF 0.1 --minCOUNT 20 --gzip &
done

