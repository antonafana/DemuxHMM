#!/usr/bin/env bash
set -euo pipefail

VCF_DIR="references"
BASE_VCF="${VCF_DIR}/mgp_REL2021_snps.vcf.gz"
RENAMED_VCF="${VCF_DIR}/mgp_REL2021_snps.chr.vcf.gz"

# Mapping from VCF chromosome names to BAM chromosome names
cat > "${VCF_DIR}/chr_map.txt" <<EOF
1   chr1
2   chr2
3   chr3
4   chr4
5   chr5
6   chr6
7   chr7
8   chr8
9   chr9
10  chr10
11  chr11
12  chr12
13  chr13
14  chr14
15  chr15
16  chr16
17  chr17
18  chr18
19  chr19
X   chrX
EOF

# Rename chromosome names
bcftools annotate \
    --rename-chrs "${VCF_DIR}/chr_map.txt" \
    -Oz \
    -o "$RENAMED_VCF" \
    "$BASE_VCF"

tabix -p vcf "$RENAMED_VCF"

# Split by chromosome
for chrom in {1..19} X
do
(
    bcftools view \
        -t "chr${chrom}" \
        "$RENAMED_VCF" \
        -Oz \
        -o "${VCF_DIR}/chromosome_chr${chrom}_snps.vcf.gz"

    tabix -p vcf "${VCF_DIR}/chromosome_chr${chrom}_snps.vcf.gz"
) &
done

wait