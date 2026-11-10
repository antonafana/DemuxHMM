# Pre-processing for the DO mice dataset

We want to use [this dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269583) on GEO for demultiplexing. It has 80 total mice embryos captured from Diversity Outbreeding (DO) mice. The DO scheme has similar characteristics to our proposed simple scheme in DemuxHMM. It uses 8 diverse mice and 30+ generations of breeding.

To start, download the overall metadata file from the SRA run selector for [the project](https://www.ncbi.nlm.nih.gov/Traces/study/?acc=PRJNA1122572&o=acc_s%3Aa). Next, read the csv into python and do post-processing to pull all SRR names and identify which should be pooled (contain the same mouse ids). Run this with
```bash
python curate_metadata.py 
```
Next, start downloading and dumping data. This may take a while...
```bash
sh sra_fetch.sh
```
After, we'll need to map reads using Cell Ranger. For this, we'll need to [download the reference genome](https://www.10xgenomics.com/support/software/cell-ranger/downloads#reference-downloads), rename files into the cellranger format, and finally perform the mapping.
```bash
wget "https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCm39-2024-A.tar.gz"
tar -zxvf refdata-gex-GRCm39-2024-A.tar.gz
sh rename_to_cellranger_format.sh
sh run_cellranger.sh
```

Now, prepare to call variants. First, download the variant files from [Jackson Lab](https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data#_Genomic_Variants). Using pre-identified sites will speed up calling significantly, and is advisable for datasets where we have a good reference like this. These VCFs are already fairly well filtered by allele frequency. After, we need to rename the chromosomes (to match the bam files) on the vcf and then split it by chromosome for more parallelizable variant calling.
```bash
mkdir -p references && cd references
wget https://ftp.ebi.ac.uk/pub/databases/mousegenomes/REL-2112-v8-SNPs_Indels/mgp_REL2021_snps.vcf.gz
wget https://ftp.ebi.ac.uk/pub/databases/mousegenomes/REL-2112-v8-SNPs_Indels/mgp_REL2021_snps.vcf.gz.csi
cd ..
sh split_vcf.sh
```
First we deduplicate barcodes (multiple pools can share a barcode). We also need to split our bam files up by chromosome and then merge all the smaller pools (by chromosome).
```bash
python dedup_barcodes.py
sh split_bam.sh
sh pool_bams.sh
```

Finally, we should be ready for variant calling. First make a set of merged cell barcodes and then start variant calling.
```bash
cat cellranger_unique/*.tsv >> all_barcodes.tsv
sh cellsnp.sh
```
Then, merge everything into one anndata with
```bash
python merge_var_calls.py
```
This script will also do basic filtering for bad cells or non-useful variants.