~/programs/cellranger-9.0.1/cellranger count --id=pbmc \
                 --transcriptome=refdata-gex-GRCh38-2024-A/ \
                 --fastqs=/mnt/geofflab/c_elegans/PBMC \
                 --localcores=100 \
                 --localmem=180 \
		 --create-bam true
