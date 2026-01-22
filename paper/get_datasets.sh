# Download fly cell atlas head and body scRNA-seq datasets
wget -P data_dir/ https://cloud.flycellatlas.org/index.php/s/RAGbG59qwaLKqEF/download?path=%2F&files=v2_fca_biohub_head_10x_raw.h5ad
wget -P data_dir/ https://cloud.flycellatlas.org/index.php/s/RAGbG59qwaLKqEF/download?path=%2F&files=v2_fca_biohub_body_10x_raw.h5ad

# Unzip the provided mappings of genes between references and the SNPs between strains
# Also contains the processed gene to SNP maps for those who wish to skip the pre-processing step
unzip data_dir/snp_and_gene_maps.zip -d data_dir

# Download demuxlet results from their paper repo
wget -P data_dir/ https://raw.githubusercontent.com/yelabucsf/demuxlet_paper_code/refs/heads/master/fig2/jy-c.seqaggr.eagle.10.sm.best