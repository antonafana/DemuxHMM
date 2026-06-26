# PBMC Pre-processing

This folder contains a dump of some of the code used to run cellsnp calling. This may be interesting
to users who want to do this efficiently, as we split up the calling for each chromosome. Data can be downloaded
from NCBI here: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583). We recommend using [sra-tools](https://github.com/ncbi/sra-tools) for fetching the data. We essentially only use sample C, which contains all the pooled individuals. For cellsnp, you'll want to
grab the AF5e-4 variant list [here](https://sourceforge.net/projects/cellsnp/files/SNPlist/). Finally, you'll want the 
demuxlet runs from [here](https://github.com/yelabucsf/demuxlet_paper_code/tree/master/fig2).