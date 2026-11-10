mkdir -p sra
cd sra

while read SRR; do
    prefetch -p "$SRR"
done < ../SRR_list.txt

mkdir -p ../fastq

MAX_JOBS=5
NUM_CORES=32

for SRR in $(cat ../SRR_list.txt); do
    # Block until fewer than MAX_JOBS are running
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
      wait -n
    done

    (   fasterq-dump \
            --include-technical \
            --split-files \
            -e "$NUM_CORES" \
            -O ../fastq \
            "$SRR" && \
        pigz -p "$NUM_CORES" ../fastq/${SRR}_*.fastq
    ) &
done

wait

# Clean up the sample index / i7 files
rm ../fastq/*_1.fastq.gz

# Clean up any remaining fastq files (this can occur if you've already done a partial run without cleaning up)
rm ../fastq/*.fastq