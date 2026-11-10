import multiprocessing as mp
from pathlib import Path
import pysam
"""
This script removes barcodes that are duplicated inter-pool by appending the pool name to each barcode.
"""

INPUT_DIR = Path("cellranger")
OUTPUT_DIR = Path("cellranger_unique")

# Number of BAMs to process simultaneously
N_PROCESSES = 16

OUTPUT_DIR.mkdir(exist_ok=True)


def process_pool(bam_file: Path):
    pool = bam_file.stem

    print(f"Processing {pool}", flush=True)

    out_bam = OUTPUT_DIR / bam_file.name

    with pysam.AlignmentFile(bam_file, "rb") as bam_in, \
         pysam.AlignmentFile(out_bam, "wb", header=bam_in.header) as bam_out:

        for read in bam_in:
            if read.has_tag("CB"):
                cb = read.get_tag("CB")
                read.set_tag("CB", f"{pool}_{cb}", value_type="Z")
            bam_out.write(read)

    pysam.index(str(out_bam))

    barcode_in = INPUT_DIR / f"{pool}.barcodes.tsv"
    barcode_out = OUTPUT_DIR / f"{pool}.barcodes.tsv"

    with open(barcode_in) as fin, open(barcode_out, "w") as fout:
        for line in fin:
            fout.write(f"{pool}_{line.strip()}\n")

    print(f"Finished {pool}", flush=True)


def main():
    bam_files = sorted(INPUT_DIR.glob("*.bam"))

    print(f"Found {len(bam_files)} BAMs")
    print(f"Running {N_PROCESSES} processes")

    with mp.Pool(N_PROCESSES) as pool:
        pool.map(process_pool, bam_files)


if __name__ == "__main__":
    main()