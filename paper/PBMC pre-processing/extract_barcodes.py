import pysam
import argparse

def extract_barcodes_and_samples(bam_path, barcodes_out, mapping_out):
    cell_barcodes = set()
    cb_to_bc = dict()

    with pysam.AlignmentFile(bam_path, "rb") as bamfile:
        for read in bamfile:
            if read.has_tag("CB") and read.has_tag("BC"):
                cb = read.get_tag("CB")
                bc = read.get_tag("BC")
                cell_barcodes.add(cb)

                # If a CB was already seen with a different BC, raise a warning
                if cb in cb_to_bc and cb_to_bc[cb] != bc:
                    print(f"Warning: Cell barcode {cb} has multiple sample barcodes: {cb_to_bc[cb]} and {bc}")
                cb_to_bc[cb] = bc

    # Write list of cell barcodes
    with open(barcodes_out, "w") as f:
        for cb in sorted(cell_barcodes):
            f.write(f"{cb}\n")

    # Write cell-to-sample mapping
    with open(mapping_out, "w") as f:
        for cb in sorted(cb_to_bc):
            bc = cb_to_bc[cb]
            f.write(f"{cb}\t{bc}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cell barcodes and their sample IDs from a BAM file.")
    parser.add_argument("bam", help="Input BAM file")
    parser.add_argument("--barcodes", default="cell_barcodes.txt", help="Output file for cell barcodes")
    parser.add_argument("--mapping", default="cell_to_sample.txt", help="Output file for cell-to-sample mapping")
    args = parser.parse_args()

    extract_barcodes_and_samples(args.bam, args.barcodes, args.mapping)

