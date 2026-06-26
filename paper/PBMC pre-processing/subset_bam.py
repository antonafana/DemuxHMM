import pysam
import argparse

def load_barcodes(barcode_file):
    with open(barcode_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def subset_bam(input_bam, barcode_file, output_bam):
    barcodes = load_barcodes(barcode_file)
    
    with pysam.AlignmentFile(input_bam, "rb") as bam_in, \
         pysam.AlignmentFile(output_bam, "wb", template=bam_in) as bam_out:
        
        for read in bam_in:
            cb = read.get_tag('CB') if read.has_tag('CB') else None
            if cb in barcodes:
                bam_out.write(read)

    pysam.index(output_bam)

def main():
    parser = argparse.ArgumentParser(description="Subset BAM file by barcode list")
    parser.add_argument("input_bam", help="Input BAM file")
    parser.add_argument("barcode_file", help="Text file with one barcode per line")
    parser.add_argument("output_bam", help="Output BAM file")

    args = parser.parse_args()
    subset_bam(args.input_bam, args.barcode_file, args.output_bam)

if __name__ == "__main__":
    main()
