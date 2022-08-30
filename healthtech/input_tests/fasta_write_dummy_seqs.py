#!/Users/maghoi/opt/anaconda3/bin/python

import argparse
import logging

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description="""
    Creates dummy fasta file with n entries and sequence repeated i times

    Example:
    # Create FASTA file of 2 entries and 10 residues with only "A"
    python fasta_write_dummy_seqs.py --fasta_entries 2 --seq "A" --seq_repeats_per_entry 10

    # Creates 2_entries_10_residues.fasta
    >dummy1
    AAAAAAAAAA
    >dummy2
    AAAAAAAAAA
    """
    )

    p.add_argument(
        "--fasta_entries",
        required=True,
        type=int,
        help="Number of FASTA entries",
    )
    p.add_argument(
        "--seq",
        default="A",
        type=str,
        help="Sequence to duplicate (not implemented)",
    )
    p.add_argument(
        "--seq_repeats_per_entry",
        default=10,
        type=int,
        help="Number of residues per entry",
    )
    p.add_argument(
        "--outfile", required=False, default="dummy.fasta", help="Output directory"
    )

    return p.parse_args()

def main(
        entries: int = 10,
        seq: str = "A",
        repeats: int = 10):
    """ main """

    id = ">dummy"
    seq = str(seq) * repeats
    outname = f"{entries}_entries_{len(seq)}_residues.fasta"

    with open(outname, "w") as out_handle:
        for i in range(1, entries+1):
            id_line = id + str(i)
            seq_line = seq

            out_handle.write(id_line + "\n")
            out_handle.write(seq_line + "\n")

    log.info(f"DONE: Wrote {entries} FASTA entries to {outname}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
    log = logging.getLogger(__name__)

    args = cmdline_args()
    residue_count = len(str(args.seq) * args.seq_repeats_per_entry)
    log.info(f"Creating dummy FASTA of {args.fasta_entries} entries of length {residue_count} ...")

    # Main
    main(args.fasta_entries, args.seq, args.seq_repeats_per_entry)
