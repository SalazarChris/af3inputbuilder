# af3_builder/msa.py
import gzip
import lzma
import os
from typing import List, Union

class A3MFormatter:
    """Validate and normalize A3M MSAs."""

    @staticmethod
    def normalize_a3m(a3m_lines: List[str], query_sequence: str) -> List[str]:
        if not a3m_lines:
            raise ValueError("A3M lines cannot be empty")

        sequences = [line.strip() for line in a3m_lines if line.strip()]

        seq0 = sequences[0].replace('.', '-')
        if seq0.upper() != query_sequence.upper():
            raise ValueError("First A3M sequence must match the query sequence")

        query_len = len(query_sequence)
        for seq in sequences:
            seq_no_insertions = "".join(c for c in seq if c.isupper() or c == "-")
            if len(seq_no_insertions) != query_len:
                raise ValueError(
                    "All sequences must match query length ignoring insertions"
                )

        return sequences

    @staticmethod
    def load_a3m_file(path: str) -> List[str]:
        if not os.path.isfile(path):
            raise ValueError(f"Provided path is not a file: {path}")

        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                return f.readlines()
        elif path.endswith(".xz"):
            with lzma.open(path, "rt") as f:
                return f.readlines()
        else:
            with open(path, "r") as f:
                return f.readlines()


class MSAValidator:
    """Validate paired and unpaired MSA fields for a sequence entity."""

    @staticmethod
    def validate_msa_fields(
        unpaired_msa: Union[str, None],
        paired_msa: Union[str, None],
    ):
        if unpaired_msa and paired_msa:
            return True
        if not unpaired_msa and not paired_msa:
            return True
        raise ValueError(
            "Invalid MSA combination: unpairedMsa and pairedMsa must be both set or both unset"
        )


def prompt_for_a3m_file() -> str:
    """Prompt user for an A3M file path."""
    path = input("Enter path to A3M file (.a3m, .gz, or .xz): ").strip()

    if not os.path.isfile(path):
        raise ValueError(f"Not a valid file: {path}")

    return path


def validate_msa_sequence(path: str, expected_sequence: str) -> bool:
    """Check if the first sequence in an A3M file matches the expected sequence."""
    try:
        lines = A3MFormatter.load_a3m_file(path)
        # Extract sequences (lines not starting with >)
        sequences = [line.strip() for line in lines if line.strip() and not line.startswith(">")]
        if not sequences:
            return False
        # Remove dots/gaps for comparison. 
        # A3M query sequence is usually the first one, all caps/dots replaced with dashes.
        first_seq = "".join(c for c in sequences[0] if c.isupper() or c == "-").replace("-", "")
        clean_expected = expected_sequence.upper().replace("-", "")
        return first_seq == clean_expected
    except Exception:
        return False


if __name__ == "__main__":
    a3m_path = prompt_for_a3m_file()
    lines = A3MFormatter.load_a3m_file(a3m_path)
    print(f"Loaded {len(lines)} lines from {a3m_path}")
