# af3_builder/templater.py
"""
Utilities for computing AlphaFold3 template indices from an mmCIF file.

Public API
----------
run_templater(query_seq, cif_path, chain_id)
    -> (query_indices, template_indices, alignment_score, coverage_pct)

build_template_block(cif_path, query_seq, chain_id)
    -> dict  ready to append to ProteinEntity(templates=[...])

These functions can be imported freely; no side-effects occur on import.
Run this module directly (`python -m af3_builder.templater`) to use the
stand-alone report generator.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple

# Bio imports moved inside functions to allow optional usage

def _extract_chain_polymer_seq(cif_path: str, chain_id: str) -> str:
    """
    Return the template chain *polymer sequence* in 1-letter codes.

    This uses mmCIF polymer sequence tables (e.g. pdbx_poly_seq_scheme),
    so unresolved residues are included (as required by AF3).
    """
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        from Bio.PDB.Polypeptide import protein_letters_3to1
    except ImportError:
        raise ImportError("Biopython is required for template sequence extraction. Please install it with: pip install biopython")
    
    mmcif = MMCIF2Dict(cif_path)

    # These fields are typically present in modern mmCIFs.
    # We match either pdb_strand_id (author chain) or asym_id (internal).
    strand = mmcif.get("_pdbx_poly_seq_scheme.pdb_strand_id", [])
    asym = mmcif.get("_pdbx_poly_seq_scheme.asym_id", [])
    seq_id = mmcif.get("_pdbx_poly_seq_scheme.seq_id", [])
    mon_id = mmcif.get("_pdbx_poly_seq_scheme.mon_id", [])

    if not mon_id or not seq_id:
        raise ValueError("mmCIF missing _pdbx_poly_seq_scheme fields needed for polymer sequence extraction")

    # Find rows for this chain
    rows = []
    for i in range(len(mon_id)):
        sid = (strand[i] if i < len(strand) else None)
        aid = (asym[i] if i < len(asym) else None)
        if sid == chain_id or aid == chain_id:
            try:
                rows.append((int(seq_id[i]), mon_id[i].upper()))
            except Exception:
                continue

    if not rows:
        raise ValueError(f"No polymer sequence rows found for chain '{chain_id}' in pdbx_poly_seq_scheme")

    rows.sort(key=lambda x: x[0])  # seq_id order (1..N)

    # Build sequence; keep unknowns as 'X' rather than skipping, to preserve indexing
    seq = []
    for _, three in rows:
        aa = protein_letters_3to1.get(three, "X")
        seq.append(aa)
    return "".join(seq)

def _align(query_seq: str, template_seq: str) -> Tuple[List[int], List[int], float, float]:
    """
    Global pairwise alignment of *query_seq* vs *template_seq*.

    Returns
    -------
    query_indices, template_indices : List[int]
        0-based positions of matched residues (gaps excluded).
    score : float
        Raw alignment score.
    coverage : float
        Percentage of the query covered by aligned residues.
    """
    try:
        from Bio import Align
    except ImportError:
        raise ImportError("Biopython is required for template alignment. Please install it with: pip install biopython")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    best = aligner.align(query_seq, template_seq)[0]
    score = best.score

    query_indices: List[int] = []
    template_indices: List[int] = []

    for q_idx, t_idx in zip(*best.indices):
        if q_idx != -1 and t_idx != -1:
            query_indices.append(int(q_idx))
            template_indices.append(int(t_idx))

    coverage = (len(query_indices) / len(query_seq) * 100) if query_seq else 0.0
    return query_indices, template_indices, score, coverage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_templater(query_seq: str, cif_path: str, chain_id: str):
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    # Check for slicing
    final_cif_path = cif_path
    try:
        from .cif_slicer import check_multiple_chains, slice_mmcif_chain
        if check_multiple_chains(cif_path):
            final_cif_path = slice_mmcif_chain(cif_path, chain_id)
    except ImportError:
        pass # gemmi not available, proceed with original file

    template_seq = _extract_chain_polymer_seq(final_cif_path, chain_id)

    if not template_seq:
        raise ValueError(f"Empty polymer sequence for chain '{chain_id}' in {final_cif_path}")

    q_idx, t_idx, score, coverage = _align(query_seq, template_seq)

    # IMPORTANT: AF3 expects templateIndices to be indices in the template sequence
    # (0..N-1), so we return t_idx as-is.
    return q_idx, t_idx, score, coverage, final_cif_path


def build_template_block(
    cif_path: str,
    query_seq: str,
    chain_id: str,
) -> dict:
    """
    Build a ready-to-use AF3 template dict for one chain.

    The returned dict can be passed directly inside
    ``ProteinEntity(templates=[build_template_block(...)])``.

    Example
    -------
    >>> block = build_template_block("8KDN.cif", my_sequence, "A")
    >>> prot = ProteinEntity(id="A", sequence=my_sequence, templates=[block])
    """
    q_idx, t_idx, _, _, final_path = run_templater(query_seq, cif_path, chain_id)
    return {
        "mmcifPath": os.path.basename(final_path), # Use basename per earlier improvements
        "queryIndices": q_idx,
        "templateIndices": t_idx,
    }


# ---------------------------------------------------------------------------
# Stand-alone report (run with: python -m af3_builder.templater)
# ---------------------------------------------------------------------------

# --- Configuration (EDIT THESE when running as a script) ---
_QUERY_SEQ_A = "MELQPPEASIAVVSIPRQLPGSHSEAGVQGLSAGDDSELGSHCVAQTGLELLASGDPLPSASQNAEMIETGSDCVTQAGLQLLASSDPPALASKNAEVTGTMSQDTEVDMKEVELNELEPEKQPMNAASGAAMSLAGAEKNGLVKIKVAEDEAEAAAAAKFTGLSKEELLKVAGSPGWVRTRWALLLLFWLGWLGMLAGAVVIIVRAPRCRELPAQKWWHTGALYRIGDLQAFQGHGAGNLAGLKGRLDYLSSLKVKGLVLGPIHKNQKDDVAQTDLLQIDPNFGSKEDFDSLLQSAKKKSIRVILDLTPNYRGENSWFSTQVDTVATKVKDALEFWLQAGVDGFQVRDIENLKDASSFLAEWQNITKGFSEDRLLIAGTNSSDLQQILSLLESNKDLLLTSSYLSDSGSTGEHTKSLVTQYLNATGNRWCSWSLSQARLLTSFLPAQLLRLYQLMLFTLPGTPVFSYGDEIGLDAAALPGQPMEAPVMLWDESSFPDIPGAVSANMTVKGQSEDPGSLLSLFRRLSDQRSKERSLLHGDFHAFSAGPGLFSYIRHWDQNERFLVVLNFGDVGLSAGLQASDLPASASLPAKADLLLSTQPGREEGSPLELERLKLEPHEGLLLRFPYAA"
_QUERY_SEQ_B = "MAGAGPKRRALAAPAAEEKEEAREKMLAAKSADGSAPAGEGEGVTLQRNITLLNGVAIIVGTIIGSGIFVTPTGVLKEAGSPGLALVVWAACGVFSIVGALCYAELGTTISKSGGDYAYMLEVYGSLPAFLKLWIELLIIRPSSQYIVALVFATYLLKPLFPTCPVPEEAAKLVACLCVLLLTAVNCYSVKAATRVQDAFAAAKLLALALIILLGFVQIGKGDVSNLDPNFSFEGTKLDVGNIVLALYSGLFAYGGWNYLNFVTEEMINPYRNLPLAIIISLPIVTLVYVLTNLAYFTTLSTEQMLSSEAVAVDFGNYHLGVMSWIIPVFVGLSCFGSVNGSLFTSSRLFFVGSREGHLPSILSMIHPQLLTPVPSLVFTCVMTLLYAFSKDIFSVINFFSFFNWLCVALAIIGMIWLRHRKPELERPIKVNLALPVFFILACLFLIAVSFWKTPVECGIGFTIILSGLPVYFFGVWWKNKPKWLLQGIFSTTVLCQKLMQVVPQET"
_TEMPLATE_FILE = "8KDN.cif"
_CHAINS = [
    (_QUERY_SEQ_A, "A"),
    (_QUERY_SEQ_B, "B"),
]
_OUTPUT_FILE = "af3_template_indices.txt"
# -----------------------------------------------------------


def _run_report(chains, cif_path: str, output_file: str):
    """Generate and save the human-readable alignment report."""
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "=" * 70,
        "AlphaFold3 Template Indices Generation Report",
        f"Generated: {timestamp}",
        f"Template File: {cif_path}",
        "=" * 70,
        "",
    ]

    for query_seq, chain_id in chains:
        label = f"CHAIN {chain_id}"
        lines.append(f"--- {label} ALIGNMENT RESULTS ---")
        try:
            q_idx, t_idx, score, coverage, new_path = run_templater(query_seq, cif_path, chain_id)
            template_seq = _extract_chain_polymer_seq(new_path, chain_id)
            lines += [
                f"Template Chain ID: {chain_id}",
                f"Query Sequence Length: {len(query_seq)}",
                f"Template Sequence Length: {len(template_seq)}",
                f"Aligned Residues: {len(q_idx)}",
                f"Alignment Score: {score:.2f}",
                f"Coverage: {coverage:.2f}%",
                f"Sliced Template Saved As: {new_path}",
                "",
                f"Query Indices ({len(q_idx)} positions):",
                str(q_idx),
                "",
                f"Template Indices ({len(t_idx)} positions):",
                str(t_idx),
                "",
                "=" * 70,
                "",
                "--- ALPHAFOLD3 JSON TEMPLATE BLOCK ---",
                f"// Place inside the 'templates' array of your Chain {chain_id} protein block:",
                "{",
                f'    "mmcifPath": "{os.path.abspath(cif_path)}",',
                f'    "queryIndices": {q_idx},',
                f'    "templateIndices": {t_idx}',
                "}",
            ]
        except Exception as exc:
            lines.append(f"ERROR processing chain {chain_id}: {exc}")

        lines += ["", "=" * 70, ""]

    for line in lines:
        print(line)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    if not os.path.exists(_TEMPLATE_FILE):
        print(f"Error: template file not found: {_TEMPLATE_FILE}")
    else:
        _run_report(_CHAINS, _TEMPLATE_FILE, _OUTPUT_FILE)
