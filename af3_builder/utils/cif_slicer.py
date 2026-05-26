import os
try:
    import gemmi
    _HAS_GEMMI = True
except ImportError:
    _HAS_GEMMI = False

def check_multiple_chains(cif_path: str) -> bool:
    """True if the CIF contains multiple chains in the scheme or atom tables."""
    if not _HAS_GEMMI:
        return False
    doc = gemmi.cif.read_file(cif_path)
    if len(doc) == 0: return False
    block = doc[0]
    
    seq_table = block.find('_pdbx_poly_seq_scheme.', ['pdb_strand_id'])
    chains = set()
    if seq_table:
        for i in range(len(seq_table)):
            chains.add(seq_table[i][0])
    
    if not chains:
        atom_table = block.find('_atom_site.', ['auth_asym_id'])
        if atom_table:
            for i in range(len(atom_table)):
                chains.add(atom_table[i][0])
                
    return len(chains) > 1

def slice_mmcif_chain(cif_path: str, target_chain: str, out_path: str = None) -> str:
    """
    Slices an mmCIF file down to a single chain by removing rows from
    _atom_site and _pdbx_poly_seq_scheme that do not match target_chain.
    This preserves the AF3-required sequence scheme metadata.
    """
    if not _HAS_GEMMI:
        raise ImportError("Gemmi is required for advanced CIF slicing.")

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    if not out_path:
        base, ext = os.path.splitext(cif_path)
        out_path = f"{base}_chain_{target_chain}{ext}"

    doc = gemmi.cif.read_file(cif_path)
    if len(doc) == 0:
        raise ValueError("Empty CIF file")
    
    block = doc[0]

    # Filter _pdbx_poly_seq_scheme
    seq_table = block.find('_pdbx_poly_seq_scheme.', ['pdb_strand_id'])
    if seq_table:
        for i in range(len(seq_table) - 1, -1, -1):
            if seq_table[i][0] != target_chain:
                seq_table.remove_row(i)

    # Filter _struct_asym
    asym_table = block.find('_struct_asym.', ['id'])
    if asym_table:
        for i in range(len(asym_table) - 1, -1, -1):
            if asym_table[i][0] != target_chain:
                asym_table.remove_row(i)

    # Filter _atom_site (try auth_asym_id first, then label_asym_id)
    atom_table = block.find('_atom_site.', ['auth_asym_id'])
    if atom_table:
        for i in range(len(atom_table) - 1, -1, -1):
            if atom_table[i][0] != target_chain:
                atom_table.remove_row(i)
    else:
        atom_table_label = block.find('_atom_site.', ['label_asym_id'])
        if atom_table_label:
            for i in range(len(atom_table_label) - 1, -1, -1):
                if atom_table_label[i][0] != target_chain:
                    atom_table_label.remove_row(i)

    doc.write_file(out_path)
    return out_path
