# af3_builder/__init__.py
"""
af3_builder  -  Core building-blocks for AlphaFold 3 input JSON.

Public API
----------
JobBuilder          - Assemble & serialise AF3 job dicts incrementally.
ProteinEntity, RNAEntity, DNAEntity, LigandEntity
                    - Typed entity objects with built-in validation.
AF3Validator        - Schema/structural validator.
ValidationError     - Exception raised on validation failure.
IDManager           - Chain-ID allocation helper.
SeedsHelper         - Seed generation & validation.
load_json / save_json / autosave_json
                    - Atomic I/O with InlineArray JSON formatting.
build_template_block / run_templater
                    - mmCIF template index computation.
reverse_complement  - 5'-3' reverse complement for DNA/RNA.
ui                  - Shared terminal UI helpers (colours, prompts, menus).
"""

from .core.job      import JobBuilder
from .core.entities import (
    ProteinEntity, RNAEntity, DNAEntity, LigandEntity,
    reverse_complement, slugify, next_spreadsheet_id,
    PROTEIN_ALPHABET, RNA_ALPHABET, DNA_ALPHABET,
)
from .validation.validator  import AF3Validator, ValidationError
from .core.id_manager import IDManager
from .utils.io         import load_json, save_json, autosave_json
from .utils.json_inline import InlineArrayEncoder
from .core.seeds      import SeedsHelper
from .core.reference  import (
    COMMON_IONS, COMMON_COFACTORS, COMMON_SMALL_MOLECULES, ALL_COMMON_LIGANDS,
    PROTEIN_PTMS, RNA_MODIFICATIONS, DNA_MODIFICATIONS,
)
from .utils.templater  import build_template_block, run_templater
from .ui.ui            import (
    RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN, BLUE, MAG, TW,
    _rule, _banner, _section, _ok, _warn, _err, _info, _tip, _divider,
    _ask, _ask_yn, _choose, _pause, _ask_file, _ask_dir, _ask_file_or_dir, _pick_file, _pick_dir,
)
from .ui.interactive   import (
    add_protein_wizard, add_rna_wizard, add_dna_wizard, add_ligand_wizard, add_common_ions_wizard,
    edit_protein_wizard, edit_rna_wizard, edit_dna_wizard, edit_ligand_wizard,
    manage_modifications_wizard, manage_templates_wizard,
    quick_delete_entity_wizard, manage_bonded_atom_pairs_wizard,
    manage_user_ccd_wizard, strip_entities_wizard,
    show_job_summary, show_help_text,
    _current_ids, _next_letter
)
