# af3_builder/core/reference.py
"""
Centralized reference lists for CCD codes, PTMs, and common ligands.
All wizards and scripts import from here — single source of truth.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IONS & SMALL MOLECULES (CCD codes)
# ═══════════════════════════════════════════════════════════════════════════════

COMMON_IONS = [
    # Divalent cations
    ("CA",  "Ca²⁺  — Calcium"),
    ("MG",  "Mg²⁺  — Magnesium"),
    ("ZN",  "Zn²⁺  — Zinc"),
    ("FE",  "Fe²⁺  — Iron (ferrous)"),
    ("FE2", "Fe²⁺  — Iron (ferrous, alt CCD)"),
    ("MN",  "Mn²⁺  — Manganese"),
    ("CO",  "Co²⁺  — Cobalt"),
    ("CU",  "Cu²⁺  — Copper"),
    ("NI",  "Ni²⁺  — Nickel"),
    ("CD",  "Cd²⁺  — Cadmium"),
    ("BA",  "Ba²⁺  — Barium"),
    ("SR",  "Sr²⁺  — Strontium"),
    # Monovalent cations
    ("NA",  "Na⁺   — Sodium"),
    ("K",   "K⁺    — Potassium"),
    ("LI",  "Li⁺   — Lithium"),
    ("RB",  "Rb⁺   — Rubidium"),
    ("CS",  "Cs⁺   — Cesium"),
    # Anions
    ("CL",  "Cl⁻   — Chloride"),
    ("BR",  "Br⁻   — Bromide"),
    ("IOD", "I⁻    — Iodide"),
    ("F",   "F⁻    — Fluoride"),
]

COMMON_COFACTORS = [
    # Nucleotides & phosphates
    ("ATP", "ATP   — Adenosine triphosphate"),
    ("ADP", "ADP   — Adenosine diphosphate"),
    ("AMP", "AMP   — Adenosine monophosphate"),
    ("GTP", "GTP   — Guanosine triphosphate"),
    ("GDP", "GDP   — Guanosine diphosphate"),
    ("NAD", "NAD   — Nicotinamide adenine dinucleotide (oxidized)"),
    ("NAP", "NAP   — NADP (oxidized)"),
    ("NAI", "NAI   — NAD (reduced, NADH)"),
    ("FAD", "FAD   — Flavin adenine dinucleotide"),
    ("FMN", "FMN   — Flavin mononucleotide"),
    ("COA", "COA   — Coenzyme A"),
    ("SAM", "SAM   — S-adenosylmethionine"),
    ("SAH", "SAH   — S-adenosylhomocysteine"),
    ("PLP", "PLP   — Pyridoxal phosphate"),
    ("TPP", "TPP   — Thiamine pyrophosphate"),
    ("HEM", "HEM   — Heme (protoporphyrin IX + Fe)"),
    ("HEC", "HEC   — Heme C"),
    # Substrate analogs
    ("AP5", "AP5   — P1,P5-di(adenosine-5')pentaphosphate"),
    ("ANP", "ANP   — AMP-PNP (non-hydrolyzable ATP analog)"),
    ("AGS", "AGS   — ATPγS (thio-ATP analog)"),
    ("GSP", "GSP   — GTPγS (non-hydrolyzable GTP analog)"),
]

COMMON_SMALL_MOLECULES = [
    ("HOH", "H₂O   — Water"),
    ("GOL", "GOL   — Glycerol"),
    ("EDO", "EDO   — Ethylene glycol"),
    ("DMS", "DMS   — DMSO"),
    ("ACT", "ACT   — Acetate"),
    ("SO4", "SO₄²⁻ — Sulfate"),
    ("PO4", "PO₄³⁻ — Phosphate"),
    ("CIT", "CIT   — Citrate"),
    ("MLI", "MLI   — Malonate"),
    ("SUC", "SUC   — Succinate"),
]

# Combined list for quick-pick menus
ALL_COMMON_LIGANDS = COMMON_IONS + COMMON_COFACTORS + COMMON_SMALL_MOLECULES


# ═══════════════════════════════════════════════════════════════════════════════
# PROTEIN POST-TRANSLATIONAL MODIFICATIONS (PTMs)
# ═══════════════════════════════════════════════════════════════════════════════

PROTEIN_PTMS = [
    # Phosphorylation
    ("SEP", "SEP — Phosphoserine (pS)"),
    ("TPO", "TPO — Phosphothreonine (pT)"),
    ("PTR", "PTR — Phosphotyrosine (pY)"),
    ("HIP", "HIP — Phosphohistidine (pH)"),
    # Methylation
    ("MLY", "MLY — N6-methyl-lysine (mono)"),
    ("MLZ", "MLZ — N6,N6-dimethyl-lysine"),
    ("M3L", "M3L — N6,N6,N6-trimethyl-lysine"),
    ("MK8", "MK8 — N6-methyl-lysine (alt)"),
    ("MAR", "MAR — Asymmetric dimethylarginine (ADMA)"),
    ("SMC", "SMC — Symmetric dimethylarginine (SDMA)"),
    # Acetylation
    ("ALY", "ALY — N6-acetyl-lysine"),
    ("AYA", "AYA — N-acetyl-alanine"),
    # Hydroxylation
    ("HYP", "HYP — 4-hydroxyproline"),
    ("HY3", "HY3 — 3-hydroxyproline"),
    ("TYS", "TYS — Sulfotyrosine"),
    # Oxidation / Redox
    ("CSO", "CSO — S-hydroxycysteine (sulfenic acid)"),
    ("CSD", "CSD — S-sulfo-cysteine (sulfinic acid)"),
    ("OCS", "OCS — Cysteinesulfonic acid"),
    ("CSS", "CSS — Disulfide-bonded cysteine pair"),
    ("MSE", "MSE — Selenomethionine"),
    # Glycosylation (common linkage residues)
    ("OGT", "OGT — O-GlcNAc-threonine"),
    ("NGT", "NGT — N-linked GlcNAc-asparagine"),
    # Ubiquitin-like
    ("LLP", "LLP — Lysine-pyridoxal-5'-phosphate"),
    # Lipidation
    ("MYR", "MYR — N-myristoyl-glycine"),
    ("PLM", "PLM — S-palmitoyl-cysteine"),
    # Crosslinks / unusual
    ("LYR", "LYR — Lysine-retinal (Schiff base)"),
    ("CIR", "CIR — Citrulline (deiminated arginine)"),
    ("DAL", "DAL — D-alanine"),
    ("DVA", "DVA — D-valine"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# RNA BASE MODIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

RNA_MODIFICATIONS = [
    ("5MC", "5MC — 5-methylcytidine (m5C)"),
    ("OMG", "OMG — 2'-O-methylguanosine (Gm)"),
    ("OMC", "OMC — 2'-O-methylcytidine (Cm)"),
    ("OMU", "OMU — 2'-O-methyluridine (Um)"),
    ("MA6", "MA6 — N6-methyladenosine (m6A)"),
    ("1MA", "1MA — 1-methyladenosine (m1A)"),
    ("PSU", "PSU — Pseudouridine (Ψ)"),
    ("H2U", "H2U — Dihydrouridine (D)"),
    ("4SU", "4SU — 4-thiouridine (s4U)"),
    ("I",   "I   — Inosine"),
    ("7MG", "7MG — 7-methylguanosine (m7G)"),
    ("2MG", "2MG — N2-methylguanosine (m2G)"),
    ("5MU", "5MU — 5-methyluridine / ribothymidine (T)"),
    ("YYG", "YYG — Wybutosine"),
    ("QUO", "QUO — Queuosine"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# DNA BASE MODIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

DNA_MODIFICATIONS = [
    ("5CM", "5CM — 5-methylcytosine (5mC)"),
    ("5HM", "5HM — 5-hydroxymethylcytosine (5hmC)"),
    ("5FC", "5FC — 5-formylcytosine (5fC)"),
    ("5CC", "5CC — 5-carboxylcytosine (5caC)"),
    ("6MA", "6MA — N6-methyladenine (6mA)"),
    ("8OG", "8OG — 8-oxoguanine"),
    ("DHU", "DHU — Dihydrouridine"),
]
