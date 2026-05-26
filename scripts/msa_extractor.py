#!/usr/bin/env python3
"""
msa_extractor.py
================
Generalized utility for extracting MSAs from AlphaFold 3 output JSON files.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

def extract_msas(input_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Scans input_dir for AF3 result JSON files and extracts MSAs to output_dir.
    
    Returns:
        List of (job_id, msa_type) tuples that were successfully saved.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # We look for files ending in _data.json which is standard for AF3 results
    # We scan recursively in case results are nested
    json_files = list(input_path.rglob("*_data.json"))
    
    if not json_files:
        # Fallback: check if the input_dir itself is a results folder containing data.json
        if (input_path / "data.json").exists():
            json_files = [input_path / "data.json"]
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # AF3 results JSON structure:
            # { "sequences": [ { "protein": { "pairedMsa": "...", "unpairedMsa": "..." }, ... }, ... ] }
            
            sequences = data.get("sequences", [])
            job_id = json_file.stem.replace("_data", "")
            
            if job_id == "data":
                # If it's just 'data.json', use the parent folder name
                job_id = json_file.parent.name
            
            extracted_count = 0
            for i, seq_entry in enumerate(sequences):
                if "protein" in seq_entry:
                    protein = seq_entry["protein"]
                    
                    # Extract PAIRED
                    paired = protein.get("pairedMsa")
                    if paired:
                        filename = f"{job_id.upper()}_SEQ{i}_PAIRED.a3m"
                        with open(output_path / filename, "w") as out:
                            out.write(paired)
                        saved_files.append((job_id, f"SEQ{i} PAIRED"))
                        extracted_count += 1
                        
                    # Extract UNPAIRED
                    unpaired = protein.get("unpairedMsa")
                    if unpaired:
                        filename = f"{job_id.upper()}_SEQ{i}_UNPAIRED.a3m"
                        with open(output_path / filename, "w") as out:
                            out.write(unpaired)
                        saved_files.append((job_id, f"SEQ{i} UNPAIRED"))
                        extracted_count += 1
            
            if extracted_count == 0:
                print(f"  [!] No MSAs found in {json_file.name}")
                
        except Exception as e:
            print(f"  [!] Error processing {json_file}: {e}")
            
    return saved_files

if __name__ == "__main__":
    # Minimal CLI for standalone use
    import sys
    if len(sys.argv) < 3:
        print("Usage: python msa_extractor.py <input_dir> <output_dir>")
        sys.exit(1)
        
    results = extract_msas(sys.argv[1], sys.argv[2])
    print(f"Extracted {len(results)} MSA files.")
