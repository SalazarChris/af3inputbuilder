# af3_builder/io.py
import json
from typing import Optional, Dict, Any
import os
import tempfile
from datetime import datetime
from .json_inline import InlineArrayEncoder


def save_json(path: str, data: dict):
    """
    Atomic save:
      - write to a temp file in the same directory
      - then os.replace() to final path (atomic on same filesystem)
    Keeps your InlineArrayEncoder formatting.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Output filename must be a non-empty string")

    out_path = path.strip()
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # create temp file in same directory for atomic replace
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(InlineArrayEncoder().encode(data))
            f.write("\n")
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def autosave_json(data: dict, *, prefix: str = "autosave", directory: str = ".") -> str:
    """
    Convenience autosave with timestamped filename.
    Returns the filepath written.
    """
    os.makedirs(directory, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(directory, f"{prefix}_{ts}.json")
    save_json(out_path, data)
    return out_path


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None
