# af3_builder/id_manager.py
from typing import List, Optional

class IDManager:
    """Assign unique sequence/entity IDs, optionally in spreadsheet-style (A, B, ..., Z, AA, AB, ...)."""

    def __init__(self):
        self._used_ids: set[str] = set()

    @staticmethod
    def to_spreadsheet_style(n: int) -> str:
        """Convert 1-based index to spreadsheet-style string (A, B, ..., AA, AB, ...)."""
        if n < 1:
            raise ValueError("Index must be >= 1")
        result = ""
        while n > 0:
            n, remainder = divmod(n-1, 26)
            result = chr(65 + remainder) + result
        return result

    @staticmethod
    def from_spreadsheet_style(s: str) -> int:
        """Convert spreadsheet-style string to 1-based index."""
        if not s or not s.isalpha():
            raise ValueError("ID must be non-empty and alphabetic")
        n = 0
        for char in s.upper():
            n = n * 26 + (ord(char) - ord('A') + 1)
        return n

    def next_id(self, existing_ids: Optional[List[str]] = None) -> str:
        """Return the next available unique ID."""
        if existing_ids:
            self._used_ids.update(existing_ids)

        idx = 1
        while True:
            candidate = self.to_spreadsheet_style(idx)
            if candidate not in self._used_ids:
                self._used_ids.add(candidate)
                return candidate
            idx += 1
