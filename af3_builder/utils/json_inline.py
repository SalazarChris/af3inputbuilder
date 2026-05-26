import json
import re

class InlineArrayEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that prints:
      • single-element arrays inline
      • multi-element arrays inline (if they contain only primitive values)
    while indenting objects normally.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("indent", 2)
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        # First do a normal pretty-print
        text = super().encode(obj)

        # 1) Inline any array that is entirely on multiple lines but contains only primitives
        # Matches things like:
        #
        # [
        #   "A",
        #   "B",
        #   "C"
        # ]
        #
        primitive_list_pattern = re.compile(
            r'\[\s+((?:"[^"]*"\s*,\s*)*(?:"[^"]*"?)|'
            r'(?:\d+\s*,\s*)*(?:\d+)|'
            r'(?:true|false|null)\s*)\s+\]',
            flags=re.IGNORECASE | re.DOTALL
        )

        def compress_list(match):
            content = match.group(1)
            # Normalize spacing inside
            parts = [p.strip() for p in content.split(",")]
            line = ", ".join(parts)
            return f"[{line}]"

        text = primitive_list_pattern.sub(lambda m: compress_list(m), text)

        # 2) Inline single-element lists using simpler regex
        single_item_pattern = re.compile(
            r'\[\s+(".*?"|\d+|true|false|null)\s+\]',
            flags=re.IGNORECASE | re.DOTALL
        )

        text = single_item_pattern.sub(r'[\1]', text)

        return text
