import os
from functools import lru_cache

DATA_FILE = os.path.join(os.path.dirname(__file__), "../..", "data", "canon_models.txt")
DATA_FILE = os.path.abspath(DATA_FILE)


@lru_cache(None)
def load_canonical_models():
    mapping = {}

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Canon model file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()

            # skip empty/comment lines
            if not raw or raw.startswith("#"):
                continue

            # expect: A -> b
            if "->" not in raw:
                continue

            left, right = raw.split("->", 1)
            original = left.strip()
            normalized = right.strip()

            mapping[original.lower()] = normalized

    return mapping


def normalize_model(model_name: str) -> str:
    """
    Returns canonical model name if found; otherwise returns
    the input converted to lowercase with basic cleanup.
    """
    if not model_name:
        return ""

    mapping = load_canonical_models()
    key = model_name.strip().lower()

    # exact match
    if key in mapping:
        return mapping[key]

    # fallback — simple normalization
    return (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("~", "_")
    )
