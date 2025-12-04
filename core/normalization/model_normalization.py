import os
from functools import lru_cache
from typing import Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), "../..", "data", "canon_models.txt")
DATA_FILE = os.path.abspath(DATA_FILE)


@lru_cache(None)
def load_canonical_models():
    """
    Load canonical model mappings from file.

    Returns:
        Dict mapping original model names (lowercase) to canonical names
    """
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

            # Store lowercase key for case-insensitive matching
            mapping[original.lower()] = normalized

    return mapping


def normalize_model(model_name: str) -> Optional[str]:
    """
    Returns canonical model name if found in canon_models.txt.

    Args:
        model_name: Original model name from NER

    Returns:
        Canonical model name if found, None otherwise

    Note:
        This function ONLY returns models from canon_models.txt.
        If a model is not in the file, it returns None.
        NO automatic normalization/fallback.
    """
    if not model_name:
        return None

    mapping = load_canonical_models()
    key = model_name.strip().lower()

    # Return canonical name if found, None otherwise
    return mapping.get(key)


def is_model_in_canon(model_name: str) -> bool:
    """
    Check if a model exists in canonical models file.

    Args:
        model_name: Model name to check

    Returns:
        True if model exists in canon_models.txt, False otherwise
    """
    if not model_name:
        return False

    mapping = load_canonical_models()
    key = model_name.strip().lower()
    return key in mapping


def get_all_canonical_models() -> dict:
    """
    Get all canonical models from file.

    Returns:
        Dictionary of original -> canonical mappings
    """
    return load_canonical_models()


# For debugging - print stats about loaded models
if __name__ == "__main__":
    print("=" * 60)
    print("CANONICAL MODELS LOADER")
    print("=" * 60)

    print(f"\nLoading from: {DATA_FILE}")

    try:
        models = load_canonical_models()
        print(f"✅ Loaded {len(models)} canonical models")

        # Show first 10 examples
        print("\n📋 First 10 examples:")
        for idx, (original, canonical) in enumerate(list(models.items())[:10], 1):
            print(f"   {idx}. '{original}' -> '{canonical}'")

        # Test normalization
        print("\n🧪 Testing normalization:")
        test_cases = [
            "US5000",
            "lxp-lb-eu",
            "MultiPlus",
            "UNKNOWN_MODEL_XYZ"
        ]

        for test in test_cases:
            result = normalize_model(test)
            if result:
                print(f"   ✅ '{test}' -> '{result}'")
            else:
                print(f"   ❌ '{test}' -> None (not in canon file)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Create the file at: {DATA_FILE}")
        print("   Format: original_name -> canonical_name")