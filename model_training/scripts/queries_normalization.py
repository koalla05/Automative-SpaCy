import json
import unicodedata
import re
from pathlib import Path

def normalize_text(text):
    # Normalize characters (e.g., non-breaking hyphen to hyphen)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\xa0', ' ')   # non-breaking space
    return text

def normalize_labels(data):
    normalized_data = []

    for item in data:
        original_text = item["text"]
        normalized_text = normalize_text(original_text)
        new_labels = []

        for label in item.get("label", []):
            raw_entity_text = label["text"]
            entity_label = label["labels"]
            clean_entity_text = normalize_text(raw_entity_text).strip()

            # Find exact match position in normalized text
            start = normalized_text.find(clean_entity_text)
            if start == -1:
                print(f"[!] Could not align entity '{clean_entity_text}' in: {normalized_text}")
                continue  # skip this label

            end = start + len(clean_entity_text)
            new_labels.append({
                "start": start,
                "end": end,
                "text": normalized_text[start:end],
                "labels": entity_label
            })

        normalized_data.append({
            "id": item.get("id", None),
            "text": normalized_text,
            "label": new_labels
        })

    return normalized_data


# ============================================
# FIX: Use absolute paths
# ============================================
# Get the directory where THIS script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Project root is one level up
PROJECT_ROOT = SCRIPT_DIR.parent
# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Input and output files
INPUT_FILE = DATA_DIR / "new_model_labels.json"
OUTPUT_FILE = DATA_DIR / "labels_normalized.json"

print(f"📂 Looking for input file at: {INPUT_FILE}")

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found at: {INPUT_FILE}")
    print(f"\n💡 Available .json files in {DATA_DIR}:")
    if DATA_DIR.exists():
        json_files = list(DATA_DIR.glob("*.json"))
        if json_files:
            for f in json_files:
                print(f"   - {f.name}")
        else:
            print("   (no .json files found)")
    else:
        print(f"   ❌ Data directory doesn't exist: {DATA_DIR}")
    exit(1)

# --- Load, normalize, and save ---
print(f"📖 Reading file...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    original_data = json.load(f)

print(f"🔧 Normalizing {len(original_data)} examples...")
normalized_data = normalize_labels(original_data)

print(f"💾 Saving to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, ensure_ascii=False, indent=2)

print(f"✅ Normalized {len(normalized_data)} examples and saved to {OUTPUT_FILE.name}")
print(f"\n📍 Full path: {OUTPUT_FILE}")