import json
import unicodedata
import re

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

# --- Load, normalize, and save ---
with open("labels.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

normalized_data = normalize_labels(original_data)

with open("labels_normalized.json", "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, ensure_ascii=False, indent=2)

print(f"✅ Normalized {len(normalized_data)} examples and saved to labels_normalized.json")
