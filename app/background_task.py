import json
import os
import csv
from typing import Dict, List
from openai import OpenAI  
from app.entity_extractor import extract_entities_spacy
from datetime import datetime

client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = "autolabeled"
CSV_PATH = os.path.join(BASE_DIR, "../data", "file-parameters.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_allowed_parameters(csv_path: str):
    eq_types = set()
    manufacturers = set()
    models = set()

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['equipment_type']:
                eq_types.add(row['equipment_type'].strip())
            if row['manufacturer']:
                manufacturers.add(row['manufacturer'].strip())
            if row['models']:
                model_list = [m.strip() for m in row['models'].split(';')]
                models.update(model_list)

    return eq_types, manufacturers, models

tool_definition = {
    "type": "function",
    "function": {
        "name": "label_entities",
        "description": "Extract EQ_TYPE, MANUFACTURER, MODEL entities from a query",
        "parameters": {
            "type": "object",
            "properties": {
                "EQ_TYPE": {"type": "array", "items": {"type": "string"}},
                "MANUFACTURER": {"type": "array", "items": {"type": "string"}},
                "MODEL": {"type": "array", "items": {"type": "string"}}
            },
            "required": []
        }
    }
}

EQ_TYPE_SET, MANUFACTURER_SET, MODEL_SET = load_allowed_parameters(CSV_PATH)

def ask_openai_with_tool(text: str) -> Dict[str, List[str]]:
    """Ask OpenAI to label entities, returning exact text spans in the query."""
    tool_call = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""
Extract entities from this user query (may be in Ukrainian):

\"{text}\"

Instructions:
- Identify EQ_TYPE, MANUFACTURER, and MODEL.
- Return the **exact substring from the query** for each entity.
- Use allowed values only as reference, but **return the query text itself**.
- Output JSON with keys: EQ_TYPE, MANUFACTURER, MODEL. Skip empty keys.

Allowed values (for reference):
EQ_TYPE: {list(EQ_TYPE_SET)}
MANUFACTURER: {list(MANUFACTURER_SET)}
MODEL: {list(MODEL_SET)}

Example output:
{{
  "EQ_TYPE": ["інвертора"],
  "MANUFACTURER": ["Victron"],
  "MODEL": ["Quattro 48/15000/200-100/100"]
}}
"""
        }],
        tools=[tool_definition],
        tool_choice={"type": "function", "function": {"name": "label_entities"}},
        temperature=0
    )

    try:
        tool_response = tool_call.choices[0].message.tool_calls[0].function.arguments
        raw_data = json.loads(tool_response)

        # Do not filter, just return spans
        result = {k: v for k, v in raw_data.items() if v}
        return result
    except Exception as e:
        print("Tool parsing error:", e)
        return {}

def process_for_annotation(text: str, annotator_id=1, annotation_id_start=1, lead_time=5.0):
    """Generate Label Studio JSON for a single query."""
    result = extract_entities_spacy(text)
    total_labels = sum(len(v) for v in result.values())

    if total_labels < 3:
        autolabel = ask_openai_with_tool(text)

        labelstudio_entry = {
            "text": text,
            "id": annotation_id_start,
            "label": [],
            "annotator": annotator_id,
            "annotation_id": annotation_id_start,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "lead_time": lead_time
        }

        for label_type, values in autolabel.items():
            for value in values:
                start = text.lower().find(value.lower())
                if start == -1:
                    # fallback: skip if not found
                    print(f"[WARN] Entity '{value}' of type {label_type} not found in text: {text}")
                    continue
                end = start + len(value)

                labelstudio_entry["label"].append({
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [label_type]
                })

        file_path = os.path.join(SAVE_DIR, "autolabel_labelstudio.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(labelstudio_entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
