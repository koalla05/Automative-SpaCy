import json
import os
from typing import Dict, List
from openai import OpenAI  
from app.entity_extractor import extract_entities_spacy

client = OpenAI()

SAVE_DIR = "autolabeled"
os.makedirs(SAVE_DIR, exist_ok=True)

def ask_openai_to_label(text: str) -> Dict[str, List[str]]:
    prompt = prompt = f"""
You are an NER assistant for equipment-related user queries.

Instructions:
- Extract entities from the following text: "{text}"
- Identify only these entity types:
  1. EQ_TYPE — the type of equipment (e.g., inverter, battery, solar system)
  2. MANUFACTURER — the brand or manufacturer (e.g., DEYE, Huawei)
  3. MODEL — the model name or code (e.g., SUN 5K-SG-EU, LUNA2000-14-S1)
- Return a single JSON object with **keys "EQ_TYPE", "MANUFACTURER", "MODEL"**.
- Each key should map to a list of strings (skip the key entirely if no entities found for it).
- Do NOT add any extra text, explanations, or commentary—only valid JSON.

Example output:
{{
  "EQ_TYPE": ["user manual", "battery"],
  "MANUFACTURER": ["Huawei"],
  "MODEL": ["LUNA2000-14-S1"]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an NER assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    try:
        json_start = response.choices[0].message.content.find('{')
        json_data = response.choices[0].message.content[json_start:]
        return json.loads(json_data)
    except Exception as e:
        print("Error parsing OpenAI response:", e)
        return {}
    

def process_for_annotation(text: str):
    result = extract_entities_spacy(text)

    total_labels = sum(len(v) for v in result.values())

    if total_labels < 3:
        autolabel = ask_openai_to_label(text)
        file_path = os.path.join(SAVE_DIR, "autolabel.json")

        data_entry = {
            "text": text,
            "spacy_result": result,
            "openai_result": autolabel
        }

        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(data_entry, f, ensure_ascii=False)
            f.write("\n")  
