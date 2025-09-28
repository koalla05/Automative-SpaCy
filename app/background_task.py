import json
import os
import csv
from typing import Dict, List
from openai import OpenAI  
from app.entity_extractor import extract_entities_spacy

client = OpenAI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = "autolabeled"
CSV_PATH = os.path.join(BASE_DIR, "data", "file-parameters.csv")
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
        "description": "Extract known EQ_TYPE, MANUFACTURER, MODEL entities from a user query",
        "parameters": {
            "type": "object",
            "properties": {
                "EQ_TYPE": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Equipment types found in the query"
                },
                "MANUFACTURER": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Manufacturers found in the query"
                },
                "MODEL": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Model names or codes found in the query"
                }
            },
            "required": [],
        }
    }
}

EQ_TYPE_SET, MANUFACTURER_SET, MODEL_SET = load_allowed_parameters(CSV_PATH)

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
    
def ask_openai_with_tool(text: str) -> Dict[str, List[str]]:
    tool_call = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an NER assistant for equipment-related user queries."},
            {"role": "user", "content": f"""
Extract entities from this text:
\"{text}\"

Return only known values from the allowed lists:
EQ_TYPE: {list(EQ_TYPE_SET)}
MANUFACTURER: {list(MANUFACTURER_SET)}
MODEL: {list(MODEL_SET)}
"""}
        ],
        tools=[tool_definition],
        tool_choice={"type": "function", "function": {"name": "label_entities"}},
        temperature=0
    )

    try:
        tool_response = tool_call.choices[0].message.tool_calls[0].function.arguments
        raw_data = json.loads(tool_response)

        # Filter again just in case
        result = {}

        if "EQ_TYPE" in raw_data:
            eqs = [x for x in raw_data["EQ_TYPE"] if x in EQ_TYPE_SET]
            if eqs: result["EQ_TYPE"] = eqs

        if "MANUFACTURER" in raw_data:
            mans = [x for x in raw_data["MANUFACTURER"] if x in MANUFACTURER_SET]
            if mans: result["MANUFACTURER"] = mans

        if "MODEL" in raw_data:
            mods = [x for x in raw_data["MODEL"] if x in MODEL_SET]
            if mods: result["MODEL"] = mods

        return result

    except Exception as e:
        print("Tool parsing error:", e)
        return {}

def process_for_annotation(text: str):
    result = extract_entities_spacy(text)

    total_labels = sum(len(v) for v in result.values())

    if total_labels < 3:
        #autolabel = ask_openai_to_label(text)
        autolabel = ask_openai_with_tool(text)
        print("asked openai")
        file_path = os.path.join(SAVE_DIR, "autolabel.json")

        data_entry = {
            "text": text,
            "spacy_result": result,
            "openai_result": autolabel
        }

        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(data_entry, f, ensure_ascii=False)
            f.write("\n")  
