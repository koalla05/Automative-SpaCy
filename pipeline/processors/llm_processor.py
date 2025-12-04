# llm_processor.py

import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please create .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)


def detect_question_type(text: str) -> str:
    """
    Detect question type based on keywords.

    Returns:
        "compat" if compatibility question, None otherwise
    """
    text_lower = text.lower()

    compat_patterns = [
        # Ukrainian
        r'\b(сумісн|сумісний|сумісна|сумісне|сумісності|сумісність)\b',
        r'\bчи можна (підключити|з\'єднати|використати)\b',
        r'\bв одну систему\b',

        # Russian
        r'\b(совмест|совместим|совместима|совместимо|совместимости|совместимость)\b',
        r'\bможно ли (подключить|соединить|использовать)\b',
        r'\bв одну систему\b',

        # English
        r'\b(compat|compatible|compatibility)\b',
        r'\bcan (i|we) (connect|use|combine)\b',
        r'\bwork (with|together)\b',
        r'\bac[- ]?coupling\b',
    ]

    for pattern in compat_patterns:
        if re.search(pattern, text_lower):
            return "compat"

    return None


# SIMPLIFIED SYSTEM PROMPT - ONLY STATUS
SYSTEM_PROMPT = """
You are a STATUS classifier for the IPG (Intelligence Preprocessing Gateway).

Your ONLY job: Determine if the query is "simple" or "complex".

STATUS = "simple" if:
- Query asks for specification parameters from model datasheets (вага, ємність, струм, напруга, потужність, ККД, etc.)
- Can be answered by direct SQL/database lookup
- Examples: 
  * "What is max charge current for Model X?"
  * "Вага Pylontech US5000"
  * "Максимальний струм для US5000 і ємність для A48100"

STATUS = "complex" if:
- Compatibility questions
- Configuration/setup questions  
- Calculations needed
- Documentation/instructions
- Vague/unclear queries
- Examples:
  * "Can I connect X to Y?"
  * "How to configure Z?"
  * "Wiring diagram for W"

Response format (STRICT):
STATUS: <simple|complex>

Nothing else. Just one line.
"""


class LLMProcessor:
    def __init__(self, par_lst: str, mod_lst: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.parameters_list = par_lst
        self.models_list = mod_lst

    def process_question(self, extracted_entities: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Sends extracted_entities + original text to OpenAI.
        NOW ONLY RETURNS STATUS.
        """
        user_content = f"""Original query: {original_text}

Extracted entities:
- Manufacturers: {[m['value'] for m in extracted_entities.get('manufacturer', [])]}
- Models: {[m['value'] for m in extracted_entities.get('model', [])]}
- Equipment types: {[e['value'] for e in extracted_entities.get('equipment_type', [])]}
- Parameters: {[p['key'] for p in extracted_entities.get('parameters', [])]}

Classify this query as "simple" or "complex"."""

        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse status
        status = self.parse_status(raw_text)

        return {
            "status": status,
            "intent": None,  # Will be determined by logic
            "param_bindings": []  # Will be built by logic
        }

    def parse_status(self, raw_text: str) -> str:
        """Extract status from LLM response."""
        for line in raw_text.split('\n'):
            line = line.strip()
            if line.startswith("STATUS:"):
                status = line.split(":", 1)[1].strip().lower()
                if status in ["simple", "complex"]:
                    return status

        # Fallback
        return "complex"


# Helper function to build param_bindings with logic (not LLM)
def build_param_bindings_logic(extracted_entities: Dict[str, Any]) -> list:
    """
    Build parameter bindings using direct logic (no LLM).
    Maps parameters to closest models by position in text.
    """
    models = extracted_entities.get("model", [])
    parameters = extracted_entities.get("parameters", [])

    if not models or not parameters:
        return []

    # Map each parameter to closest model by position
    bindings_dict = {}  # model_value -> [param_keys]

    for param in parameters:
        param_pos = param.get("position", 0)

        # Find closest model
        closest_model = min(
            models,
            key=lambda m: abs(m.get("position", 0) - param_pos)
        )

        model_value = closest_model["value"]
        param_key = param["key"]

        if model_value not in bindings_dict:
            bindings_dict[model_value] = []

        if param_key not in bindings_dict[model_value]:
            bindings_dict[model_value].append(param_key)

    # Convert to list format
    param_bindings = []
    for model, params in bindings_dict.items():
        param_bindings.append({
            "model": model,
            "parameters": params
        })

    return param_bindings


# Helper function to determine intent with logic (not LLM)
def determine_intent_logic(question_type: str, extracted_entities: Dict[str, Any]) -> str:
    """
    Determine question intent using logic.

    Args:
        question_type: "compat" or None
        extracted_entities: Extracted entities

    Returns:
        Intent string
    """
    if question_type == "compat":
        return "compatibility_query"

    # If has parameters, it's sql_query
    if extracted_entities.get("parameters"):
        return "sql_query"

    # If has models but no parameters, uncertain
    if extracted_entities.get("model"):
        return "uncertain"

    return "uncertain"


if __name__ == "__main__":
    from pprint import pprint

    # Test 1: Regular spec query
    print("=" * 60)
    print("Test 1: Regular spec query")
    print("=" * 60)

    extracted_entities_example = {
        "manufacturer": [{"value": "luxpower", "confidence": 0.84, "position": 50}],
        "model": [{"value": "lxp_lb", "confidence": 0.88, "position": 60}],
        "equipment_type": [{"value": "inverter", "confidence": 0.88, "position": 45}],
        "parameters": [
            {"key": "max_charge_current_a", "confidence": 0.95, "position": 10},
            {"key": "max_discharge_current_a", "confidence": 0.90, "position": 15}
        ]
    }

    processor = LLMProcessor("", "")

    query1 = "Який максимальний струм заряджання/розряджання на інверторі LuxPower?"
    llm_result = processor.process_question(extracted_entities_example, query1)

    print(f"\nQuery: {query1}")
    print(f"LLM Result:")
    pprint(llm_result)

    # Build param_bindings with logic
    param_bindings = build_param_bindings_logic(extracted_entities_example)
    print(f"\nParam bindings (logic):")
    pprint(param_bindings)

    # Determine intent with logic
    question_type = detect_question_type(query1)
    intent = determine_intent_logic(question_type, extracted_entities_example)
    print(f"\nQuestion type: {question_type}")
    print(f"Intent (logic): {intent}")

    # Test 2: Compatibility query
    print("\n" + "=" * 60)
    print("Test 2: Compatibility query")
    print("=" * 60)

    query2 = "Чи сумісний Pylontech US5000 з Victron MultiPlus?"
    question_type2 = detect_question_type(query2)
    intent2 = determine_intent_logic(question_type2, {"model": [{"value": "us5000"}]})

    print(f"\nQuery: {query2}")
    print(f"Question type: {question_type2}")
    print(f"Intent (logic): {intent2}")
    print("(LLM not called for compat queries)")