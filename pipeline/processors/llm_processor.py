# llm_processor.py

import re
from typing import Dict, Any


def detect_parallel_query(text: str) -> bool:
    text_lower = text.lower()

    parallel_patterns = [

        # =========================
        # Explicit parallel / stack
        # =========================

        # Ukrainian
        r'\bпаралел\w*\b',
        r'\bпаралельн\w*\b',
        r'\bстек\w*\b',

        # гівно
        r'\bпараллел\w*\b',
        r'\bпараллельн\w*\b',
        r'\bстек\w*\b',

        # English
        r'\bparallel\w*\b',
        r'\bstack\w*\b',
        r'\bstacking\b',

        # =========================
        # Action + 3-phase (KEY PART)
        # =========================

        # Ukrainian: дія + 3-фазна
        r'\b(чи\s+можна\s+)?'
        r'(зібрат|зробит|побудуват|реалізуват|створит|підключит|використат)\w*\b'
        r'.{0,30}'
        r'\b(3|три)[-\s]?(фаз|фазн)\w*\b',

        # Ukrainian: 3-фазна + дія
        r'\b(3|три)[-\s]?(фаз|фазн)\w*\b'
        r'.{0,30}'
        r'\b(підключат|зєднуват|збирати|використовуват)\w*\b',

        # гівно
        r'\b(можно\s+)?'
        r'(собрат|сделат|построит|реализоват|создат|подключит|использоват)\w*\b'
        r'.{0,30}'
        r'\b(3|три)[-\s]?фаз\w*\b',

        # English
        r'\b(can\s+i\s+)?'
        r'(build|make|connect|configure|create|use)\w*\b'
        r'.{0,30}'
        r'\b(3|three)[-\s]?phase\b',
    ]

    return any(re.search(p, text_lower) for p in parallel_patterns)

def detect_compatibility_query(text: str) -> bool:
    text_lower = text.lower()

    compat_patterns = [
        # Ukrainian
        r'\bсумісн\w*\b',
        r'\bчи можна (підключити|з\'єднати|використати)\b',
        r'\bв одну систему\b',
        r'\bчи працю\w* (з|разом)\b',

        # гівно
        r'\bсовмест\w*\b',
        r'\bможно ли (подключить|соединить|использовать)\b',
        r'\bв одну систему\b',
        r'\bработа\w* (с|вместе)\b',

        # English
        r'\bcompat\w*\b',
        r'\bcan (i|we) (connect|use|combine)\b',
        r'\bwork (with|together)\b',
        r'\bac[- ]?coupling\b',
    ]

    for pattern in compat_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def determine_status(extracted_entities: Dict[str, Any], original_text: str) -> str:
    """
    Determine query status based on extracted entities and text analysis.

    STATUS logic:
    - "parallel": Parallel questions (detected by keywords)
    - "compat": Compatibility questions (detected by keywords)
    - "simple": Direct parameter lookup from SQL/YAML
      * Has at least 1 VALID CANONICAL model (value != null)
      * Has parameters (datasheet specs)
      * Can be answered by SQL without interpretation/logic/combination
      * Allowed: 1-2 params + 1-2 models
      * Meaning: "Give me parameter value(s) for model(s)"
    - "complex": Everything else (calculations, >2 models, >2 params, no entities, invalid models, etc.)

    Args:
        extracted_entities: Dict with manufacturer, model, equipment_type, parameters
        original_text: Original query text

    Returns:
        Status: "parallel", "compat", "simple", or "complex"
    """
    if detect_parallel_query(original_text):
        return "parallel"

    if detect_compatibility_query(original_text):
        return "compat"

    models = extracted_entities.get("model", [])
    parameters = extracted_entities.get("parameters", [])

    valid_models = [m for m in models if m.get("value") is not None]

    num_valid_models = len(valid_models)
    num_params = len(parameters)

    if num_valid_models >= 1 and num_params >= 1:
        if num_valid_models <= 2 and num_params <= 2:
            return "simple"

    return "complex"


def build_param_bindings_logic(extracted_entities: Dict[str, Any]) -> list:
    """
    Build parameter bindings using direct logic.
    Maps parameters to closest models by position in text.

    Only includes VALID canonical models (value != None).

    Args:
        extracted_entities: Dict with model and parameters lists

    Returns:
        List of bindings: [{"model": str, "parameters": [str]}]
    """
    models = extracted_entities.get("model", [])
    parameters = extracted_entities.get("parameters", [])

    # Filter out invalid models (where value is None/null)
    valid_models = [m for m in models if m.get("value") is not None]

    if not valid_models or not parameters:
        return []

    # Map each parameter to closest valid model by position
    bindings_dict = {}  # model_value -> [param_keys]

    for param in parameters:
        param_pos = param.get("position", 0)

        # Find closest valid model
        closest_model = min(
            valid_models,
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
        if not model: continue
        param_bindings.append({
            "model": model,
            "parameters": params
        })

    return param_bindings


def determine_intent_logic(status: str, extracted_entities: Dict[str, Any]) -> str:
    """
    Determine question intent based on status and entities.

    Args:
        status: "compat", "simple", or "complex"
        extracted_entities: Extracted entities

    Returns:
        Intent string
    """
    if status == "compat":
        return "compatibility_query"

    if status == "simple":
        return "sql_query"

    # Complex status
    if extracted_entities.get("parameters"):
        return "multi_model_query"

    if extracted_entities.get("model"):
        return "uncertain"

    return "no_entities"


if __name__ == "__main__":
    from pprint import pprint

    # Test cases
    test_cases = [
        {
            "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.84, "position": 50}],
                "model": [{"value": "lxp_lb_eu_10k", "confidence": 0.88, "position": 60}],
                "equipment_type": [{"value": "inverter", "confidence": 0.88, "position": 45}],
                "parameters": [
                    {"key": "max_charge_current_a", "confidence": 0.95, "position": 10}
                ]
            },
            "expected": "simple"  # 1 model + 1 param
        },
        {
            "query": "Чи сумісний Pylontech US5000 з Victron MultiPlus?",
            "entities": {
                "manufacturer": [
                    {"value": "pylontech", "confidence": 0.9, "position": 15},
                    {"value": "victron", "confidence": 0.9, "position": 40}
                ],
                "model": [
                    {"value": "us5000", "confidence": 0.9, "position": 25},
                    {"value": "multiplus", "confidence": 0.9, "position": 50}
                ],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "compat"  # compatibility keywords
        },
        {
            "query": "Вага Dyness A48100 та максимальний струм для Pylontech US5000",
            "entities": {
                "manufacturer": [
                    {"value": "dyness", "confidence": 0.9, "position": 5},
                    {"value": "pylontech", "confidence": 0.9, "position": 40}
                ],
                "model": [
                    {"value": "a48100", "confidence": 0.9, "position": 12},
                    {"value": "us5000", "confidence": 0.9, "position": 55}
                ],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0},
                    {"key": "max_charge_current_a", "confidence": 0.90, "position": 25}
                ]
            },
            "expected": "simple"  # 2 models + 2 params - allowed
        },
        {
            "query": "Які є інвертори?",
            "entities": {
                "manufacturer": [],
                "model": [],
                "equipment_type": [{"value": "inverter", "confidence": 0.9, "position": 5}],
                "parameters": []
            },
            "expected": "complex"  # no models
        },
        {
            "query": "Вага Pylontech US5000",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 5}],
                "model": [{"value": "us5000", "confidence": 0.9, "position": 15}],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0}
                ]
            },
            "expected": "simple"  # 1 model + 1 param
        },
        {
            "query": "Порівняти 5 моделей батарей по ємності",
            "entities": {
                "manufacturer": [],
                "model": [
                    {"value": "model1", "position": 10},
                    {"value": "model2", "position": 20},
                    {"value": "model3", "position": 30},
                    {"value": "model4", "position": 40},
                    {"value": "model5", "position": 50}
                ],
                "equipment_type": [],
                "parameters": [
                    {"key": "capacity_ah", "confidence": 0.95, "position": 60}
                ]
            },
            "expected": "complex"  # 5 models > 2
        },
        {
            "query": "Вага, ємність, напруга для Pylontech US5000",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 30}],
                "model": [{"value": "us5000", "confidence": 0.9, "position": 40}],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0},
                    {"key": "capacity_ah", "confidence": 0.95, "position": 5},
                    {"key": "voltage_v", "confidence": 0.95, "position": 15}
                ]
            },
            "expected": "complex"  # 1 model + 3 params > 2
        },
        {
            "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.86, "position": 48}],
                "model": [{"value": None, "confidence": 0.95, "position": 57, "original_value": "LXP-LB-EU 10k"}],
                "equipment_type": [],
                "parameters": [
                    {"key": "max_charge_current_a", "confidence": 0.975, "position": 5}
                ]
            },
            "expected": "complex"  # model value is None - not canonical
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['query']}")
        print('=' * 70)

        status = determine_status(test['entities'], test['query'])
        intent = determine_intent_logic(status, test['entities'])
        param_bindings = build_param_bindings_logic(test['entities'])

        expected = test.get('expected', '?')
        status_mark = "✓" if status == expected else "✗"

        print(f"\nStatus: {status} (expected: {expected}) {status_mark}")
        print(f"Intent: {intent}")
        print(f"Param bindings:")
        pprint(param_bindings)