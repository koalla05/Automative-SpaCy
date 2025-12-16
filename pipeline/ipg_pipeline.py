# file: ipg_pipeline.py
from pipeline.exctractors.parameter_extractor import process_question as extract_entities
from pipeline.processors.llm_processor import (
    determine_status,
    build_param_bindings_logic,
    determine_intent_logic
)


class IPGPipeline:
    def __init__(self):
        """Initialize pipeline without LLM."""
        pass


    def process(self, text: str):
        extraction_result = extract_entities(text)
        extracted_entities = extraction_result["extracted_entities"]

        routing = extraction_result.get("routing")

        status = determine_status(extracted_entities, text)

        if status in ["simple", "complex"]:
            param_bindings = build_param_bindings_logic(extracted_entities)
        else:
            param_bindings = []

        question_intent = determine_intent_logic(status, extracted_entities)

        if status == "compat":
            routing = {
                "recommended_strategy": "compatibility_check",
                "message": "Compatibility query detected",
                "entities": {
                    "manufacturers": [m["value"] for m in extracted_entities.get("manufacturer", [])],
                    "models": [m["value"] for m in extracted_entities.get("model", [])],
                    "equipment_types": [e["value"] for e in extracted_entities.get("equipment_type", [])]
                }
            }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "question_raw": text,
            "status": status,
            "question_intent": question_intent,
            "extracted_entities": extracted_entities,
            "routing": routing
        }


# -------- Example usage --------
if __name__ == "__main__":
    import json

    pipeline = IPGPipeline()

    test_queries = [
        "Яка максимальна вхідна потужність по фем на LuxPwer LXP-LB-EU 10k?",
        "гранична напруга зарядки Pylontech US5000?",
        "Який максимальний розрядний струм на інверторі Deye SUN-6K-SG05LP1-EU",
        "Яка номінальна вихідна частота на інверторі Sofar 12KTLX-G3?", # ??
        "Яка номінальна вихідна потужність на інверторі Sofar 110KTLX-G4?",
        "Яка кількість входів постійного струму на інверторі Sofar 110KTLX-G4?", # ?
        "Яка максимальна напруга заряду на інверторі LuxPower SNA 5000?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {idx}: {query}")
        print('=' * 80)

        result = pipeline.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))