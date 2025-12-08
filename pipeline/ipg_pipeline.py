# file: ipg_pipeline.py
from pipeline.exctractors.parameter_extractor import process_question as extract_entities
from pipeline.processors.llm_processor import (
    determine_status,
    build_param_bindings_logic,
    determine_intent_logic
)
from core.config import DEFAULT_PARAM_GLOSSARY
from core.normalization.model_normalization import load_canonical_models


class IPGPipeline:
    def __init__(self):
        """Initialize pipeline without LLM."""
        pass

    def find_closest_manufacturer(self, model_value: str, extracted_entities: dict) -> str:
        """
        Find the manufacturer that corresponds to the given model.
        Match by checking which manufacturer appears closest to the model in the original text.
        """
        manufacturers = extracted_entities.get("manufacturer", [])
        models = extracted_entities.get("model", [])

        if not manufacturers:
            return ""

        target_model = None
        for m in models:
            if m.get("value") == model_value or m.get("original_value", "").lower() in model_value.lower():
                target_model = m
                break

        if not target_model or "position" not in target_model:
            return manufacturers[0]["value"]

        model_pos = target_model["position"]
        closest_manuf = min(manufacturers, key=lambda m: abs(m.get("position", 999999) - model_pos))
        return closest_manuf["value"]

    def build_final_routing(
            self,
            status: str,
            param_bindings: list,
            extracted_entities: dict,
            text: str
    ) -> dict:
        """
        Build routing based on status and param_bindings.

        Args:
            status: "compat", "simple", or "complex"
            param_bindings: List of model->parameters mappings
            extracted_entities: All extracted entities
            text: Original query text

        Returns:
            Routing dictionary
        """
        # COMPAT status
        if status == "compat":
            return {
                "recommended_strategy": "compatibility_check",
                "message": "Compatibility query detected",
                "entities": {
                    "manufacturers": [m["value"] for m in extracted_entities.get("manufacturer", [])],
                    "models": [m["value"] for m in extracted_entities.get("model", [])],
                    "equipment_types": [e["value"] for e in extracted_entities.get("equipment_type", [])]
                }
            }

        # SIMPLE status
        if status == "simple":
            if not param_bindings:
                return {
                    "recommended_strategy": "error",
                    "message": "Simple query but no parameter bindings found",
                    "options": []
                }

            eq_types = extracted_entities.get("equipment_type", [])
            eq_type_value = eq_types[0]["value"] if eq_types else None

            # Build sub_queries from param_bindings
            sub_queries = []
            for binding in param_bindings:
                model = binding.get("model", "")
                parameters = binding.get("parameters", [])

                manufacturer_value = self.find_closest_manufacturer(model, extracted_entities)

                for param in parameters:
                    sub_queries.append({
                        "manufacturer": manufacturer_value,
                        "model": model,
                        "equipment_type": eq_type_value,
                        "parameter": param,
                        "original_part": text[:100]
                    })

            if len(sub_queries) == 1:
                return {
                    "recommended_strategy": "single_query",
                    "sub_query": sub_queries[0]
                }
            else:
                return {
                    "recommended_strategy": "multi_query",
                    "sub_queries": sub_queries
                }

        # COMPLEX status
        # Check if we have invalid models
        models = extracted_entities.get("model", [])
        valid_models = [m for m in models if m.get("value") is not None]
        has_invalid_models = len(models) > 0 and len(valid_models) == 0

        if has_invalid_models:
            message = "No entities"
        elif not models and not extracted_entities.get("parameters"):
            message = "No entities"
        else:
            message = "Query requires advanced processing"

        return {
            "recommended_strategy": "complex_query",
            "message": message
        }

    def process(self, text: str):
        """
        Full IPG pipeline (NO LLM):
        1) Extract entities via extractor_module
        2) Determine status by calculation (compat/simple/complex)
        3) Build param_bindings with logic
        4) Determine intent with logic
        5) Build routing
        6) Return schema-compliant result
        """
        # Step 1: Extract entities
        extraction_result = extract_entities(text)
        extracted_entities = extraction_result["extracted_entities"]

        # Step 2: Determine status by logic (compat/simple/complex)
        status = determine_status(extracted_entities, text)

        # Step 3: Build param_bindings (only for simple/complex with params)
        if status in ["simple", "complex"]:
            param_bindings = build_param_bindings_logic(extracted_entities)
        else:
            param_bindings = []

        # Step 4: Determine intent
        question_intent = determine_intent_logic(status, extracted_entities)

        # Step 5: Build routing
        routing = self.build_final_routing(
            status,
            param_bindings,
            extracted_entities,
            text
        )

        # Step 6: Return result
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "question_raw": text,
            "status": status,  # compat, simple, or complex
            "question_intent": question_intent,
            "extracted_entities": extracted_entities,
            "routing": routing
        }


# -------- Example usage --------
if __name__ == "__main__":
    import json

    pipeline = IPGPipeline()

    test_queries = [
        # Simple query - single model + parameter
        "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",

        # Compat query
        "Чи сумісний Pylontech US5000 з Victron MultiPlus?",

        # simple query - multi-model with parameters
        "Вага Dyness A48100 та максимальний струм для Pylontech US5000",

        # Complex query - no parameters
        "Які є інвертори Victron?",

        # Simple query - single model, single param
        "Вага Pylontech US5000",

        # Compat with "can I connect"
        "Can I connect Huawei LUNA2000 with Fronius Primo?",

        # Complex - invalid model (should show "No valid canonical model names")
        "Максимальний струм на Unknown Model XYZ-123",
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {idx}: {query}")
        print('=' * 80)

        result = pipeline.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))