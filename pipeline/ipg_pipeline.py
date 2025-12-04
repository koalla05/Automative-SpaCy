# file: ipg_pipeline.py
from pipeline.exctractors.parameter_extractor import process_question as extract_entities
from pipeline.processors.llm_processor import (
    LLMProcessor,
    detect_question_type,
    build_param_bindings_logic,
    determine_intent_logic
)
from core.config import DEFAULT_PARAM_GLOSSARY
from core.normalization.model_normalization import load_canonical_models


class IPGPipeline:
    def __init__(self, llm_model="gpt-4o-mini"):
        parameters_list = self.format_parameters_list(DEFAULT_PARAM_GLOSSARY)
        models_list = self.build_models_list()
        self.llm = LLMProcessor(parameters_list, models_list, model=llm_model)

    @staticmethod
    def build_models_list() -> str:
        """Generate IPG-compatible models list from canonical loader."""
        canon_models = load_canonical_models()
        lines = [f"{normalized}" for normalized in canon_models.values()]
        return "\n".join(lines)

    @staticmethod
    def format_parameters_list(param_dict: dict) -> str:
        """
        Converts a parameter glossary dict into a string suitable for SYSTEM_PROMPT.
        param_dict: {parameter_key: [synonyms]}
        """
        lines = []
        for key, synonyms in param_dict.items():
            lines.append(f'{key}: {synonyms}')
        return "\n".join(lines)

    def find_closest_manufacturer(self, model_value: str, extracted_entities: dict) -> str:
        """
        Find the manufacturer that corresponds to the given model.
        Match by checking which manufacturer appears closest to the model in the original text.
        """
        manufacturers = extracted_entities.get("manufacturer", [])
        models = extracted_entities.get("model", [])

        if not manufacturers:
            return ""

        # Find the model entity that matches model_value
        target_model = None
        for m in models:
            if m.get("value") == model_value or m.get("original_value", "").lower() in model_value.lower():
                target_model = m
                break

        if not target_model or "position" not in target_model:
            # Fallback: return first manufacturer
            return manufacturers[0]["value"]

        # Find closest manufacturer by position
        model_pos = target_model["position"]
        closest_manuf = min(manufacturers, key=lambda m: abs(m.get("position", 999999) - model_pos))
        return closest_manuf["value"]

    def build_final_routing(
            self,
            param_bindings: list,
            extracted_entities: dict,
            text: str,
            question_type: str = None
    ) -> dict:
        """
        Build routing based on param_bindings.

        Args:
            param_bindings: List of model->parameters mappings
            extracted_entities: All extracted entities
            text: Original query text
            question_type: "compat" or None

        Returns:
            Routing dictionary
        """
        # If compatibility query, return special routing
        if question_type == "compat":
            return {
                "recommended_strategy": "compatibility_check",
                "message": "Compatibility query detected",
                "entities": {
                    "manufacturers": [m["value"] for m in extracted_entities.get("manufacturer", [])],
                    "models": [m["value"] for m in extracted_entities.get("model", [])],
                    "equipment_types": [e["value"] for e in extracted_entities.get("equipment_type", [])]
                }
            }

        if not param_bindings:
            return {
                "recommended_strategy": "noEntities",
                "message": "No valid parameter bindings found",
                "options": []
            }

        # Find equipment type (usually same for all)
        eq_types = extracted_entities.get("equipment_type", [])
        eq_type_value = eq_types[0]["value"] if eq_types else None

        # Build sub_queries from param_bindings
        sub_queries = []
        for binding in param_bindings:
            model = binding.get("model", "")
            parameters = binding.get("parameters", [])

            # Find the correct manufacturer for this specific model
            manufacturer_value = self.find_closest_manufacturer(model, extracted_entities)

            for param in parameters:
                sub_queries.append({
                    "manufacturer": manufacturer_value,
                    "model": model,
                    "equipment_type": eq_type_value,
                    "parameter": param,
                    "original_part": text[:100]  # first 100 chars as context
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

    def process(self, text: str):
        """
        Full IPG pipeline:
        1) Detect question_type by keywords
        2) Extract entities via extractor_module
        3) Determine intent with logic (not LLM)
        4) Send to LLM ONLY if not compat (to get STATUS only)
        5) Build param_bindings with logic (not LLM)
        6) Build routing
        7) Return schema-compliant result
        """
        # Step 1: Detect question type
        question_type = detect_question_type(text)

        # Step 2: NER + fuzzy extraction
        extraction_result = extract_entities(text)
        extracted_entities = extraction_result["extracted_entities"]

        # Step 3: Determine intent with LOGIC (not LLM)
        question_intent = determine_intent_logic(question_type, extracted_entities)

        # Step 4: LLM processing (ONLY for STATUS, skip if compat)
        if question_type == "compat":
            # For compatibility queries, skip LLM entirely
            status = "complex"
        else:
            # Ask LLM only for STATUS
            llm_result = self.llm.process_question(extracted_entities, text)
            status = llm_result.get("status", "complex")

        # Step 5: Build param_bindings with LOGIC (not LLM)
        if question_type == "compat":
            param_bindings = []
        else:
            param_bindings = build_param_bindings_logic(extracted_entities)

        # Step 6: Build routing
        routing = self.build_final_routing(
            param_bindings,
            extracted_entities,
            text,
            question_type
        )

        # Step 7: Return complete result
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "question_raw": text,
            "status": status,
            "question_type": question_type,  # "compat" or None
            "question_intent": question_intent,  # determined by logic
            "extracted_entities": extracted_entities,
            "routing": routing
        }


# -------- Example usage --------
if __name__ == "__main__":
    import json

    pipeline = IPGPipeline()

    test_queries = [
        # Spec query
        "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",

        # Compatibility query
        "Чи сумісний Pylontech US5000 з Victron MultiPlus?",

        # Multi-model spec query
        "Вага Dyness A48100 та максимальний струм для Pylontech US5000",
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {idx}: {query}")
        print('=' * 80)

        result = pipeline.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))