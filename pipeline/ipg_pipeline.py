# file: ipg_pipeline.py
from pipeline.exctractors.parameter_extractor import process_question as extract_entities
from pipeline.processors.llm_processor import LLMProcessor, detect_question_type
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

    def build_final_routing(self, llm_result: dict, extracted_entities: dict, text: str) -> dict:
        """
        Build routing based on LLM's param_bindings decision.
        Uses the actual model names from NER (not concatenated manufacturer+model).
        """
        param_bindings = llm_result.get("param_bindings", [])

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
                    "model": model,  # Use model name exactly as LLM provided (from NER)
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

    def build_parameters_from_llm(self, fuzzy_parameters: list, llm_result: dict) -> list:
        """
        Build final parameters list using LLM's param_bindings as the source of truth.
        - If fuzzy found the parameter, use its metadata (confidence, position, etc.)
        - If LLM added a parameter fuzzy missed, create it with LLM confidence
        - If fuzzy found a parameter LLM rejected, discard it
        """
        param_bindings = llm_result.get("param_bindings", [])

        # Build mapping: param_key -> [models]
        param_to_models = {}
        llm_confirmed_params = set()

        for binding in param_bindings:
            model = binding.get("model", "")
            for param in binding.get("parameters", []):
                llm_confirmed_params.add(param)
                if param not in param_to_models:
                    param_to_models[param] = []
                param_to_models[param].append(model)

        # Index fuzzy parameters by key for quick lookup
        fuzzy_by_key = {p["key"]: p for p in fuzzy_parameters}

        # Build final parameter list
        final_parameters = []

        for param_key in llm_confirmed_params:
            # Check if fuzzy matching found this parameter
            if param_key in fuzzy_by_key:
                # Use fuzzy's metadata
                param = fuzzy_by_key[param_key].copy()
            else:
                # LLM found it but fuzzy didn't - create new entry
                param = {
                    "key": param_key,
                    "confidence": 0.90,  # High confidence from LLM
                    "match_type": "llm_detected",
                    "synonym_matched": "detected by LLM from context"
                }

            # Add model mapping
            models = param_to_models.get(param_key, ["UNKNOWN"])
            if len(models) > 1:
                param["batch"] = True
                param["mapped_to_model"] = ", ".join(models)
            else:
                param["mapped_to_model"] = models[0]

            final_parameters.append(param)

        return final_parameters

    def process(self, text: str):
        """
        Full IPG pipeline:
        1) Detect question_type by keywords
        2) Extract entities via extractor_module
        3) Send to LLM to get STATUS / INTENT / PARAM_BINDINGS (skip if compat)
        4) Enrich parameters with mappings
        5) Build routing from LLM param_bindings
        6) Return schema-compliant result
        """
        # Step 1: Detect question type (NEW)
        question_type = detect_question_type(text)

        # Step 2: NER + fuzzy extraction
        extraction_result = extract_entities(text)
        extracted_entities = extraction_result["extracted_entities"]

        # Step 3: LLM processing (skip if compat)
        if question_type == "compat":
            # For compatibility queries, skip LLM entirely
            llm_result = {
                "status": "complex",
                "intent": "compatibility_query",
                "param_bindings": []  # No param bindings for compat
            }
        else:
            # Normal processing through LLM
            llm_result = self.llm.process_question(extracted_entities, text)

        # Step 4: Build parameters from LLM param_bindings (LLM is source of truth)
        final_params = self.build_parameters_from_llm(
            extracted_entities.get("parameters", []),
            llm_result
        )
        extracted_entities["parameters"] = final_params

        # Step 5: Build routing from LLM param_bindings
        routing = self.build_final_routing(llm_result, extracted_entities, text)

        # Step 6: Build final schema-compliant output
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "question_raw": text,
            "status": llm_result.get("status", "complex"),
            "question_type": question_type,  # NEW FIELD
            "question_intent": llm_result.get("intent", "uncertain"),
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
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print('=' * 80)
        result = pipeline.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))