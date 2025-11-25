# file: ipg_pipeline.py
from extractor_module import process_question as extract_entities
from llm_processor import LLMProcessor
from config import DEFAULT_PARAM_GLOSSARY
from app.model_normalization import load_canonical_models  # your loader

class IPGPipeline:
    def __init__(self, llm_model="gpt-4o-mini"):
        parameters_list_with_type = self.format_parameters_list(DEFAULT_PARAM_GLOSSARY)
        models_list_with_type = self.build_models_list_with_type()
        self.llm = LLMProcessor(parameters_list_with_type, models_list_with_type, model=llm_model)

    @staticmethod
    def build_models_list_with_type(eq_type="inverter") -> str:
        """Generate IPG-compatible models list from canonical loader."""
        canon_models = load_canonical_models()
        lines = [f"{normalized} ({eq_type})" for normalized in canon_models.values()]
        return "\n".join(lines)

    @staticmethod
    def format_parameters_list(param_dict: dict, eq_type="inverter") -> str:
        """
        Converts a parameter glossary dict into a string suitable for SYSTEM_PROMPT.

        param_dict: {parameter_key: [synonyms]}
        eq_type: equipment type to include in parentheses
        """
        lines = []
        for key, synonyms in param_dict.items():
            lines.append(f'{key} ({eq_type}): {synonyms}')
        return "\n".join(lines)

    def process(self, text: str):
        """
        Full IPG pipeline:
        1) Extract entities via extractor_module
        2) Send to LLM to get STATUS / INTENT / PARAM_BINDINGS
        3) Return combined result
        """
        extraction_result = extract_entities(text)
        extracted_entities = extraction_result["extracted_entities"]


        llm_result = self.llm.process_question(
            extracted_entities
        )

        return {
            "question_raw": text,
            "extracted_entities": extracted_entities,
            "llm_decision": llm_result,
            "routing": extraction_result.get("routing", {})
        }


# -------- Example usage --------
if __name__ == "__main__":
    text = "Який максимальний струм заряджання/розряджання АКБ на інверторі LuxPwer LXP-LB-EU 10k?"
    pipeline = IPGPipeline()
    result = pipeline.process(text)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
