# file: ipg_pipeline.py
import logging
import time
import os
from typing import Optional

from pipeline.exctractors.parameter_extractor import process_question as extract_entities
from pipeline.processors.llm_processor import (
    build_param_bindings_logic,
    determine_intent_logic
)
from pipeline.processors.hybrid_classifier import classify as hybrid_classify

logger = logging.getLogger("ipg_pipeline")


class IPGPipeline:
    def __init__(self, openai_client: Optional[object] = None):
        """
        Initialize pipeline with optional OpenAI client for hybrid classification.
        
        Args:
            openai_client: Optional OpenAI client. If not provided, will attempt to
                          import from OPENAI_API_KEY environment variable.
                          If unavailable, falls back to keyword-only classification.
        """
        self.openai_client = openai_client
        
        # Try to initialize OpenAI client if not provided
        if self.openai_client is None and os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                logger.info("OpenAI client initialized from OPENAI_API_KEY")
            except ImportError:
                logger.warning("OpenAI package not available - using keyword-only classification")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e} - using keyword-only classification")
        
        logger.info("IPGPipeline initialized (hybrid classifier enabled)")

    def process(self, text: str):
        """
        Process a user query through the entire pipeline.
        
        Uses hybrid classification: keyword-based detection with optional LLM fallback
        for "complex" queries when OpenAI client is available.
        
        Args:
            text: User query text
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"Processing query: {text[:100]}..." if len(text) > 100 else f"Processing query: {text}")
        
        try:
            extraction_result = extract_entities(text)
            extracted_entities = extraction_result["extracted_entities"]

            routing = extraction_result.get("routing")

            # Use hybrid classifier: keyword-based with optional LLM fallback
            status, classification_meta = hybrid_classify(
                text,
                extracted_entities,
                client=self.openai_client
            )
            
            logger.debug(
                f"Classification: status={status}, confidence=KW confidence, "
                f"llm_called={classification_meta.get('llm_called', False)}"
            )
            logger.debug(f"Classification metadata: {classification_meta}")

            if status in ["simple", "complex"]:
                param_bindings = build_param_bindings_logic(extracted_entities)
            else:
                param_bindings = []

            question_intent = determine_intent_logic(status, extracted_entities)
            logger.debug(f"Determined intent: {question_intent}")

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

            result = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "question_raw": text,
                "status": status,
                "question_intent": question_intent,
                "extracted_entities": extracted_entities,
                "routing": routing,
                "classification_meta": classification_meta  # Include classification details
            }
            
            elapsed = time.time() - start_time
            llm_info = f" (LLM upgraded)" if classification_meta.get("upgraded") else ""
            logger.info(f"Query processed successfully in {elapsed:.2f}s - Status: {status}{llm_info}")
            
            return result
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error processing query: {e} (elapsed: {elapsed:.2f}s)")
            raise


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
        "Яка максимальна напруга заряду на інверторі LuxPower SNA 5000?",
        "Привіт, яка максимальна вхідна потужність по фем на LuxPwer LXP-LB-EU 10k?",
        "Доброго вечора!",
        "Здоров",
        "Здоров, інвертор Deye SUN-6K-SG05LP1-EU вага",
        "Яка форма вхідної напруги у SNA6000?",
        "Какой струм заряду у SH6.0RS?",
        "Який вихідний струм у SG25CX-P2(5)",
        "Яка номінальна ємність акб А*год АКБ Dyness Dl5.0C?",
        "Яка максимальна кількість в одній системі АКБ Dybess DL5.0C?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {idx}: {query}")
        print('=' * 80)

        result = pipeline.process(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))