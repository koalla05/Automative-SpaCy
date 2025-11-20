# file: extractor_module.py
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from app.model_normalization import normalize_model
import spacy

from app.normalization import clean_word, normalize_entity
from app.entity_extractor import extract_entities_spacy
from config import PASSPORT_TEMPLATES, OTHER_INTENTS, DEFAULT_PARAM_GLOSSARY

# Thresholds
INTENT_SIMILARITY_THRESHOLD = 0.65
PARAMETER_CONFIDENCE_THRESHOLD = 0.75
FUZZY_MATCH_THRESHOLD = 85

EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load spacy for position tracking
nlp = spacy.load("full_ner_model")

# regex for values (numbers + possible units)
VALUE_REGEX = re.compile(r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>[A-Za-zА-Яа-я%°/μµhHkKWhkggVv]+)?")


# ------------- Utilities -------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(util.cos_sim(a, b).numpy())


def embed_text(text: str):
    return EMBED_MODEL.encode(text, convert_to_tensor=True)


# ------------- Intent recognition -------------
def classify_intent(text: str, passport_templates: List[str] = PASSPORT_TEMPLATES,
                    other_templates: List[str] = OTHER_INTENTS) -> Dict[str, Any]:
    """
    Returns:
      {
        "intent": "passport" | "other" | "unknown",
        "intent_label": str (best match text),
        "confidence": float (0..1),
        "status": "simple"|"complex"
      }
    """
    q_emb = embed_text(text)

    best_label = None
    best_score = 0.0
    best_bucket = "unknown"

    # Check passport_templates
    for tpl in passport_templates:
        tpl_emb = embed_text(tpl)
        score = cosine_sim(q_emb, tpl_emb)
        if score > best_score:
            best_score = score
            best_label = tpl
            best_bucket = "passport"

    # Check other templates
    for tpl in other_templates:
        tpl_emb = embed_text(tpl)
        score = cosine_sim(q_emb, tpl_emb)
        if score > best_score:
            best_score = score
            best_label = tpl
            best_bucket = "other"

    if best_score < INTENT_SIMILARITY_THRESHOLD:
        return {
            "intent": best_bucket,
            "intent_label": best_label if best_label else "unknown",
            "confidence": best_score,
            "status": "complex"
        }
    else:
        status = "simple" if best_bucket == "passport" else "complex"
        return {
            "intent": best_bucket,
            "intent_label": best_label,
            "confidence": best_score,
            "status": status
        }


# ------------- Enhanced NER with confidence and position -------------
def extract_entities_with_metadata(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entities with confidence scores and positions
    Returns: {
        "MANUFACTURER": [{"value": str, "confidence": float, "position": int}, ...],
        "MODEL": [{"value": str, "confidence": float, "position": int, "original_value": str}, ...],
        "EQ_TYPE": [{"value": str, "confidence": float, "position": int}]
    }
    """
    doc = nlp(text)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for ent in doc.ents:
        label = ent.label_
        cleaned = clean_word(ent.text)
        normalized = normalize_entity(cleaned, label)

        # Calculate confidence based on entity length and context
        confidence = min(0.95, 0.7 + (len(ent.text) / 50))

        entity_dict = {
            "value": normalized,
            "confidence": confidence,
            "position": ent.start_char
        }

        # For models, keep original value
        if label == "MODEL":
            entity_dict["original_value"] = ent.text

        if label not in grouped:
            grouped[label] = []

        # Avoid duplicates
        if not any(e["value"] == normalized for e in grouped[label]):
            grouped[label].append(entity_dict)

    return grouped


# ------------- Parameter extraction with fuzzy matching -------------
def find_parameters(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> List[Dict[str, Any]]:
    """
    Detect parameters in text and return full phrase containing them.
    """
    lower = text.lower()
    results = []

    # Build synonym map
    synonym_to_key = {syn.lower(): key for key, syns in param_glossary.items() for syn in syns}

    found_positions = set()

    for syn, key in synonym_to_key.items():
        start = 0
        while True:
            idx = lower.find(syn, start)
            if idx == -1:
                break

            pos = idx + 1
            if any(abs(p - pos) < 5 for p in found_positions):
                start = idx + len(syn)
                continue

            # Grab phrase around parameter
            phrase_start = max(0, text.rfind(' ', 0, idx))
            phrase_end = text.find('.', idx + len(syn))
            if phrase_end == -1:
                phrase_end = len(text)
            phrase = text[phrase_start:phrase_end].strip()

            results.append({
                "key": key,
                "synonym_matched": syn,
                "confidence": 0.95,
                "position": pos,
                "extracted_value": phrase
            })

            found_positions.add(pos)
            start = idx + len(syn)

    # Sort and deduplicate
    results.sort(key=lambda r: r["position"])
    final_results = []
    seen_keys = {}
    for r in results:
        key = r["key"]
        if key not in seen_keys or r["confidence"] > seen_keys[key]["confidence"]:
            if key in seen_keys:
                final_results.remove(seen_keys[key])
            seen_keys[key] = r
            final_results.append(r)

    return final_results



# ------------- Build routing with batch support -------------
def build_routing(ner_entities: Dict[str, List[Dict[str, Any]]],
                  parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map parameters to the closest model and manufacturer, using canonical model names.
    """
    manufacturers = ner_entities.get("MANUFACTURER", [])
    models = ner_entities.get("MODEL", [])

    if not manufacturers and not models and not parameters:
        return {"recommended_strategy": "noEntities", "message": "No entities detected", "options": []}

    # Normalize models first
    for m in models:
        m["canonical"] = normalize_model(m["original_value"])

    sub_queries = []

    for param in parameters:
        # find closest model
        closest_model = None
        min_dist = float('inf')
        for model in models:
            dist = abs(param["position"] - model["position"])
            if dist < min_dist:
                min_dist = dist
                closest_model = model

        canonical_name = closest_model["canonical"] if closest_model else "ALL_LISTED"
        manufacturer_name = manufacturers[0]["value"] if manufacturers else ""

        sub_queries.append({
            "manufacturer": manufacturer_name,
            "model": canonical_name,
            "parameter": param["key"],
            "original_part": param.get("extracted_value", "")
        })

    if len(sub_queries) == 1:
        return {"recommended_strategy": "single_query", "sub_query": sub_queries[0]}
    else:
        return {"recommended_strategy": "multi_query", "sub_queries": sub_queries}

# ------------- Main processing function -------------
def process_question(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> Dict[str, Any]:
    """
    Main function to process a question and return complete JSON structure
    """
    question_raw = text

    # 1. Intent classification
    intent_info = classify_intent(text)

    # 2. Enhanced NER with metadata
    ner_entities = extract_entities_with_metadata(text)

    # 3. Parameter extraction with fuzzy matching
    params_extracted = find_parameters(text, param_glossary)

    # 4. Build extracted_entities structure
    extracted_entities = {
        "manufacturer": ner_entities.get("MANUFACTURER", []),
        "model": [
            {
                "value": normalize_model(m["original_value"]),
                "confidence": m["confidence"],
                "position": m["position"],
                "original_value": m["original_value"]
            }
            for m in ner_entities.get("MODEL", [])
        ],
        "equipment_type": ner_entities.get("EQ_TYPE", [None])[0],
        "parameters": params_extracted
    }

    # 5. Build routing
    routing = build_routing(ner_entities, params_extracted)

    # 6. Determine final status
    status = intent_info["status"]
    if routing.get("recommended_strategy") == "noEntities":
        status = "no_entities"

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "question_raw": question_raw,
        "status": status,
        "question_intent": intent_info,
        "extracted_entities": extracted_entities,
        "routing": routing
    }


# ------------- Example usage -------------
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Які технічні характеристики Dyness A48100? який максимальний струм заряду і ємність в Ah?",
        "Який максимальний розрядний струм для інвертора LuxPower SNA 6000?",
        "Яка ємність у Deye BOS-G25 та вага у Sofar HYD 10KTL?"
    ]

    import json

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {q}")
        print('=' * 60)
        out = process_question(q)
        print(json.dumps(out, ensure_ascii=False, indent=2))