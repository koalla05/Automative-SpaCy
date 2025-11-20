# file: extractor_module.py
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from app.model_normalization import normalize_model
import spacy

from app.normalization import clean_word, normalize_entity
from config import DEFAULT_PARAM_GLOSSARY

# Thresholds
PARAMETER_CONFIDENCE_THRESHOLD = 0.75
FUZZY_MATCH_THRESHOLD = 85  # Threshold for fuzzy matching
EXACT_MATCH_THRESHOLD = 95  # Threshold for considering a match as exact
MIN_WORD_LENGTH_FOR_FUZZY = 5  # Minimum word length to attempt fuzzy matching
MIN_SYNONYM_LENGTH_FOR_FUZZY = 5  # Minimum synonym length for fuzzy matching

# Load spacy for position tracking
nlp = spacy.load("full_ner_model")

# Sentence transformers for semantic similarity (lazy load)
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


# Conjunction and punctuation patterns for splitting
SPLIT_PATTERNS = [
    r'\s+(?:і|та|и|а|й|,|;)\s+',  # Ukrainian/Russian conjunctions and punctuation
    r'\s+(?:and|or|,|;)\s+',  # English conjunctions and punctuation
]

CONJUNCTION_REGEX = re.compile('|'.join(SPLIT_PATTERNS), re.IGNORECASE)


# ------------- Utilities -------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return util.cos_sim(a, b).item()


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching - handle plurals and common variations"""
    text = text.lower().strip()

    # Ukrainian plural normalization
    text = re.sub(r'(ів|ами|ах|ям|ях)$', '', text)  # Remove plural endings
    text = re.sub(r'(и|і)$', '', text)  # Remove plural и/і

    # Russian plural normalization
    text = re.sub(r'(ов|ами|ах|ам|ях)$', '', text)

    # English plural normalization
    text = re.sub(r'(s|es)$', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_stopword_or_common(word: str) -> bool:
    """Check if word is a stopword or common word that shouldn't be matched as parameter"""
    stopwords = {
        # Ukrainian
        'які', 'який', 'яка', 'яке', 'які', 'що', 'чи', 'для', 'від', 'при',
        'під', 'над', 'про', 'без', 'через', 'після', 'перед', 'біля', 'коло',
        'поза', 'між', 'поміж', 'серед', 'вздовж', 'всередині',
        # Russian
        'какой', 'какая', 'какое', 'какие', 'что', 'для', 'от', 'при',
        'под', 'над', 'про', 'без', 'через', 'после', 'перед',
        # English
        'what', 'which', 'how', 'for', 'from', 'with', 'without', 'the',
        'this', 'that', 'these', 'those', 'and', 'or'
    }
    return word.lower().strip() in stopwords


def split_into_segments(text: str) -> List[Dict[str, Any]]:
    """
    Split text into segments by conjunctions and punctuation.
    Returns list of segments with their positions.
    """
    segments = []
    last_end = 0

    for match in CONJUNCTION_REGEX.finditer(text):
        if last_end < match.start():
            segment_text = text[last_end:match.start()].strip()
            if segment_text:
                segments.append({
                    "text": segment_text,
                    "start": last_end,
                    "end": match.start()
                })
        last_end = match.end()

    # Add final segment
    if last_end < len(text):
        segment_text = text[last_end:].strip()
        if segment_text:
            segments.append({
                "text": segment_text,
                "start": last_end,
                "end": len(text)
            })

    # If no splits found, return entire text as one segment
    if not segments:
        segments = [{"text": text, "start": 0, "end": len(text)}]

    return segments


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
            "position": ent.start_char,
            "end_position": ent.end_char
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


# ------------- Enhanced parameter extraction with fuzzy matching -------------
def find_parameters(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> List[Dict[str, Any]]:
    """
    Detect parameters in text using exact and fuzzy matching.
    Handles plurals and variations.
    """
    lower = text.lower()
    results = []

    # Build synonym map with normalized versions
    synonym_to_key = {}
    normalized_synonyms = {}

    for key, syns in param_glossary.items():
        for syn in syns:
            syn_lower = syn.lower()
            synonym_to_key[syn_lower] = key
            normalized_syn = normalize_for_matching(syn_lower)
            if normalized_syn not in normalized_synonyms:
                normalized_synonyms[normalized_syn] = []
            normalized_synonyms[normalized_syn].append((syn_lower, key))

    found_positions = set()

    # Phase 1: Exact matching
    for syn, key in synonym_to_key.items():
        start = 0
        while True:
            idx = lower.find(syn, start)
            if idx == -1:
                break

            pos = idx

            # Check if already found nearby
            if any(abs(p - pos) < 5 for p in found_positions):
                start = idx + len(syn)
                continue

            # Grab phrase around parameter
            phrase_start = max(0, text.rfind(' ', 0, idx))
            phrase_end = text.find('.', idx + len(syn))
            if phrase_end == -1:
                phrase_end = min(len(text), idx + len(syn) + 100)
            phrase = text[phrase_start:phrase_end].strip()

            results.append({
                "key": key,
                "synonym_matched": syn,
                "confidence": 0.95,
                "position": pos,
                "end_position": pos + len(syn),
                "extracted_value": phrase,
                "match_type": "exact"
            })

            found_positions.add(pos)
            start = idx + len(syn)

    # Phase 2: Fuzzy matching for missed terms
    # Extract both single words and multi-word phrases
    candidates = []

    # Single words
    for match in re.finditer(r'\b[\w\-]+\b', text, re.UNICODE):
        word = match.group(0)
        # Only consider words with minimum length and not stopwords
        if len(word) >= MIN_WORD_LENGTH_FOR_FUZZY and not is_stopword_or_common(word):
            candidates.append({
                "text": word,
                "position": match.start(),
                "end_position": match.end(),
                "word_count": 1
            })

    # Multi-word phrases (2-5 words)
    words = re.findall(r'\b[\w\-]+\b', text, re.UNICODE)
    word_positions = [m.start() for m in re.finditer(r'\b[\w\-]+\b', text, re.UNICODE)]

    for i in range(len(words)):
        for length in range(2, min(6, len(words) - i + 1)):  # 2-5 word phrases
            phrase = ' '.join(words[i:i + length])
            if len(phrase) >= MIN_WORD_LENGTH_FOR_FUZZY:
                phrase_start = word_positions[i]
                phrase_end = word_positions[i + length - 1] + len(words[i + length - 1])
                candidates.append({
                    "text": phrase,
                    "position": phrase_start,
                    "end_position": phrase_end,
                    "word_count": length
                })

    for candidate in candidates:
        text_chunk = candidate["text"]
        pos = candidate["position"]
        word_count = candidate["word_count"]

        # Skip if already matched
        if any(abs(p - pos) < 5 for p in found_positions):
            continue

        text_normalized = normalize_for_matching(text_chunk)

        # Try fuzzy matching against all synonyms
        best_match = None
        best_score = 0
        best_key = None
        best_match_type = None

        for syn, key in synonym_to_key.items():
            syn_normalized = normalize_for_matching(syn)

            # Skip very short synonyms for fuzzy matching
            if len(syn_normalized) < MIN_SYNONYM_LENGTH_FOR_FUZZY:
                continue

            # Calculate word count similarity for multi-word phrases
            syn_word_count = len(syn_normalized.split())

            # Prefer candidates with similar word counts
            word_count_diff = abs(word_count - syn_word_count)

            # Skip if word counts are too different for multi-word synonyms
            if syn_word_count > 1 and word_count_diff > 2:
                continue

            # Skip if candidate is much shorter than synonym
            if len(text_normalized) < len(syn_normalized) * 0.4:
                continue

            # Calculate different fuzzy ratios
            ratio = fuzz.ratio(text_normalized, syn_normalized)
            partial_ratio = fuzz.partial_ratio(text_normalized, syn_normalized)
            token_sort_ratio = fuzz.token_sort_ratio(text_normalized, syn_normalized)

            # Choose scoring method based on synonym structure
            if syn_word_count > 1:
                # For multi-word synonyms, use token_sort_ratio which handles word order
                score = token_sort_ratio
                # Bonus for exact word count match
                if word_count == syn_word_count:
                    score = min(100, score + 5)
            else:
                # For single words, be strict
                score = ratio

            if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
                best_score = score
                best_match = syn
                best_key = key
                best_match_type = "fuzzy"

        if best_match and best_key:
            # Additional validation: check if already found this parameter
            # Skip duplicates with lower confidence
            already_found = any(r["key"] == best_key and r["confidence"] > best_score / 100.0 for r in results)
            if already_found:
                continue

            # Grab phrase around parameter
            phrase_start = max(0, text.rfind(' ', 0, pos))
            phrase_end = text.find('.', candidate["end_position"])
            if phrase_end == -1:
                phrase_end = min(len(text), candidate["end_position"] + 100)
            phrase = text[phrase_start:phrase_end].strip()

            confidence = best_score / 100.0

            results.append({
                "key": best_key,
                "synonym_matched": best_match,
                "confidence": confidence,
                "position": pos,
                "end_position": candidate["end_position"],
                "extracted_value": phrase,
                "match_type": best_match_type,
                "fuzzy_score": best_score
            })

            found_positions.add(pos)

    # Sort by position
    results.sort(key=lambda r: r["position"])

    # Deduplicate - keep highest confidence for each key
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


# ------------- Smart mapping of parameters to models -------------
def map_parameters_to_models(
        parameters: List[Dict[str, Any]],
        models: List[Dict[str, Any]],
        manufacturers: List[Dict[str, Any]],
        eq_types: List[Dict[str, Any]],
        text: str
) -> List[Dict[str, Any]]:
    """
    Intelligently map each parameter to its corresponding model/manufacturer/eq_type
    based on proximity and text segmentation.
    """
    # Split text into segments
    segments = split_into_segments(text)

    # Assign entities to segments
    for segment in segments:
        segment["models"] = []
        segment["manufacturers"] = []
        segment["eq_types"] = []
        segment["parameters"] = []

    # Assign models to segments
    for model in models:
        for segment in segments:
            if segment["start"] <= model["position"] < segment["end"]:
                segment["models"].append(model)
                break

    # Assign manufacturers to segments
    for manufacturer in manufacturers:
        for segment in segments:
            if segment["start"] <= manufacturer["position"] < segment["end"]:
                segment["manufacturers"].append(manufacturer)
                break

    # Assign eq_types to segments
    for eq_type in eq_types:
        for segment in segments:
            if segment["start"] <= eq_type["position"] < segment["end"]:
                segment["eq_types"].append(eq_type)
                break

    # Assign parameters to segments
    for param in parameters:
        for segment in segments:
            if segment["start"] <= param["position"] < segment["end"]:
                segment["parameters"].append(param)
                break

    # Build sub-queries based on segments
    sub_queries = []

    for segment in segments:
        seg_models = segment["models"]
        seg_manufacturers = segment["manufacturers"]
        seg_eq_types = segment["eq_types"]
        seg_parameters = segment["parameters"]

        # If segment has parameters
        if seg_parameters:
            # Determine model for this segment
            if seg_models:
                model_value = seg_models[0]["canonical"]
            elif models:  # Use closest model from entire text
                closest_model = min(models, key=lambda m: abs(m["position"] - segment["start"]))
                model_value = closest_model["canonical"]
            else:
                model_value = "ALL_LISTED"

            # Determine manufacturer
            if seg_manufacturers:
                manufacturer_value = seg_manufacturers[0]["value"]
            elif manufacturers:
                closest_manuf = min(manufacturers, key=lambda m: abs(m["position"] - segment["start"]))
                manufacturer_value = closest_manuf["value"]
            else:
                manufacturer_value = ""

            # Determine equipment type
            if seg_eq_types:
                eq_type_value = seg_eq_types[0]["value"]
            elif eq_types:
                closest_eq = min(eq_types, key=lambda e: abs(e["position"] - segment["start"]))
                eq_type_value = closest_eq["value"]
            else:
                eq_type_value = None

            # Create sub-query for each parameter in this segment
            for param in seg_parameters:
                sub_queries.append({
                    "manufacturer": manufacturer_value,
                    "model": model_value,
                    "equipment_type": eq_type_value,
                    "parameter": param["key"],
                    "original_part": param.get("extracted_value", ""),
                    "confidence": param.get("confidence", 0.0),
                    "match_type": param.get("match_type", "unknown")
                })

    # If no parameters found in segments, use proximity-based mapping
    if not sub_queries:
        for param in parameters:
            # Find closest model
            if models:
                closest_model = min(models, key=lambda m: abs(m["position"] - param["position"]))
                model_value = closest_model["canonical"]
            else:
                model_value = "ALL_LISTED"

            # Find closest manufacturer
            if manufacturers:
                closest_manuf = min(manufacturers, key=lambda m: abs(m["position"] - param["position"]))
                manufacturer_value = closest_manuf["value"]
            else:
                manufacturer_value = ""

            # Find closest eq_type
            if eq_types:
                closest_eq = min(eq_types, key=lambda e: abs(e["position"] - param["position"]))
                eq_type_value = closest_eq["value"]
            else:
                eq_type_value = None

            sub_queries.append({
                "manufacturer": manufacturer_value,
                "model": model_value,
                "equipment_type": eq_type_value,
                "parameter": param["key"],
                "original_part": param.get("extracted_value", ""),
                "confidence": param.get("confidence", 0.0),
                "match_type": param.get("match_type", "unknown")
            })

    return sub_queries


# ------------- Build routing with enhanced mapping -------------
def build_routing(ner_entities: Dict[str, List[Dict[str, Any]]],
                  parameters: List[Dict[str, Any]],
                  text: str) -> Dict[str, Any]:
    """
    Map parameters to the closest model and manufacturer using smart segmentation.
    """
    manufacturers = ner_entities.get("MANUFACTURER", [])
    models = ner_entities.get("MODEL", [])
    eq_types = ner_entities.get("EQ_TYPE", [])

    if not manufacturers and not models and not parameters:
        return {
            "recommended_strategy": "noEntities",
            "message": "No entities detected",
            "options": []
        }

    # Normalize models first
    for m in models:
        m["canonical"] = normalize_model(m["original_value"])

    # Smart mapping
    sub_queries = map_parameters_to_models(
        parameters, models, manufacturers, eq_types, text
    )

    if len(sub_queries) == 0:
        return {
            "recommended_strategy": "noEntities",
            "message": "No valid parameter-model mappings found",
            "options": []
        }
    elif len(sub_queries) == 1:
        return {
            "recommended_strategy": "single_query",
            "sub_query": sub_queries[0]
        }
    else:
        return {
            "recommended_strategy": "multi_query",
            "sub_queries": sub_queries
        }


# ------------- Main processing function -------------
def process_question(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> Dict[str, Any]:
    """
    Main function to process a question and return complete JSON structure
    """
    question_raw = text

    # 1. Enhanced NER with metadata
    ner_entities = extract_entities_with_metadata(text)

    # 2. Parameter extraction with fuzzy matching
    params_extracted = find_parameters(text, param_glossary)

    # 3. Build extracted_entities structure
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
        "equipment_type": ner_entities.get("EQ_TYPE", []),
        "parameters": params_extracted
    }

    # 4. Build routing with smart mapping
    routing = build_routing(ner_entities, params_extracted, text)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "question_raw": question_raw,
        "extracted_entities": extracted_entities,
        "routing": routing
    }


# ------------- Example usage -------------
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Які технічні характеристики Dyness A48100? який максимальний струм заряду і ємність в Ah?",
        "Який максимального розрядного струму для інвертора LuxPower SNA 6000?",
        "Яка ємність у Deye BOS-G25 та вага у Sofar HYD 10KTL?",
        "Максимальний струм заряджання і вага для Pylontech US5000",
        "Weight and capacity of BYD Battery-Box Premium HVS 7.7"
    ]

    import json

    for q in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {q}")
        print('=' * 80)
        out = process_question(q)
        print(json.dumps(out, ensure_ascii=False, indent=2))