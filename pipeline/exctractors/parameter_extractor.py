# file: parameter_extractor.py
from typing import List, Dict, Any
from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from core.normalization.model_normalization import normalize_model
import spacy

from core.normalization.entity_normalization import clean_word, normalize_entity
from core.normalization.model_metadata import get_model_metadata
from core.config import DEFAULT_PARAM_GLOSSARY

# Thresholds
PARAMETER_CONFIDENCE_THRESHOLD = 0.75
FUZZY_MATCH_THRESHOLD = 80
EXACT_MATCH_THRESHOLD = 95
MIN_WORD_LENGTH_FOR_FUZZY = 4  # LOWERED from 5
MIN_SYNONYM_LENGTH_FOR_FUZZY = 4  # LOWERED from 5
BORDER_TOLERANCE = 2

CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_FILE_DIR.parent.parent
SPACY_MODEL_PATH = PROJECT_ROOT / "models" / "full_ner_model"

nlp = spacy.load(SPACY_MODEL_PATH)



SPLIT_PATTERNS = [
    r'\s+(?:і|та|и|а|й|,|;)\s+',  # Ukrainian/гівно conjunctions
    r'\s+(?:and|or|,|;)\s+',  # English conjunctions
]

CONJUNCTION_REGEX = re.compile('|'.join(SPLIT_PATTERNS), re.IGNORECASE)


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching - handle plurals and common variations"""
    text = text.lower().strip()

    text = re.sub(r'(ів|ами|ах|ям|ях)$', '', text)
    text = re.sub(r'(и|і)$', '', text)

    text = re.sub(r'(ов|ами|ах|ам|ях)$', '', text)

    text = re.sub(r'(s|es)$', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_stopword_or_common(word: str) -> bool:
    """Check if word is a stopword or common word that shouldn't be matched as parameter"""
    stopwords = {
        # Ukrainian
        'які', 'який', 'яка', 'яке', 'що', 'чи', 'для', 'від', 'при',
        'під', 'над', 'про', 'без', 'через', 'після', 'перед', 'біля', 'коло',
        'поза', 'між', 'поміж', 'серед', 'вздовж', 'всередині',
        # гівно
        'какой', 'какая', 'какое', 'какие', 'что', 'для', 'от', 'при',
        'под', 'над', 'про', 'без', 'через', 'после', 'перед',
        # English
        'what', 'which', 'how', 'for', 'from', 'with', 'without', 'the',
        'this', 'that', 'these', 'those', 'and', 'or', 'give', 'me', 'tell',
        'inverter', 'battery'
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

    if last_end < len(text):
        segment_text = text[last_end:].strip()
        if segment_text:
            segments.append({
                "text": segment_text,
                "start": last_end,
                "end": len(text)
            })

    if not segments:
        segments = [{"text": text, "start": 0, "end": len(text)}]

    return segments


def extract_entities_with_metadata(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entities with confidence scores and positions.
    Now uses improved model normalization with cleaning.
    Also enriches with metadata from CSV if canonical model is found.
    """
    doc = nlp(text)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for ent in doc.ents:
        label = ent.label_
        cleaned = clean_word(ent.text)
        normalized = normalize_entity(cleaned, label)

        if label == "MANUFACTURER" and is_stopword_or_common(normalized):
            continue

        confidence = min(0.95, 0.7 + (len(ent.text) / 50))

        entity_dict = {
            "value": normalized,
            "confidence": confidence,
            "position": ent.start_char,
            "end_position": ent.end_char
        }

        if label == "MODEL":
            entity_dict["original_value"] = ent.text
            # Use improved normalization with cleaning
            canonical = normalize_model(ent.text)
            entity_dict["value"] = canonical  # Will be None if not in canon

            # If canonical model found, get metadata from CSV
            if canonical:
                metadata = get_model_metadata(canonical)
                if metadata:
                    entity_dict["metadata"] = metadata

        if label not in grouped:
            grouped[label] = []

        # Avoid duplicates
        if not any(e["value"] == entity_dict["value"] for e in grouped[label]):
            grouped[label].append(entity_dict)

    return grouped


def find_parameters(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> List[Dict[str, Any]]:
    """
    Detect parameters in text using exact and fuzzy matching.
    Handles plurals and variations.
    """
    lower = text.lower()
    results = []

    synonym_to_key = {}
    normalized_synonyms = {}

    # Build synonym maps (strip syns to avoid leading/trailing spaces)
    for key, syns in param_glossary.items():
        for syn in syns:
            syn_lower = syn.lower().strip()
            synonym_to_key[syn_lower] = key
            normalized_syn = normalize_for_matching(syn_lower)
            if normalized_syn not in normalized_synonyms:
                normalized_synonyms[normalized_syn] = []
            normalized_synonyms[normalized_syn].append((syn_lower, key))

    found_positions = set()

    sorted_synonyms = sorted(synonym_to_key.items(), key=lambda x: len(x[0]), reverse=True)

    # Exact matches (use the actual matched substring indices)
    for syn, key in sorted_synonyms:
        if not syn:
            continue
        start = 0
        while True:
            idx = lower.find(syn, start)
            if idx == -1:
                break

            pos = idx

            if any(abs(p - pos) < 5 for p in found_positions):
                start = idx + len(syn)
                continue

            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = (idx + len(syn) >= len(text)) or not text[idx + len(syn)].isalnum()

            if not (before_ok and after_ok):
                start = idx + len(syn)
                continue

            phrase = syn  # the matched synonym itself
            phrase_start = idx
            phrase_end = idx + len(syn)

            results.append({
                "key": key,
                "synonym_matched": syn,
                "confidence": 0.95,
                "position": phrase_start,
                "end_position": phrase_end,
                "extracted_value": text[phrase_start:phrase_end].strip(),
                "match_type": "exact"
            })

            found_positions.add(pos)
            start = idx + len(syn)

    # Build fuzzy candidates (words and n-grams)
    candidates = []
    words = re.findall(r'\b[\w\-]+\b', text, re.UNICODE)
    word_positions = [m.start() for m in re.finditer(r'\b[\w\-]+\b', text, re.UNICODE)]

    for i in range(len(words)):
        for length in range(5, 1, -1):  # 5,4,3,2 words
            if i + length <= len(words):
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

        word = words[i]
        if len(word) >= MIN_WORD_LENGTH_FOR_FUZZY and not is_stopword_or_common(word):
            candidates.append({
                "text": word,
                "position": word_positions[i],
                "end_position": word_positions[i] + len(word),
                "word_count": 1
            })

    # Fuzzy matching: pick best synonym, then find the real occurrence in original text
    for candidate in candidates:
        text_chunk = candidate["text"]
        pos = candidate["position"]
        word_count = candidate["word_count"]

        if any(abs(p - pos) < 5 for p in found_positions):
            continue

        text_normalized = normalize_for_matching(text_chunk)

        best_match = None
        best_score = 0
        best_key = None
        best_match_type = None

        for syn, key in synonym_to_key.items():
            syn_normalized = normalize_for_matching(syn)

            if len(syn_normalized) < MIN_SYNONYM_LENGTH_FOR_FUZZY:
                continue

            syn_word_count = len(syn_normalized.split())
            word_count_diff = abs(word_count - syn_word_count)
            if syn_word_count > 1 and word_count_diff > 3:
                continue

            if len(text_normalized) < len(syn_normalized) * 0.3:
                continue

            ratio = fuzz.ratio(text_normalized, syn_normalized)
            partial_ratio = fuzz.partial_ratio(text_normalized, syn_normalized)
            token_sort_ratio = fuzz.token_sort_ratio(text_normalized, syn_normalized)
            token_set_ratio = fuzz.token_set_ratio(text_normalized, syn_normalized)

            if syn_word_count > 1:
                score = max(token_sort_ratio, token_set_ratio)
                if word_count == syn_word_count:
                    score = min(100, score + 5)
                if partial_ratio > score:
                    score = (score + partial_ratio) / 2
            else:
                score = max(ratio, partial_ratio)

            if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
                best_score = score
                best_match = syn
                best_key = key
                best_match_type = "fuzzy"

        if best_match and best_key:
            # Avoid duplicate-like entries (keeps using actual match positions below)
            already_found = any(
                r["key"] == best_key and
                abs(r["position"] - pos) < 50 and
                r["confidence"] > best_score / 100.0
                for r in results
            )
            if already_found:
                continue

            # Find all occurrences of the chosen synonym in the original text (lowercased),
            # choose the one closest to the candidate position.
            syn_l = best_match.lower().strip()
            occurrences = [m.start() for m in re.finditer(re.escape(syn_l), lower)]
            if occurrences:
                # pick occurrence closest to candidate position
                match_start = min(occurrences, key=lambda o: abs(o - pos))
                match_end = match_start + len(syn_l)
            else:
                # fallback: try to find within a small window around pos
                window_start = max(0, pos - 20)
                window_end = min(len(lower), pos + 20)
                found = lower.find(syn_l, window_start, window_end)
                if found != -1:
                    match_start = found
                    match_end = found + len(syn_l)
                else:
                    # last-resort fallback: use candidate bounds (but this should be rare)
                    match_start = pos
                    match_end = candidate["end_position"]

            # Extract exact substring from original text for extracted_value
            extracted_substring = text[match_start:match_end].strip()

            confidence = best_score / 100.0

            results.append({
                "key": best_key,
                "synonym_matched": best_match,
                "confidence": confidence,
                "position": match_start,
                "end_position": match_end,
                "extracted_value": extracted_substring,
                "match_type": best_match_type,
                "fuzzy_score": best_score
            })

            found_positions.add(match_start)

    # Sort and de-duplicate overlapping/conflicting results (preserve your original logic)
    results.sort(key=lambda r: r["position"])

    final_results = []
    seen_keys = {}

    for r in results:
        key = r["key"]
        pos = r["position"]
        end_pos = r["end_position"]

        is_overlapping = False
        for existing in final_results:
            existing_start = existing["position"]
            existing_end = existing["end_position"]

            if (existing_start <= pos < existing_end or
                    existing_start < end_pos <= existing_end or
                    (pos <= existing_start and end_pos >= existing_end)):

                is_overlapping = True

                if r["confidence"] > existing["confidence"]:
                    final_results.remove(existing)
                    if existing["key"] in seen_keys and seen_keys[existing["key"]] == existing:
                        del seen_keys[existing["key"]]
                    is_overlapping = False
                break

        if is_overlapping:
            continue

        if key in seen_keys:
            existing = seen_keys[key]
            distance = abs(pos - existing["position"])

            if distance > 50:
                final_results.append(r)
            elif r["confidence"] > existing["confidence"]:
                final_results.remove(existing)
                seen_keys[key] = r
                final_results.append(r)
        else:
            seen_keys[key] = r
            final_results.append(r)

    final_results.sort(key=lambda r: r["position"])

    return final_results


def map_parameters_to_models(parameters, models, manufacturers, eq_types, text):
    """
    Map each parameter to closest model using segment-aware logic.
    Fixed to properly handle parameters that span across segment boundaries.
    """
    # Normalize models - add canonical field
    for m in models:
        m["canonical"] = normalize_model(m.get("original_value", ""))

    # Filter valid models (canonical != None)
    valid_models = [m for m in models if m.get("canonical")]

    if not valid_models:
        # No valid models - return parameters with ALL_LISTED
        sub_queries = []
        for param in parameters:
            manufacturer_value = manufacturers[0]["value"] if manufacturers else ""
            eq_type_value = eq_types[0]["value"] if eq_types else None

            sub_queries.append({
                "manufacturer": manufacturer_value,
                "model": "ALL_LISTED",
                "equipment_type": eq_type_value,
                "parameter": param["key"],
                "original_part": param.get("extracted_value", ""),
                "confidence": param.get("confidence", 0.0),
                "match_type": param.get("match_type", "unknown")
            })
        return sub_queries

    # Split text into segments by conjunctions
    segments = split_into_segments(text)

    print(f"\n🔍 DEBUG: Found {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"  Segment {i + 1}: '{seg['text']}' (pos {seg['start']}-{seg['end']})")

    def overlaps_with_segment(ent_start, ent_end, seg_start, seg_end):
        return not (ent_end < seg_start - BORDER_TOLERANCE or
                    ent_start > seg_end + BORDER_TOLERANCE)

    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        segment["models"] = [
            m for m in valid_models
            if seg_start <= m.get("position", 0) < seg_end
        ]

        # For parameters, use overlap detection (they can span multiple words)
        segment["parameters"] = [
            p for p in parameters
            if overlaps_with_segment(
                p.get("position", 0),
                p.get("end_position", p.get("position", 0)),
                seg_start,
                seg_end
            )
        ]

        segment["manufacturers"] = [
            m for m in manufacturers
            if seg_start <= m.get("position", 0) < seg_end
        ]

        segment["eq_types"] = [
            e for e in eq_types
            if seg_start <= e.get("position", 0) < seg_end
        ]

        print(f"\n  Segment {segments.index(segment) + 1} entities:")
        print(f"    Models: {[m['canonical'] for m in segment['models']]}")
        print(f"    Params: {[p['key'] for p in segment['parameters']]}")
        print(f"    Param positions: {[(p['key'], p['position'], p['end_position']) for p in segment['parameters']]}")

    sub_queries = []

    for seg_idx, segment in enumerate(segments):
        seg_models = segment.get("models", [])
        seg_params = segment.get("parameters", [])
        seg_manufacturers = segment.get("manufacturers", [])
        seg_eq_types = segment.get("eq_types", [])

        if not seg_params:
            print(f"\n  ⚠️ Segment {seg_idx + 1}: No parameters, skipping")
            continue  # Skip segments without parameters

        print(f"\n  ✅ Segment {seg_idx + 1}: Processing {len(seg_params)} parameter(s)")

        for param in seg_params:
            if seg_models:
                closest_model = min(seg_models,
                                    key=lambda m: abs(m.get("position", 0) - param.get("position", 0)))
                print(f"    - {param['key']} → {closest_model['canonical']} (same segment)")
            else:
                closest_model = min(valid_models,
                                    key=lambda m: abs(m.get("position", 0) - param.get("position", 0)))
                print(f"    - {param['key']} → {closest_model['canonical']} (closest overall)")

            model_value = closest_model["canonical"]

            if seg_manufacturers:
                closest_manuf = min(seg_manufacturers,
                                    key=lambda m: abs(m.get("position", 0) - param.get("position", 0)))
                manufacturer_value = closest_manuf["value"]
            else:
                model_metadata = closest_model.get("metadata", {})
                manufacturer_value = model_metadata.get("manufacturer", "")
                if not manufacturer_value and manufacturers:
                    manufacturer_value = manufacturers[0]["value"]

            if seg_eq_types:
                eq_type_value = seg_eq_types[0]["value"]
            else:
                model_metadata = closest_model.get("metadata", {})
                eq_type_value = model_metadata.get("equipment_type")
                if not eq_type_value and eq_types:
                    eq_type_value = eq_types[0]["value"]

            sub_queries.append({
                "manufacturer": manufacturer_value,
                "model": model_value,
                "equipment_type": eq_type_value,
                "parameter": param["key"],
                "original_part": param.get("extracted_value", ""),
                "confidence": param.get("confidence", 0.0),
                "match_type": param.get("match_type", "unknown")
            })

    print(f"\n✅ Total sub-queries created: {len(sub_queries)}\n")
    return sub_queries


def build_routing(ner_entities, parameters, text):
    """
    Build routing structure from entities.
    Fixed to properly distinguish single vs multi query.
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

    sub_queries = map_parameters_to_models(parameters, models, manufacturers, eq_types, text)

    if len(sub_queries) == 0:
        return {
            "recommended_strategy": "noEntities",
            "message": "No valid parameter-model bindings",
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


def process_question(text: str, param_glossary: Dict[str, List[str]] = DEFAULT_PARAM_GLOSSARY) -> Dict[str, Any]:
    question_raw = text

    ner_entities = extract_entities_with_metadata(text)
    params_extracted = find_parameters(text, param_glossary)

    models = []
    for m in ner_entities.get("MODEL", []):
        model_dict = {
            "value": normalize_model(m["original_value"]),
            "confidence": m["confidence"],
            "position": m["position"],
            "original_value": m["original_value"]
        }
        if "metadata" in m:
            model_dict["metadata"] = m["metadata"]
        models.append(model_dict)

    manufacturers_ner = ner_entities.get("MANUFACTURER", [])
    eq_types_ner = ner_entities.get("EQ_TYPE", [])

    manufacturers = list(manufacturers_ner)
    eq_types = list(eq_types_ner)

    for model in models:
        if model.get("value") and "metadata" in model:
            metadata = model["metadata"]
            if not manufacturers and metadata.get("manufacturer"):
                manufacturers.append({
                    "value": metadata["manufacturer"],
                    "confidence": 0.95,
                    "position": model["position"],
                    "end_position": model["position"],
                    "source": "metadata"
                })
            if not eq_types and metadata.get("equipment_type"):
                eq_types.append({
                    "value": metadata["equipment_type"],
                    "confidence": 0.95,
                    "position": model["position"],
                    "end_position": model["position"],
                    "source": "metadata"
                })

    extracted_entities = {
        "manufacturer": manufacturers,
        "model": models,
        "equipment_type": eq_types,
        "parameters": params_extracted
    }

    routing = build_routing(
        {"MANUFACTURER": manufacturers, "MODEL": models, "EQ_TYPE": eq_types},
        params_extracted,
        text
    )

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "question_raw": question_raw,
        "extracted_entities": extracted_entities,
        "routing": routing
    }


if __name__ == "__main__":
    if __name__ == "__main__":
        test_queries = [
            "Який максимальний струм заряджання/розряджання АКБ на інверторі LuxPwer LXP-LB-EU 10k?",
            "Скільки MPPT трекерів та входів у інвертора Fronius Verto?",
            "напруга запуску на інверторі uawei SUN2000-15KTL-M2 та вага  SUN2000-15KTL-M2 ?"
            # "Який діапазон постійної вхідної напруги на інверторі Victron MultiPlus-II 48/3000/35-32 и 48/5000/70-50?",
            # # # === BASIC SINGLE PARAMETER QUERIES ===
            # "Яка ємність Dyness A48100?",
            # "Weight of Pylontech US5000",
            # "Максимальний струм заряджання для Growatt SPF 5000",
            # "What is the max charge current for Tesla Powerwall 2?",
            #
            # # === MULTIPLE PARAMETERS, SINGLE MODEL ===
            # "Які технічні характеристики Dyness A48100? який максимальний струм заряду і ємність в Ah?",
            # "Вага, ємність та максимальний струм розряду для BYD Battery-Box Premium HVS 10.2",
            # "Give me weight, capacity and voltage range for Pylontech US3000C",
            # "Максимальна потужність заряду, напруга абсорбції і float voltage для Victron MultiPlus II 48/5000",
            # #
            # # === MULTIPLE MODELS WITH DIFFERENT PARAMETERS ===
            # "Яка ємність у Deye BOS-G25 та вага у Sofar HYD 10KTL?",
            # "Максимальний струм заряджання для Pylontech US5000 і ємність для Dyness A48100",
            # "Compare weight of Growatt SPF 5000 and max AC power of Victron MultiPlus 48/3000",
            # "Вага Huawei LUNA2000-5-S0 та максимальна потужність Solax X3-Hybrid-8.0",
            #
            # # === COMPLEX MULTI-MODEL, MULTI-PARAMETER ===
            # "Ємність та вага для Dyness A48100, максимальний струм для Pylontech US5000 і напруга для BYD HVS 7.7",
            # "Які MPPT trackers у Fronius Symo Hybrid, efficiency для SolarEdge SE7600H та nominal AC power for Huawei SUN2000-5KTL",
            #
            # # === INVERTER-SPECIFIC QUERIES ===
            # "Скільки MPPT трекерів у інвертора Fronius Primo 8.2?",
            # "What is the max PV input power for SolarEdge SE10K?",
            # "Nominal AC voltage and frequency range for Growatt MIN 6000TL-XH",
            # "Максимальна вхідна напруга з панелей та кількість стрінгів на MPPT для Huawei SUN2000-10KTL-M1",
            # "Efficiency and max AC output power of Victron MultiPlus II 48/5000/70-50",
            #
            # # === BATTERY-SPECIFIC QUERIES ===
            # "Тип батареї та діапазон напруги для інвертора Deye SUN-12K-SG04LP3",
            # "Battery voltage range and charging strategy for Sofar HYD 6000-ES",
            # "Чи є захист від зворотної полярності у Pylontech US5000?",
            # "Does LuxPower SNA 5000 support battery temperature sensor?",
            #
            # # === PROTECTION FEATURES ===
            # "Які захисти є у інвертора Growatt SPF 5000 ES?",
            # "Anti-islanding protection and ground fault monitoring for SolarEdge SE7600H",
            # "Чи є PID recovery та string monitoring у Fronius Symo 10.0-3-M?",
            # "Arc fault protection and surge protection in Huawei SUN2000-8KTL-M1",
            #
            # # === PHYSICAL SPECIFICATIONS ===
            # "Габарити та вага Victron MultiPlus 48/5000",
            # "IP rating, dimensions and cooling type for Growatt MIN 11400TL-XH",
            # "Рівень шуму, робочий температурний діапазон та клас захисту для SMA Sunny Tripower 10.0",
            #
            # # === MIXED LANGUAGE QUERIES ===
            # "Яка capacity у Pylontech та weight для Dyness?",
            # "Max charge current для BYD HVS 10.2 і efficiency of Victron MultiPlus",
            # "Ємність в Ah for Tesla Powerwall and вага у кг для Sonnen Eco 10",
            #
            # # === QUERIES WITH TYPOS (testing fuzzy matching) ===
            # "Макимальний стурм зарядки для Pylontech US5000",  # typos
            # "Вага та емность для Dyness A48100",  # емность instead of ємність
            # "Waight and capasity of BYD Battery-Box",  # English typos
            #
            # # === QUERIES WITH PLURALS ===
            # "Які ємності у Pylontech US5000 та Dyness A48100?",  # plural form
            # "Weights of Pylontech US5000, BYD HVS 7.7 and Dyness A48100",
            # "Максимальні струми заряджання для інверторів Growatt та Victron",
            #
            # # === EDGE CASES ===
            # "Порівняти характеристики Dyness A48100 та BYD HVS 10.2",  # "compare specs"
            # "Що краще - Pylontech US5000 чи Dyness A48100?",  # opinion question
            # "Скільки коштує Victron MultiPlus 48/5000?",  # price question (not in glossary)
            # "Де купити Growatt SPF 5000?",  # "where to buy" (OTHER intent)
            # "Як налаштувати Fronius Symo 10.0?",  # configuration question (OTHER intent)
            #
            # # === NO ENTITIES ===
            # "Які бувають типи інверторів?",  # general question, no specific model
            # "Що таке MPPT?",  # definition question
            # "Hello, how are you?",  # irrelevant query
            #
            # # === EQUIPMENT TYPE QUERIES ===
            # "Максимальна потужність для гібридного інвертора Deye SUN-12K",
            # "Battery capacity for energy storage system Pylontech US5000",
            # "Вага для АКБ Dyness A48100 та інвертора Growatt SPF 5000",
            #
            # # === NUMERIC VALUES IN QUERY ===
            # "Чи підтримує Victron MultiPlus 48V батареї?",
            # "Can Fronius Primo handle 600V input voltage?",
            # "Чи може Growatt MIN 11400 працювати з 250A струмом?",
            #
            # # === LONG COMPLEX QUERIES ===
            # "Мені потрібно порівняти технічні характеристики трьох інверторів: Fronius Symo Hybrid 5.0 (цікавить максимальна потужність та ККД), SolarEdge SE7600H (хочу знати кількість MPPT та максимальну вхідну напругу) і Huawei SUN2000-8KTL (потрібна інформація про nominal AC power та захист від перенапруги)",
            #
            # # === ABBREVIATIONS AND TECHNICAL TERMS ===
            # "Max Isc per MPPT for Fronius Primo 8.2",
            # "THD and power factor range for Victron MultiPlus",
            # "ККД Euro та MPPT efficiency для SolarEdge SE10K",
            # "Voc max and MPP voltage range for Growatt MIN 6000TL-XH",
        ]

        import json

        print(f"\n{'#' * 80}")
        print(f"# RUNNING {len(test_queries)} TEST QUERIES")
        print(f"{'#' * 80}\n")

        for idx, q in enumerate(test_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test {idx}/{len(test_queries)}: {q}")
            print('=' * 80)
            out = process_question(q)
            print(json.dumps(out, ensure_ascii=False, indent=2))