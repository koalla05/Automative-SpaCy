import spacy
from typing import Dict, List
from core.normalization.entity_normalization import clean_word, normalize_entity

nlp = spacy.load("full_ner_model")

def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    grouped: Dict[str, List[str]] = {}

    for ent in doc.ents:
        label = ent.label_
        cleaned = clean_word(ent.text)
        normalized = normalize_entity(cleaned, label)

        if label not in grouped:
            grouped[label] = []

        if normalized not in grouped[label]:
            grouped[label].append(normalized)

    return grouped
