import logging
import spacy
from typing import Dict, List
from core.normalization.entity_normalization import clean_word, normalize_entity
from pipeline.models import ModelManager

logger = logging.getLogger("ipg_pipeline")

def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """
    Extract entities using the spaCy NER model.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        Dictionary mapping entity labels to list of normalized values
    """
    try:
        nlp = ModelManager.get_nlp()
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
    except Exception as e:
        logger.error(f"Failed to extract entities with spaCy: {e}")
        return {}
