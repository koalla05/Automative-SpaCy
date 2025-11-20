# file: test_extractor_module.py
import pytest
from extractor_module import (
    classify_intent,
    extract_entities_with_metadata,
    find_parameters,
    build_routing,
    process_question
)

# Mock glossary for testing
TEST_GLOSSARY = {
    "max_current": ["максимальний струм", "max current", "макс. струм"],
    "capacity": ["ємність", "capacity", "battery size"]
}

# ------------------ Tests ------------------

def test_classify_intent_passport():
    text = "Який максимальний струм заряду для Dyness A48100?"
    result = classify_intent(text, passport_templates=["паспорт батареї"])
    assert "intent" in result
    assert result["intent"] in ["passport", "other", "unknown"]
    assert 0 <= result["confidence"] <= 1


def test_extract_entities_with_metadata_basic():
    text = "Dyness A48100 має струм 50A"
    entities = extract_entities_with_metadata(text)
    assert "MANUFACTURER" in entities or "MODEL" in entities
    # Check metadata keys
    for label, items in entities.items():
        for e in items:
            assert "value" in e
            assert "confidence" in e
            assert "position" in e
            if label == "MODEL":
                assert "original_value" in e


def test_find_parameters_exact_and_fuzzy():
    text = "Максимальний струм батареї Dyness A48100 50A, ємність 100 Ah"
    params = find_parameters(text, param_glossary=TEST_GLOSSARY)
    keys = [p["key"] for p in params]
    assert "max_current" in keys
    assert "capacity" in keys
    for p in params:
        assert "confidence" in p
        assert 0 <= p["confidence"] <= 1
        assert "position" in p
        assert "extracted_value" in p


def test_build_routing_batch_and_single():
    ner_entities = {
        "MANUFACTURER": [{"value": "Dyness", "confidence": 0.9, "position": 0}],
        "MODEL": [
            {"value": "A48100", "confidence": 0.95, "position": 7, "original_value": "A48100"},
            {"value": "A48200", "confidence": 0.95, "position": 15, "original_value": "A48200"}
        ]
    }
    parameters = [{"key": "max_current", "extracted_value": "50A", "confidence": 0.95, "position": 20}]
    routing = build_routing(ner_entities, parameters)
    assert routing["recommended_strategy"] == "multi_query"
    assert all("manufacturer" in sq and "model" in sq and "parameter" in sq for sq in routing["sub_queries"])


def test_process_question_integration():
    text = "Які технічні характеристики Dyness A48100? Максимальний струм і ємність батареї"
    result = process_question(text, param_glossary=TEST_GLOSSARY)
    assert "status" in result
    assert "question_intent" in result
    assert "extracted_entities" in result
    assert "routing" in result
    # Check routing structure
    routing = result["routing"]
    assert routing["recommended_strategy"] in ["single_query", "multi_query", "noEntities"]


if __name__ == "__main__":
    pytest.main()
