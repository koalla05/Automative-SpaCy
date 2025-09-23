from fastapi import APIRouter
from app.models import Query
from app.entity_extractor import extract_entities_spacy

router = APIRouter()

@router.post("/extract_entities")
def extract_entities(query: Query):
    return extract_entities_spacy(query.text)
