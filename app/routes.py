# from fastapi import APIRouter, BackgroundTasks
# from app.models import Query
# from app.entity_extractor import extract_entities_spacy
# from app.background_task import process_for_annotation
#
# router = APIRouter()
#
# @router.post("/extract_entities")
# def extract_entities(query: Query, background_tasks: BackgroundTasks):
#     text = query.text
#     result = extract_entities_spacy(text)
#
#     #background_tasks.add_task(process_for_annotation, text)
#     return result

from fastapi import APIRouter, BackgroundTasks
from app.models import Query
from extractor_module import process_question   # ← use your big module

router = APIRouter()

@router.post("/extract_entities")
def extract_entities(query: Query, background_tasks: BackgroundTasks):
    text = query.text

    # run your complete pipeline from extractor_module.py
    result = process_question(text)

    # Optionally run async post-processing
    # background_tasks.add_task(process_for_annotation, text)

    return result
