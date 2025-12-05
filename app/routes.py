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
from pipeline.ipg_pipeline import IPGPipeline

router = APIRouter()
pipeline = IPGPipeline()

@router.post("/extract_entities")
def extract_entities(query: Query, background_tasks: BackgroundTasks):
    text = query.text

    # run the full IPG pipeline
    result = pipeline.process(text)

    # Optionally run async post-processing
    # background_tasks.add_task(process_for_annotation, text)

    return result
