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

from fastapi import APIRouter, BackgroundTasks, HTTPException
import logging
from app.models import Query
from pipeline.ipg_pipeline import IPGPipeline

logger = logging.getLogger("ipg_pipeline")

router = APIRouter()
pipeline = IPGPipeline()


@router.post("/extract_entities")
def extract_entities(query: Query, background_tasks: BackgroundTasks):
    """
    Extract entities and process query through the IPG pipeline.
    
    Args:
        query: Query object containing the text to process
        background_tasks: FastAPI background tasks for async operations
        
    Returns:
        Dictionary with extracted entities, status, and routing information
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        if not query.text or not query.text.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        logger.info(f"Request received for query: {query.text[:80]}...")
        
        # Run the full IPG pipeline
        result = pipeline.process(query.text)
        
        logger.info(f"Query successfully processed - returning result")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query: {str(e)}"
        )
