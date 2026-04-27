from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.routes import router
from app.logging_config import setup_logging
from pipeline.models import ModelManager

# Setup logging
logger = setup_logging(log_level=logging.INFO, use_colors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown events.
    Manages model loading and cleanup.
    """
    # ============ STARTUP ============
    logger.info("🚀 Application starting up...")
    try:
        # Load the spaCy model on startup
        model = ModelManager.get_nlp()
        logger.info(f"✅ Loaded NER model successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        raise
    
    yield  # Application runs here
    
    # ============ SHUTDOWN ============
    logger.info("🛑 Application shutting down...")
    ModelManager.cleanup()
    logger.info("✅ Application shutdown complete")


app = FastAPI(
    title="IPG Pipeline API",
    description="Equipment Parameter Extraction and Query Processing",
    lifespan=lifespan
)

app.include_router(router)

logger.info("FastAPI application initialized")
