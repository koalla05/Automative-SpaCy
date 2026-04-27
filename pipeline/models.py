# file: pipeline/models.py
"""
Centralized model management to prevent duplicate loading and ensure consistent lifecycle.
"""

import logging
from pathlib import Path
from typing import Optional
from spacy.language import Language
import spacy

logger = logging.getLogger("ipg_pipeline")

class ModelManager:
    """
    Singleton pattern for managing spaCy NER model.
    Ensures only one model instance is loaded across the entire application.
    """
    _models = {}
    
    @classmethod
    def get_nlp(cls) -> Language:
        """
        Get or initialize the spaCy NLP model.
        
        Returns:
            spacy.Language: Loaded NLP model
            
        Raises:
            RuntimeError: If model cannot be loaded
        """
        if "nlp" not in cls._models:
            try:
                model_path = Path(__file__).resolve().parent.parent / "models" / "full_ner_model"
                
                if not model_path.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_path}")
                
                logger.info(f"Loading spaCy model from: {model_path}")
                cls._models["nlp"] = spacy.load(str(model_path))
                logger.info("✅ SpaCy model loaded successfully")
                
            except FileNotFoundError as e:
                logger.error(f"Model file not found: {e}")
                raise RuntimeError(f"Failed to load spaCy model: {e}")
            except OSError as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Failed to load spaCy model: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading model: {e}")
                raise RuntimeError(f"Failed to load spaCy model: {e}")
        
        return cls._models["nlp"]
    
    @classmethod
    def cleanup(cls) -> None:
        """
        Clean up loaded models. Call this on application shutdown.
        """
        if cls._models:
            logger.info("Cleaning up models")
            cls._models.clear()
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if model is already loaded."""
        return "nlp" in cls._models
