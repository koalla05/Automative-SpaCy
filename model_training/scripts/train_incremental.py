"""
Incremental training script to improve existing NER model.

This script:
1. Loads your existing trained model
2. Adds new labeled examples
3. Continues training to improve performance
4. Validates and saves the updated model
"""
import json
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ============= CONFIGURATION =============
EXISTING_MODEL_PATH = "../models/full_ner_model"
NEW_LABELS_FILE = "model_training/data/labels.json"
OUTPUT_MODEL_PATH = "../models/full_ner_model_updated"

N_ITER = 10  # Fewer iterations for incremental training
BATCH_SIZE = (4.0, 32.0, 1.5)
DROPOUT = 0.3


# ============= HELPER FUNCTIONS =============

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\u00A0', ' ')  # non-breaking space
    text = text.replace('\\/', '/')  # escaped slash
    return text


def remove_duplicate_entities(entities: List[Tuple]) -> List[Tuple]:
    """Remove duplicate entity annotations."""
    seen = set()
    unique = []
    for start, end, label in entities:
        key = (start, end, label)
        if key not in seen:
            seen.add(key)
            unique.append((start, end, label))
    return unique


def load_training_data(filepath: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load and prepare training data from JSON file.

    Returns:
        List of (text, annotations) tuples
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_data = []
    skipped_count = 0

    for item in raw_data:
        text = normalize_text(item["text"])
        entities = []

        for label in item.get("label", []):
            start = label["start"]
            end = label["end"]
            ent_labels = label["labels"]

            for ent_label in ent_labels:
                entities.append((start, end, ent_label))

        entities = remove_duplicate_entities(entities)

        # Validate alignment
        try:
            nlp_tmp = spacy.blank("xx")
            doc = nlp_tmp.make_doc(text)
            _ = Example.from_dict(doc, {"entities": entities})
            train_data.append((text, {"entities": entities}))
        except Exception as e:
            skipped_count += 1
            print(f"[!] Skipped: {e}\n    Text: {text[:50]}...\n")

    print(f"✅ Loaded {len(train_data)} training examples")
    print(f"⚠️  Skipped {skipped_count} invalid examples\n")

    return train_data


def evaluate_model(nlp: spacy.Language, test_data: List[Tuple]) -> Dict[str, float]:
    """
    Evaluate model performance on test data.

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    from spacy.scorer import Scorer

    scorer = Scorer()
    examples = []

    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    scores = scorer.score(examples)

    return {
        "precision": scores.get("ents_p", 0.0),
        "recall": scores.get("ents_r", 0.0),
        "f1": scores.get("ents_f", 0.0)
    }


# ============= MAIN TRAINING FUNCTION =============

def train_incremental(
        model_path: str,
        train_data: List[Tuple],
        n_iter: int = 10,
        dropout: float = 0.3,
        output_path: str = None
):
    """
    Train existing model incrementally with new data.

    Args:
        model_path: Path to existing trained model
        train_data: List of (text, annotations) tuples
        n_iter: Number of training iterations
        dropout: Dropout rate for training
        output_path: Where to save updated model (default: overwrite original)
    """
    # Load existing model
    print(f"📂 Loading existing model from: {model_path}")
    nlp = spacy.load(model_path)

    # Get NER component
    if "ner" not in nlp.pipe_names:
        raise ValueError("Model doesn't have NER pipeline!")

    ner = nlp.get_pipe("ner")

    # Add any new labels from training data
    print("\n🏷️  Checking for new entity labels...")
    new_labels = set()
    for _, annotations in train_data:
        for ent in annotations.get("entities", []):
            label = ent[2]
            if label not in ner.labels:
                ner.add_label(label)
                new_labels.add(label)

    if new_labels:
        print(f"   Added new labels: {new_labels}")
    else:
        print("   No new labels to add")

    # Split data for validation (80/20)
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.8)
    train_set = train_data[:split_idx]
    validation_set = train_data[split_idx:]

    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_set)} examples")
    print(f"   Validation: {len(validation_set)} examples")

    # Evaluate before training
    print("\n📈 Performance BEFORE training:")
    before_scores = evaluate_model(nlp, validation_set)
    print(f"   Precision: {before_scores['precision']:.3f}")
    print(f"   Recall:    {before_scores['recall']:.3f}")
    print(f"   F1 Score:  {before_scores['f1']:.3f}")

    # Train the model
    print(f"\n🎓 Starting incremental training ({n_iter} iterations)...")

    # Disable other pipes during training for speed
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()

        for itn in range(n_iter):
            random.shuffle(train_set)
            losses = {}
            batches = minibatch(train_set, size=compounding(*BATCH_SIZE))

            for batch in batches:
                examples = []
                for text, annots in batch:
                    doc = nlp.make_doc(text)
                    annots["entities"] = remove_duplicate_entities(
                        annots["entities"]
                    )
                    try:
                        example = Example.from_dict(doc, annots)
                        examples.append(example)
                    except ValueError as e:
                        print(f"[!] Skipped during training: {e}")

                if examples:
                    nlp.update(examples, drop=dropout, losses=losses)

            # Print progress
            print(f"   Epoch {itn + 1}/{n_iter} - Loss: {losses.get('ner', 0):.3f}")

    # Evaluate after training
    print("\n📈 Performance AFTER training:")
    after_scores = evaluate_model(nlp, validation_set)
    print(f"   Precision: {after_scores['precision']:.3f} "
          f"({after_scores['precision'] - before_scores['precision']:+.3f})")
    print(f"   Recall:    {after_scores['recall']:.3f} "
          f"({after_scores['recall'] - before_scores['recall']:+.3f})")
    print(f"   F1 Score:  {after_scores['f1']:.3f} "
          f"({after_scores['f1'] - before_scores['f1']:+.3f})")

    # Save updated model
    output_path = output_path or model_path
    print(f"\n💾 Saving updated model to: {output_path}")
    nlp.to_disk(output_path)
    print("✅ Model saved successfully!")

    return nlp, after_scores


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Incrementally train existing NER model with new data"
    )
    parser.add_argument(
        "--model",
        default=EXISTING_MODEL_PATH,
        help="Path to existing model"
    )
    parser.add_argument(
        "--data",
        default=NEW_LABELS_FILE,
        help="Path to normalized labels JSON"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_MODEL_PATH,
        help="Path to save updated model"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=N_ITER,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
        help="Dropout rate"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Training data not found: {args.data}")

    # Load training data
    print("=" * 60)
    print("INCREMENTAL NER MODEL TRAINING")
    print("=" * 60)

    train_data = load_training_data(args.data)

    # Train model
    updated_model, scores = train_incremental(
        model_path=args.model,
        train_data=train_data,
        n_iter=args.iterations,
        dropout=args.dropout,
        output_path=args.output
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Updated model saved to: {args.output}")
    print(f"Final F1 Score: {scores['f1']:.3f}")