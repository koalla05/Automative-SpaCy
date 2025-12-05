"""
Safe incremental training workflow.
Run from model_training directory.
"""
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import spacy
from spacy.training.example import Example


class ModelTrainingWorkflow:
    """Manages safe incremental training workflow."""

    def __init__(
        self,
        model_path: str,
        training_data_path: str,
        backup_dir: str = None
    ):
        # Get the model_training directory (where this script's parent is)
        script_dir = Path(__file__).resolve().parent  # scripts/
        self.base_dir = script_dir.parent  # model_training/

        # Build absolute paths
        self.model_path = Path(model_path).resolve()
        self.training_data_path = Path(training_data_path).resolve()

        if backup_dir:
            self.backup_dir = Path(backup_dir).resolve()
        else:
            # Put backups next to the model
            self.backup_dir = self.model_path.parent / "backups"

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.temp_model_path = self.model_path.parent / f"{self.model_path.name}_temp"
        self.backup_path = None

        # Display paths
        print(f"\n📂 Configuration:")
        print(f"   Base directory: {self.base_dir}")
        print(f"   Model: {self.model_path}")
        print(f"   Training data: {self.training_data_path}")
        print(f"   Backups: {self.backup_dir}")

        # Validate
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found at: {self.model_path}\n"
                f"   Expected directory with meta.json inside"
            )

        if not self.training_data_path.exists():
            raise FileNotFoundError(
                f"❌ Training data not found at: {self.training_data_path}"
            )

        print(f"✅ All paths validated\n")

    def backup_model(self) -> Path:
        """Create timestamped backup of current model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.model_path.name}_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        print(f"📦 Creating backup: {backup_path}")
        shutil.copytree(self.model_path, backup_path)
        self.backup_path = backup_path
        print(f"✅ Backup created")

        return backup_path

    def validate_training_data(self) -> Dict[str, Any]:
        """Validate training data before training."""
        print("\n🔍 Validating training data...")

        with open(self.training_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load model to check entity labels
        nlp = spacy.load(str(self.model_path))
        ner = nlp.get_pipe("ner")
        existing_labels = set(ner.labels)

        stats = {
            "total_examples": len(data),
            "valid_examples": 0,
            "invalid_examples": 0,
            "entity_counts": {},
            "new_labels": set(),
            "errors": []
        }

        for idx, item in enumerate(data):
            try:
                text = item["text"]
                entities = []

                for label in item.get("label", []):
                    start = label["start"]
                    end = label["end"]
                    for ent_label in label["labels"]:
                        entities.append((start, end, ent_label))

                        if ent_label not in stats["entity_counts"]:
                            stats["entity_counts"][ent_label] = 0
                        stats["entity_counts"][ent_label] += 1

                        if ent_label not in existing_labels:
                            stats["new_labels"].add(ent_label)

                # Test alignment
                doc = nlp.make_doc(text)
                _ = Example.from_dict(doc, {"entities": entities})
                stats["valid_examples"] += 1

            except Exception as e:
                stats["invalid_examples"] += 1
                stats["errors"].append({
                    "index": item["id"],
                    "text": item["text"][:50],
                    "error": str(e)
                })

        # Print report
        print(f"\n📊 Validation Report:")
        print(f"   Total examples: {stats['total_examples']}")
        print(f"   ✅ Valid: {stats['valid_examples']}")
        print(f"   ❌ Invalid: {stats['invalid_examples']}")

        if stats["entity_counts"]:
            print(f"\n   Entity distribution:")
            for label, count in stats["entity_counts"].items():
                print(f"      {label}: {count}")

        if stats["new_labels"]:
            print(f"\n   🆕 New labels to add: {stats['new_labels']}")

        if stats["errors"]:
            print(f"\n   ⚠️  First few errors:")
            for error in stats["errors"]:
                print(f"      Example {error['index']}: {error['error']}")

        return stats

    def rollback(self):
        """Restore model from backup if training failed."""
        if not self.backup_path:
            print("⚠️  No backup found to rollback to")
            return False

        print(f"\n🔄 Rolling back to backup: {self.backup_path}")

        # Remove failed model
        if self.temp_model_path.exists():
            shutil.rmtree(self.temp_model_path)

        if self.model_path.exists():
            shutil.rmtree(self.model_path)

        # Restore backup
        shutil.copytree(self.backup_path, self.model_path)
        print("✅ Rollback complete")

        return True

    def train_model(self, n_iter: int = 10):
        """Train the model incrementally."""
        print(f"\n🎓 Starting incremental training ({n_iter} iterations)...")

        # Load training data
        with open(self.training_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load model
        nlp = spacy.load(str(self.model_path))
        ner = nlp.get_pipe("ner")

        # Add new labels if needed
        for item in data:
            for label in item.get("label", []):
                for ent_label in label["labels"]:
                    if ent_label not in ner.labels:
                        ner.add_label(ent_label)
                        print(f"   Added new label: {ent_label}")

        # Prepare training examples
        import random
        from spacy.training.example import Example

        train_data = []
        for item in data:
            text = item["text"]
            entities = []
            for label in item.get("label", []):
                for ent_label in label["labels"]:
                    entities.append((label["start"], label["end"], ent_label))
            train_data.append((text, {"entities": entities}))

        # Train
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.resume_training()

            for itn in range(n_iter):
                random.shuffle(train_data)
                losses = {}

                for text, annotations in train_data:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.3, losses=losses, sgd=optimizer)

                print(f"   Epoch {itn + 1}/{n_iter} - Loss: {losses.get('ner', 0):.3f}")

        # Save to temp location
        print(f"\n💾 Saving trained model to: {self.temp_model_path}")
        nlp.to_disk(self.temp_model_path)
        print("✅ Model saved")

    def run(self, n_iter: int = 10, min_improvement: float = 0.0) -> bool:
        """Run complete training workflow with safety checks."""
        print("\n" + "=" * 60)
        print("SAFE INCREMENTAL TRAINING WORKFLOW")
        print("=" * 60)

        try:
            # Step 1: Backup
            self.backup_model()

            # Step 2: Validate
            validation = self.validate_training_data()

            if validation["invalid_examples"] > validation["valid_examples"] * 0.2:
                print("\n❌ Too many invalid examples (>20%). Please fix data first.")
                return False

            # Step 3: Train
            self.train_model(n_iter)

            # Step 4: Replace old model with new one
            print(f"\n✅ Training successful! Replacing old model...")
            shutil.rmtree(self.model_path)
            shutil.move(str(self.temp_model_path), str(self.model_path))

            print(f"\n{'=' * 60}")
            print("✅ WORKFLOW COMPLETE!")
            print(f"{'=' * 60}")
            print(f"Updated model: {self.model_path}")
            print(f"Backup saved at: {self.backup_path}")

            return True

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print("\nAttempting rollback...")
            self.rollback()
            return False


# ============= CLI =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Safe incremental training workflow"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Absolute path to existing model"
    )
    parser.add_argument(
        "--data",
        default="data/labels_normalized.json",
        help="Path to training data (relative to model_training folder)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Training iterations"
    )

    args = parser.parse_args()

    # Resolve data path relative to model_training directory
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    if not Path(args.data).is_absolute():
        data_path = base_dir / args.data
    else:
        data_path = args.data

    workflow = ModelTrainingWorkflow(
        model_path=args.model,
        training_data_path=str(data_path)
    )

    success = workflow.run(n_iter=args.iterations)

    exit(0 if success else 1)