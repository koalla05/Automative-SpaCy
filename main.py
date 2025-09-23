import json
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import unicodedata

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('\u2011', '-')  # non-breaking hyphen
    text = text.replace('\u00A0', ' ')  # non-breaking space
    text = text.replace('\\/', '/')    # escaped slash
    return text

def remove_duplicate_entities(entities):
    seen = set()
    unique = []
    for start, end, label in entities:
        key = (start, end, label)
        if key not in seen:
            seen.add(key)
            unique.append((start, end, label))
    return unique

# --- Load normalized data ---
with open("labels_normalized.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# --- Prepare training data ---
TRAIN_DATA = []
skipped_count = 0

for item in raw_data:
    text = normalize_text(item["text"])
    entities = []

    for label in item.get("label", []):
        start = label["start"]
        end = label["end"]
        ent_label = label["labels"]
        for l in ent_label:
            entities.append((start, end, l))

    entities = remove_duplicate_entities(entities)
    doc = None
    try:
        # Check alignment before adding
        nlp_tmp = spacy.blank("xx")
        doc = nlp_tmp.make_doc(text)
        _ = Example.from_dict(doc, {"entities": entities})
        TRAIN_DATA.append((text, {"entities": entities}))
    except Exception as e:
        skipped_count += 1
        print(f"[!] Skipped due to error: {e}\n--> Text: {text}\n--> Entities: {entities}\n")

print(f"\n✅ Prepared {len(TRAIN_DATA)} training examples")
print(f"⚠️  Skipped {skipped_count} bad examples\n")

# --- Initialize blank model ---
nlp = spacy.blank("xx")  # multilingual

# --- Add NER pipeline ---
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# --- Add labels ---
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# --- Train the model ---
optimizer = nlp.begin_training()
n_iter = 20

for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.5))
    for batch in batches:
        examples = []
        for text, annots in batch:
            doc = nlp.make_doc(text)
            annots["entities"] = remove_duplicate_entities(annots["entities"])
            try:
                example = Example.from_dict(doc, annots)
                examples.append(example)
            except ValueError as e:
                print(f"[!] Skipped example during training: {e}")
        if examples:
            nlp.update(examples, drop=0.3, losses=losses)
    print(f"Epoch {itn+1}/{n_iter} - Losses: {losses}")

# --- Save the model ---
nlp.to_disk("full_ner_model")
print("\n✅ Model saved to: full_ner_model")
