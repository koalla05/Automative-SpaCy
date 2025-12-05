import spacy

# Load your trained model
nlp = spacy.load("../models/full_ner_model")

# Example new queries
new_queries = [
    "Чи можна підключити Fronius SYMO 10.0-3-M до акумулятора?",
    "Скільки MPPT в LuxPower LXP-8K?",
    "Який інвертор сумісний із акумулятором Victron?",
    "Чи є у інвертора Solar KKJGSFH-4235 вхід ЮСБ",
    "Чи є мережевий інвертор Solis на 700 Вт?,"
    "Які контролери Victron ви використовуєте?",
    "Хто в Києві з дилерів цікавиться Fronius?",
    "Яка гарантія на інвертори Solis-30K?",
    "Які параметри у акамулятора AbiSolar 315?",
    "Чи можна підключити PowerCube-H2 від PylonTech до чайника?",
    "Хто в Києві з дилерів цікавиться фроніус?",
    "Як підключити інвертор Віктрон Quattro 3kVA (12-48)V до мережі?",
    "Хто в Києві з дилерів цікавиться Фроніус і Хуавей?",
    "Хто в Києві з дилерів цікавиться інвертором Фроніус, акамулятором Віктрон і Хуавеєм?"
]

text = ["Чи доступний даташіт на сонячний інвертор SUN 5K-SG-EU від DEYE?,"
"Які параметри у акумулятора AbiSolar 315?,"]

# Run NER
for query in text:
    doc = nlp(query)
    print(f"\nQuery: {query}")
    for ent in doc.ents:
        print(f"{ent.text} → {ent.label_}")
