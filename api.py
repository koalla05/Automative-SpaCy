from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re
from typing import List, Dict, Union

# Load your trained spaCy model
nlp = spacy.load("full_ner_model")

app = FastAPI()

# ========== Normalization Data ==========
equipment_normalization = {
    "INVERTER": [
        # Ukrainian
        "інвертор", "інвертора", "інвертори", "інверторів", "інвертору", "інверторі", "інвертором", "інверт",
        # Russian
        "инвертор", "инвертора", "инверторы", "инверторов", "инвертору", "инверторе", "инвертором", "инверт",
        # English
        "inverter", "inverters", "invert"
    ],
    "BATTERY": [
        # Ukrainian
        "акумулятор", "акумулятора", "акумулятори", "акумуляторів", "акумулятору", "акумуляторі", "акумулятором", "акум", "акумулятора?",
        # Russian
        "аккумулятор", "аккумулятора", "аккумуляторы", "аккумуляторов", "аккумулятору", "аккумуляторе", "аккумулятором",
        # English
        "battery", "batteries"
    ],
    "COMMUNICATION_HUB": [
        # Ukrainian
        "комунікаційний хаб", "комунікаційного хаба", "комунікаційні хаби", "комунікаційних хабів", "хаб",
        # Russian
        "коммуникационный хаб", "коммуникационного хаба", "коммуникационные хабы", "коммуникационных хабов",
        # English
        "communication hub", "communication hubs", "hub"
    ],
    "ENERGY_STORAGE": [
        # Ukrainian
        "система зберігання енергії", "системи зберігання енергії", "системи енергозбереження", "системою енергозбереження", "системами енергозбереження",
        # Russian
        "система хранения энергии", "системы хранения энергии", "система энергосбережения",
        # English
        "energy storage", "energy storage solution"
    ],
    "DATA_LOGGER": [
        # Ukrainian
        "даталогер", "даталогера", "даталогери", "даталогерів",
        # Russian
        "даталоггер", "даталоггера", "даталоггеры", "даталоггеров",
        # English
        "data logger", "data loggers"
    ],
    "COMMUNICATION_BOX": [
        # Ukrainian
        "комунікаційна коробка", "комунікаційної коробки", "комунікаційні коробки", "комунікаційних коробок",
        # Russian
        "коммуникационная коробка", "коммуникационной коробки", "коммуникационные коробки", "коммуникационных коробок",
        # English
        "communication box", "communication boxes"
    ],
    "MONITORING_SYSTEM": [
        # Ukrainian
        "система моніторингу", "системи моніторингу", "системою моніторингу", "системами моніторингу",
        # Russian
        "система мониторинга", "системы мониторинга",
        # English
        "monitoring system", "monitoring systems"
    ],
    "ENERGY_METER": [
        # Ukrainian
        "енергомір", "енергоміра", "енергоміри", "енергомірів",
        # Russian
        "энергомер", "энергомера", "энергомеры", "энергомеров",
        # English
        "energy meter", "energy meters", "electricity meter", "electricity meters"
    ],
    "EV_CHARGER": [
        # Ukrainian
        "зарядний пристрій для електромобіля", "зарядні пристрої для електромобілів",
        # Russian
        "зарядное устройство для электромобиля", "зарядные устройства для электромобилей",
        # English
        "electric vehicle charger", "EV charger", "EV chargers"
    ],
    "ANTI_REFLEX_BOX": [
        # Ukrainian
        "анти-рефлюкс коробка", "анти-рефлюкс коробки",
        # Russian
        "анти-рефлюкс коробка", "анти-рефлюкс коробки",
        # English
        "anti-reflux box", "anti-reflux boxes"
    ],
    "SOLAR_ENERGY_SYSTEM": [
        # Ukrainian
        "сонячна енергетична система", "сонячні енергетичні системи", "сонячна система",
        # Russian
        "солнечная энергетическая система", "солнечные энергетические системы", "солнечная система",
        # English
        "solar energy system", "solar system", "photovoltaic system"
    ],
    "EXPORT_POWER_MANAGER": [
        # Ukrainian
        "менеджер експорту потужності", "менеджери експорту потужності",
        # Russian
        "менеджер экспорта мощности", "менеджеры экспорта мощности",
        # English
        "export power manager", "export power managers"
    ]
}

manufacturer_normalization = {
    "DEYE": ["deye", "DEYE", "ДЕЙ", "дейе", "Deye Inverter Technology Co., Ltd.", "Deye Inverter Technology", "Deye Inverter Tech Co., Ltd.", "Deye Inverter Tech", "Deye Inverter Tech Co Ltd", "Deye Inverter Tech Co.,Ltd."],
    "HUAWEI": ["huawei", "Huawei", "Хуавей", "Хуавеї", "Хуавеєм", "Хуавею", "Хуавеєм", "Хуавеї", "Huawei Technologies Co., Ltd.", "Huawei Tech", "Huawei Technologies", "Huawei Tech Co., Ltd.", "Huawei Technologies Co Ltd", "Huawei Tech Co Ltd", "Huawei Tech Co.,Ltd."],
    "DAQIN": ["daqin new energy tech", "daqin", "daqin energy", "Daqin New Energy Tech (Taizhou) Co., Ltd.", "Daqin New Energy Tech", "Daqin Energy", "Daqin New Energy Tech Co., Ltd.", "Daqin New Energy Tech Co Ltd", "Daqin New Energy Tech Co.,Ltd.", "Daqin Energy Co., Ltd.", "Daqin Energy Co Ltd", "Daqin Energy Co.,Ltd."],
    "DYNES": ["dyness", "Dyness", "Дайнес", "дайнес", "Dyness Energy Co., Ltd.", "Dyness Energy", "Dyness Energy Co., Ltd.", "Dyness Energy Co Ltd", "Dyness Energy Co.,Ltd.", "Dyness Co., Ltd.", "Dyness Co Ltd", "Dyness Co.,Ltd."],
    "ATMOSFERA": ["атмосфера", "Атмосфера", "Atmosfera", "Atmosphere"],
    "NINGBO_GINLONG": ["ningbo ginlong", "Ginlong", "Гінлонг", "Ningbo Ginlong Technologies Co., Ltd."],
    "SOLIS": ["solis", "Solis", "Соліс", "соліс", "Solis Power", "Solis Power Co., Ltd.", "Solis Power Co Ltd", "Solis Power Co.,Ltd."],
    "FRONIUS": ["fronius", "Fronius", "Фроніус", "фроніус", "Fronius International GmbH", "Fronius GmbH", "Fronius Co., Ltd.", "Fronius Co Ltd", "Fronius Co.,Ltd."],
    "LUXPOWER": ["luxpower", "LuxPower", "ЛюксПауер", "LuxPowe,r", "ЛюксПауер", "люкспавер", "Люкс Павер", "Люкс Пауер", "люкс павер"],  # note typo in original list
    "VICTRON": ["victron", "Victron", "Віктрон", "віктрон", "Victron Energy B.V.", "Victron Energy", "Victron Energy B.V.", "Victron Energy BV", "Victron Energy BVBA", "Victron Energy B.V. Co.,Ltd.", "виктрон енерджі", "Віктрон Енерджі", "віктрона", "Віктрона", "виктрона"],
    "SOFAR": ["sofar", "Sofar", "Софар", "софар", "Sofar Solar Co., Ltd.", "Sofar Solar", "Sofar Solar Co.,Ltd.", "Sofar Co., Ltd.", "Sofar Co Ltd", "Sofar Co.,Ltd.", "Sofar Electric Co., Ltd.", "Sofar Electric", "Sofar Electric Co.,Ltd.", "Sofar Electric Co Ltd"],
    "PYLONTECH": ["pylontech", "PylonTech", "ПайлонТек", "pylontek", "Pylontek", "Пайлонтек", "Pylon Tech", "Pylon Tech Co., Ltd.", "Pylon Tech Co Ltd", "Pylon Tech Co.,Ltd.", "ПілонТек", "Пілонтек", "пилонтек", "пілон тек", "Пілон Тек"],
    "SUNGROW": ["sungrow", "Sungrow", "Сангров", "sangrov", "Sangrov", "Сангров", "Sungrow Power Supply Co., Ltd.", "Sungrow Power Supply", "Sungrow Power Supply Co.,Ltd.", "Sungrow Co., Ltd.", "Sungrow Co Ltd", "Sungrow Co.,Ltd.", "сангроу", "Сангроу", "сангров"]
}

# ========== Clean and Normalize ==========

def clean_word(word: str) -> str:
    return re.sub(r"^[^\w\d\-\.]+|[^\w\d\-\.]+$", "", word.strip())

def normalize_entity(word: str, label: str) -> str:
    word_lower = word.lower()
    if label == "EQ_TYPE":
        for canon, variants in equipment_normalization.items():
            if any(word_lower == v.lower() for v in variants):
                return canon
    elif label == "MANUFACTURER":
        for canon, variants in manufacturer_normalization.items():
            if any(word_lower == v.lower() for v in variants):
                return canon
    return word  # fallback

# ========== Request Model ==========
class Query(BaseModel):
    text: str

# ========== Endpoint ==========
@app.post("/extract_entities/spacy", response_model=Dict[str, List[str]])
def extract_entities(query: Query):
    doc = nlp(query.text)
    grouped: Dict[str, List[str]] = {}

    for ent in doc.ents:
        label = ent.label_
        cleaned = clean_word(ent.text)
        normalized = normalize_entity(cleaned, label)

        if label not in grouped:
            grouped[label] = []

        if normalized not in grouped[label]:
            grouped[label].append(normalized)

    return grouped
