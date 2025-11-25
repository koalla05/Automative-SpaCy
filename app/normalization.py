import re

equipment_normalization = {
    "inverter": [
        # Ukrainian
        "інвертор", "інвертора", "інвертори", "інверторів", "інвертору", "інверторі", "інвертором", "інверт",
        # гівно
        "инвертор", "инвертора", "инверторы", "инверторов", "инвертору", "инверторе", "инвертором", "инверт",
        # English
        "inverter", "inverters", "invert", "inverter/charger"
    ],
    "battery": [
        "акумулятор", "акумулятора", "акумулятори", "акумуляторів", "акумулятору", "акумуляторі", "акумулятором", "акум", "акумулятора?",
        "аккумулятор", "аккумулятора", "аккумуляторы", "аккумуляторов", "аккумулятору", "аккумуляторе", "аккумулятором",
        "energy storage battery", "battery", "batteries"
    ],
    "communication_hub": [
        "комунікаційний хаб", "комунікаційного хаба", "комунікаційні хаби", "комунікаційних хабів", "хаб",
        "коммуникационный хаб", "коммуникационного хаба", "коммуникационные хабы", "коммуникационных хабов",
        "communication hub", "communication hubs", "hub"
    ],
    "energy_storage": [
        "система зберігання енергії", "системи зберігання енергії", "системи енергозбереження", "системою енергозбереження", "системами енергозбереження",
        "система хранения энергии", "системы хранения энергии", "система энергосбережения",
        "energy storage", "energy storage solution"
    ],
    "data_logger": [
        "даталогер", "даталогера", "даталогери", "даталогерів",
        "даталоггер", "даталоггера", "даталоггеры", "даталоггеров",
        "data logger", "data loggers"
    ],
    "communication_box": [
        "комунікаційна коробка", "комунікаційної коробки", "комунікаційні коробки", "комунікаційних коробок",
        "коммуникационная коробка", "коммуникационной коробки", "коммуникационные коробки", "коммуникационных коробок",
        "communication box", "communication boxes"
    ],
    "monitoring_system": [
        "система моніторингу", "системи моніторингу", "системою моніторингу", "системами моніторингу",
        "система мониторинга", "системы мониторинга",
        "monitoring system", "monitoring systems"
    ],
    "energy_meter": [
        "electricity meter", "енергомір", "енергоміра", "енергоміри", "енергомірів",
        "энергомер", "энергомера", "энергомеры", "энергомеров",
        "energy meter", "energy meters", "electricity meter", "electricity meters"
    ],
    "ev_charger": [
        "зарядний пристрій для електромобіля", "зарядні пристрої для електромобілів",
        "зарядное устройство для электромобиля", "зарядные устройства для электромобилей",
        "electric vehicle charger", "EV charger", "EV chargers"
    ],
    "anti_reflex_box": [
        "анти-рефлюкс коробка", "анти-рефлюкс коробки",
        "анти-рефлюкс коробка", "анти-рефлюкс коробки",
        "anti-reflux box", "anti-reflux boxes"
    ],
    "solar_energy_system": [
        "сонячна енергетична система", "сонячні енергетичні системи", "сонячна система",
        "солнечная энергетическая система", "солнечные энергетические системы", "солнечная система",
        "solar energy system", "solar system", "photovoltaic system"
    ],
    "export_power_manager": [
        "менеджер експорту потужності", "менеджери експорту потужності",
        "менеджер экспорта мощности", "менеджеры экспорта мощности",
        "export power manager", "export power managers"
    ],
    "wlan_dongle": [
        "WLAN Dongle", "wlan dongle", "WLAN dongle",
        "WLAN донгл", "донгл WLAN", "донгл вайфай",
        "WLAN донгл", "донгл WLAN", "донгл вайфай"
    ]
}

equipment_normalization.update({
    "communication_box": [
        "communication box", "Smart Communication Box", "smart communication box", "smart communication boxes",
        "smart com box", "smart com boxes",
        "розумний комунікаційний блок", "смарт комунікаційний блок", "смарт бокс зв'язку",
        "розумні комунікаційні блоки", "смарт блоки зв'язку",
        "умный коммуникационный блок", "смарт коммуникационный блок", "смарт бокс связи",
        "умные коммуникационные блоки", "смарт блоки связи"
    ],
    "energy_meter": [
        "energy meter", "Smart Energy Meter", "smart meter", "Energy Meter", "smart energy meter", "smart energy meters",
        "енергомір", "розумний лічильник", "розумні лічильники", "смарт лічильник", "смарт лічильники",
        "энергомер", "умный счётчик", "умные счётчики", "смарт счётчик", "смарт счётчики"
    ],
    "electric_vehicle_charger": [
        "electric vehicle charger", "electric vehicle chargers", "ev charger", "ev chargers",
        "зарядка для електромобіля", "зарядний пристрій для електромобіля",
        "зарядки для електромобілів", "зарядні пристрої для електромобілів",
        "зарядка для электромобиля", "зарядное устройство для электромобиля",
        "зарядки для электромобилей", "зарядные устройства для электромобилей"
    ],
    "energy_storage": [
        "energy storage solution", "energy storage solutions",
        "система зберігання енергії", "системи зберігання енергії",
        "система хранения энергии", "системы хранения энергии"
    ],
    "energy_storage_cabinet": [
        "energy storage cabinet", "energy storage cabinets",
        "шафа зберігання енергії", "шафи зберігання енергії",
        "шкаф хранения энергии", "шкафы хранения энергии"
    ],
    "compatibility_battery_list": [
        "Compatible Lithium Battery List.md", "Compatibility list", "Compatibility battery list", "compatibility battery list", "battery compatibility list",
        "список сумісних батарей", "список сумісності акумуляторів",
        "список совместимых батарей", "список совместимости аккумуляторов"
    ]
})


manufacturer_normalization = {
    "acrel": ["acrel", "Acrel", "Акрел", "Acrel Co., Ltd.", "Acrel Electric", "Acrel Electric Co., Ltd."],
    "deye": ["Ningbo Deye Inverter Technology Co.", "ningbo deye inverter technology co.", "Deye", "deye", "DEYE", "ДЕЙ", "дейе", "Deye Inverter Technology Co., Ltd.", "Deye Inverter Technology", "Deye Inverter Tech Co., Ltd.", "Deye Inverter Tech", "Deye Inverter Tech Co Ltd", "Deye Inverter Tech Co.,Ltd.", "Ningbo Deye Inverter Technology Co.,Ltd", "Ningbo Deye Inverter Technology Co., Ltd.", "Ningbo Deye Inverter Tech Co., Ltd", "Deye Inverter", "Deye Inverter Co.,Ltd"],
    "huawei": ["Huawei Technologies Co.", "huawei technologies co.", "huawei", "Huawei", "Хуавей", "Хуавеї", "Хуавеєм", "Хуавею", "Хуавеєм", "Хуавеї", "Huawei Technologies Co., Ltd.", "Huawei Tech", "Huawei Technologies", "Huawei Tech Co., Ltd.", "Huawei Technologies Co Ltd", "Huawei Tech Co Ltd", "Huawei Tech Co.,Ltd.", "Huawei Technologies", "Huawei Tech", "Huawei Tech Co., Ltd.", "Huawei Technologies Co Ltd", "Huawei"],
    "dyness": ["dyness", "Dyness", "Дайнес", "дайнес", "Dyness Energy Co., Ltd.", "Dyness Energy", "Dyness Energy Co., Ltd.", "Dyness Energy Co Ltd", "Dyness Energy Co.,Ltd.", "Dyness Co., Ltd.", "Dyness Co Ltd", "Dyness Co.,Ltd.", "Daqin New Energy Tech (Taizhou) Co., Ltd.", "DAQIN NEW ENERGY TECHNOLOGY (TAIZHOU) CO., LTD.", "Daqin New Energy Tech (Taizhou) Co.", "DAQIN NEW ENERGY TECHNOLOGY (TAIZHOU) CO.", "daqin new energy tech (taizhou) co.", "daqin new energy technology (taizhou) co.", "daqin new energy tech", "daqin", "daqin energy", "Daqin New Energy Tech (Taizhou) Co., Ltd.", "Daqin New Energy Tech", "Daqin Energy", "Daqin New Energy Tech Co., Ltd.", "Daqin New Energy Tech Co Ltd", "Daqin New Energy Tech Co.,Ltd.", "Daqin Energy Co., Ltd.", "Daqin Energy Co Ltd", "Daqin Energy Co.,Ltd."],
    "atmosfera": ["атмосфера", "Атмосфера", "Atmosfera", "Atmosphere"],
    "solis": ["Ginlong Technologies Co., Ltd.", "Ginlong Technologies Co.", "ginlong technologies co.", "solis", "Solis", "Соліс", "соліс", "Solis Power", "Solis Power Co., Ltd.", "Solis Power Co Ltd", "Solis Power Co.,Ltd.", "Solis", "Solis Power", "Ginlong Solis", "Ningbo Ginlong Technologies Co., Ltd.", "Ningbo Ginlong Technologies Co., Ltd.", "Ningbo Ginlong Technologies Co., Ltd", "Ningbo Deye Inverter Technology Co.,Ltd", "Ginlong (Ningbo) Technologies Co., Ltd.", "Ginlong Technologies Co., Ltd.", "Ginlong (Ningbo) Technologies Co., Ltd.", "ningbo ginlong", "Ginlong", "Гінлонг", "Ningbo Ginlong Technologies Co., Ltd."],
    "fronius": ["fronius", "Fronius", "Фроніус", "фроніус", "Fronius International GmbH", "Fronius GmbH", "Fronius Co., Ltd.", "Fronius Co Ltd", "Fronius Co.,Ltd."],
    "luxpower": ["Lux Power Technology Co., Ltd", "Lux Power Technology Co., LTD", "LUX POWER TECHNOLOGY CO., LTD", "Lux Power Technology Co., LTD", "Shenzhen Lux Power Technology Co.", "Lux Powertek", "Luxpower", "Lux Power Technology Co.", "Lux Power Technology", "Lux Power Tech", "Lux Power", "Lux", "LUX POWER TECHNOLOGY CO.", "lux", "люкс", "lux power", "lux power tech", "lux power technology", "lux power technology co.", "lux powertek", "luxpower", "LuxPower", "ЛюксПауер", "LuxPowe,r", "ЛюксПауер", "люкспавер", "Люкс Павер", "Люкс Пауер", "люкс павер", "Shenzhen LuxPower", "Shenzhen LuxPower Technology Co., Ltd.", "LuxPower Technology Co.,Ltd.", "Luxpower Tech", "Luxpower Technology", "ЛюксПавер"],  # note typo in original list
    "victron": ["victronenergy", "victron energy", "victron", "Victron", "Віктрон", "віктрон", "Victron Energy B.V.", "Victron Energy", "Victron Energy B.V.", "Victron Energy BV", "Victron Energy BVBA", "Victron Energy B.V. Co.,Ltd.", "виктрон енерджі", "Віктрон Енерджі", "віктрона", "Віктрона", "виктрона"],
    "sofar": [ "Shenzhen Shouhang New Energy Co., LTD", "shenzhen sofarsolar co.", "sofarsolar", "sofar", "Sofar", "Софар", "софар", "Sofar Solar Co., Ltd.", "Sofar Solar", "Sofar Solar Co.,Ltd.", "Sofar Co., Ltd.", "Sofar Co Ltd", "Sofar Co.,Ltd.", "Sofar Electric Co., Ltd.", "Sofar Electric", "Sofar Electric Co.,Ltd.", "Sofar Electric Co Ltd", "Shenzhen SOFARSOLAR Co., Ltd.", "Shenzhen SOFARSOLAR CO., Ltd.", "Shenzhen Sofarsolar", "SOFARSOLAR", "Sofarsolar", "SOFAR SOLAR", "SOFAR SOLAR Co., Ltd.", "SOFAR SOLAR CO., Ltd."],
    "pylontech": ["Pylontech", "PylonTech", "Pylon Techology", "Pylon Technology", "Pylon Technologies Co.", "pylon technologies co.", "pylon techology", "pylon technology", "pylontech", "PylonTech", "ПайлонТек", "pylontek", "Pylontek", "Пайлонтек", "Pylon Tech", "Pylon Tech Co., Ltd.", "Pylon Tech Co Ltd", "Pylon Tech Co.,Ltd.", "ПілонТек", "Пілонтек", "пилонтек", "пілон тек", "Пілон Тек", "Pylon Technology", "Pylon Techology", "Pylon Technologies", "Pylon Technologies Co., Ltd", "Pylon Technologies Co.,Ltd"],
    "sungrow": ["Sungrow Power Supply Co., Ltd", "SUNGROW POWER SUPPLY CO.", "sungrow power supply co.", "sungrow", "Sungrow", "Сангров", "sangrov", "Sangrov", "Сангров", "Sungrow Power Supply Co., Ltd.", "Sungrow Power Supply", "Sungrow Power Supply Co.,Ltd.", "Sungrow Co., Ltd.", "Sungrow Co Ltd", "Sungrow Co.,Ltd.", "сангроу", "Сангроу", "сангров", "SUNGROW POWER SUPPLY CO., LTD.", "Sungrow Power Supply Co., Ltd.", "Sungrow Power Supply Co.,Ltd.", "Sungrow Power Supply Co Ltd", "SUNGROW", "Sungrow", "Сангроу"]
}


manufacturer_normalization.update({
    "deye": [
        "ningbo deye inverter technology co., ltd.",
        "deye inverter technology",
        "deye", "деє", "дее", "дее інвертор", "деє інвертор",
        "нинбо деє інвертер", "нинбо деє інвертор технолоджі",
        "нингбо дее инвертор", "нингбо дее инвертор технолоджи"
    ],
    "huawei": [
        "huawei technologies co., ltd.",
        "huawei technologies", "huawei", "хуавей", "хуавэй"
    ],
    "shenzhen": [
        "Shenzhen Lux Power Technology Co., Ltd", "shenzhen sofarsolar co., ltd.", "sofarsolar", "sofar solar",
        "шеньчжень софарсолар", "софарсолар", "софар солар"
    ],
    "sungrow": [
        "sungrow power supply co., ltd.", "sungrow power supply", "sungrow",
        "сангроу", "сангров", "сангроу пауер"
    ],
    "pylontech": [
        "Pylontech", "PylonTech", "Pylon Techology", "Pylon Technology", "Pylon Technologies Co.", "pylon technology", "pylon technologies co., ltd.",
        "пайлон технолоджі", "пайлон", "пілон", "пайлон технолоджіс",
        "пайлон технологии", "пайлон технолоджи"
    ],
    "pylontech": [
        "pylontech", "pylon tech", "pylon", "пайлонтек", "пайлон тек", "пілонтек"
    ],
    "shenzhen": [
        "Shenzhen Shouhang New Energy Co., Ltd.",
        "Shenzhen Shouhang New Energy Co., LTD",
        "Shenzhen Shouhang New Energy",
        "Shenzhen Shouhang New Energy Co Ltd",
        "Shenzhen Shouhang New Energy",
        "Шэньчжэнь Шоуханг Новая Энергия",
        "Шэньчжэнь Шоуxанг Новая Энергия"
    ]
})


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
