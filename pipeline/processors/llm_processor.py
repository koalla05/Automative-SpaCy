# llm_processor.py

import re
import logging
from typing import Dict, Any

logger = logging.getLogger("ipg_pipeline")


def detect_lifestyle_query(text: str) -> bool:
    text = text.lower().strip()

    lifestyle_patterns = [
        # === GREETINGS (UA + гівно + EN) ===
        r'\bпривіт(ик|ки)?\b',
        r'\bдобр(ий|ого)\s+д(ень|ня)\b',
        r'\bдобр(ий|ого)\s+веч(ір|ора)\b',
        r'\bдобр(ий|ого)\s+ран(ок|ку)\b',
        r'\bвітаю( вас)?\b',
        r'\bздоров(енькі)?\b',
        r'\bал(ло|ьо)\b',
        r'\bна\s+зв[ʼ’`]?язку\b',
        r'\bє\s+хто\b',

        r'\bпривет\b',
        r'\bдобр(ый|ого)\s+(день|вечер|утро)\b',
        r'\bздравств(уй|уйте)\b',
        r'\bна\s+связи\b',
        r'\bесть\s+кто\b',

        r'\bhi\b', r'\bhello\b', r'\bhey\b', r'\bgood\s+(morning|evening|afternoon)\b',

        # === FAREWELLS ===
        r'\bбувай(те)?\b',
        r'\bдо\s+побачення\b',
        r'\bна\s+все\s+добре\b',
        r'\bгарн(ого|ий)\s+(дня|вечора)\b',
        r'\bдо\s+зв[ʼ’`]?язку\b',
        r'\bпочуємось\b',

        r'\bпока\b',
        r'\bдо\s+свидания\b',
        r'\bвсего\s+доброго\b',

        r'\bbye\b', r'\bgoodbye\b', r'\bsee\s+you\b',

        # === GRATITUDE ===
        r'\bдякую\b',
        r'\bщиро\s+дякую\b',
        r'\bвдячн(ий|а)\b',
        r'\bспасибі\b',
        r'\bдякс\b',

        r'\bспасибо\b',
        r'\bблагодарю\b',

        r'\bthanks\b', r'\bthank\s+you\b', r'\bthx\b',

        # === META / IDENTITY ===
        r'\b(ти|ви)\s+(хто|що)\b',
        r'\bхто\s+ти\b',
        r'\bти\s+бот\b',
        r'\bти\s+людин(а|и)\b',
        r'\bяк\s+тебе\s+звати\b',
        r'\bщо\s+ти\s+вмієш\b',
        r'\bяк\s+ти\s+працюєш\b',

        r'\bты\s+кто\b',
        r'\bкто\s+ты\b',
        r'\bты\s+бот\b',

        r'\bwho\s+are\s+you\b',
        r'\bare\s+you\s+a\s+bot\b',

        # === SMALL TALK ===
        r'\bяк\s+справи\b',
        r'\bяк\s+ти\b',
        r'\bщо\s+нового\b',
        r'\bяк\s+життя\b',
        r'\bяк\s+настрій\b',

        r'\bкак\s+дела\b',
        r'\bкак\s+ты\b',

        r'\bhow\s+are\s+you\b',
        r'\bwhat[’\']?s\s+up\b',

        # === SHORT REACTIONS ONLY ===
        r'^\s*(ок|окей|норм|нормально|топ|супер|клас|ok|okay)\s*$'
    ]

    # === EMOJI-ONLY ===
    emoji_only = re.fullmatch(r'[👍👌🙂😂✅❤️🔥\s]+', text)
    if emoji_only:
        return True

    combined_pattern = re.compile('|'.join(lifestyle_patterns), re.IGNORECASE)
    return bool(combined_pattern.search(text))


def detect_parallel_query(text: str) -> bool:
    text_lower = text.lower()

    parallel_patterns = [

        # =========================
        # Explicit parallel / stack
        # =========================

        # Ukrainian
        r'\bпаралел\w*\b',
        r'\bпаралельн\w*\b',
        r'\bстек\w*\b',

        # гівно
        r'\bпараллел\w*\b',
        r'\bпараллельн\w*\b',
        r'\bстек\w*\b',

        # English
        r'\bparallel\w*\b',
        r'\bstack\w*\b',
        r'\bstacking\b',

        # =========================
        # Action + 3-phase (KEY PART)
        # =========================

        # Ukrainian: дія + 3-фазна
        r'\b(чи\s+можна\s+)?'
        r'(зібрат|зробит|побудуват|реалізуват|створит|підключит|використат)\w*\b'
        r'.{0,30}'
        r'\b(3|три)[-\s]?(фаз|фазн)\w*\b',

        # Ukrainian: 3-фазна + дія
        r'\b(3|три)[-\s]?(фаз|фазн)\w*\b'
        r'.{0,30}'
        r'\b(підключат|зєднуват|збирати|використовуват)\w*\b',

        # гівно
        r'\b(можно\s+)?'
        r'(собрат|сделат|построит|реализоват|создат|подключит|использоват)\w*\b'
        r'.{0,30}'
        r'\b(3|три)[-\s]?фаз\w*\b',

        # English
        r'\b(can\s+i\s+)?'
        r'(build|make|connect|configure|create|use)\w*\b'
        r'.{0,30}'
        r'\b(3|three)[-\s]?phase\b',
    ]

    return any(re.search(p, text_lower) for p in parallel_patterns)


def detect_compatibility_query(text: str) -> bool:
    text_lower = text.lower()

    compat_patterns = [
        # Ukrainian
        r'\bсумісн\w*\b',
        r'\bчи можна (підключити|з\'єднати|використати)\b',
        r'\bв одну систему\b',
        r'\bчи працю\w* (з|разом)\b',

        # гівно
        r'\bсовмест\w*\b',
        r'\bможно ли (подключить|соединить|использовать)\b',
        r'\bв одну систему\b',
        r'\bработа\w* (с|вместе)\b',

        # English
        r'\bcompat\w*\b',
        r'\bcan (i|we) (connect|use|combine)\b',
        r'\bwork (with|together)\b',
        r'\bac[- ]?coupling\b',
    ]

    for pattern in compat_patterns:
        if re.search(pattern, text_lower):
            return True

    return False


def detect_error_code_query(text: str) -> bool:
    text_lower = text.lower()

    error_patterns = [
        # === Ukrainian ===
        r'\bкод\w*\s+помилк\w*\b',
        r'\bпомилк\w*\s+код\w*\b',
        r'\bпомилк\w*\b',
        r'\bпомилк\w*\s+на\s+екран\w*\b',
        r'\bщо\s+(означа\w*|значить)\s+(ця|цей|це|той|та)\s+(помилка|код|статус)\b',
        r'\bщо\s+робити\s+(якщо|коли|якшо)\b',
        r'\bчому\s+(блимає|моргає|горить|світиться|показує)\b',
        r'\b(з[`\'ʼ]явилась?|з[`\'ʼ]явився|виникл\w*)\s+(помилка|код|статус)\b',
        r'\b(інвертор|батарея|зарядний)\s+показу\w*\s+(помилку|код|статус)\b',
        r'\bяк(ий|а|е)?\s+статус\b',
        r'\bстатус\s+помилк\w*\b',
        r'\bE\d{2,4}\b',
        r'\bErr\w*\b',
        r'\bFault\b',

        # === Russian ===
        r'\bкод\w*\s+ошибк\w*\b',
        r'\bошибк\w*\s+код\w*\b',
        r'\bошибк\w*\b',
        r'\bошибк\w*\s+на\s+экран\w*\b',
        r'\bчто\s+(означа\w*|значит)\s+(эта|этот|это|тот|та)\s+(ошибка|код|статус)\b',
        r'\bчто\s+делать\s+(если|когда)\b',
        r'\bпочему\s+(мигает|горит|светится|показывает)\b',
        r'\b(появилась?|появился|возникл\w*)\s+(ошибка|код|статус)\b',
        r'\b(инвертор|батарея|зарядн\w*)\s+показыва\w*\s+(ошибку|код|статус)\b',
        r'\bкак(ой|ая|ое)?\s+статус\b',

        # === English ===
        r'\berror\s+code\w*\b',
        r'\bfault\s+code\w*\b',
        r'\bwhat\s+(does|is)\s+(error|fault|code|status)\b',
        r'\bwhat\s+to\s+do\s+(if|when)\b',
        r'\bwhy\s+(is\s+it\s+)?(blinking|flashing|showing|displaying)\b',
        r'\b(error|fault|warning|alarm)\s+(appeared|occurred|showing)\b',
        r'\bwhat\s+does\s+(this|the)\s+(status|code|error)\s+mean\b',

        # === Specific alphanumeric codes & well-known abbreviations (all languages) ===
        # These also match in detect_specific_error_code; including them here ensures
        # the KW classifier routes to error_code in the first place.
        r'\b[EeFfWwBb]\d{2,4}\b',          # E0049, F04, W12 …
        r'\bErr\w*\d+\b',                   # Err03, ERR_05 …
        r'\b[EeFfWwBb]-\d{2,4}\b',          # F-12 …
        r'\b[A-Z]{2,5}_\d{2,4}\b',          # OCP_01 …
        r'\bOVP\b', r'\bUVP\b', r'\bOCP\b', r'\bOTP\b',
        r'\bBMS\s+alarm\b', r'\bBMS\s+fault\b',
        r'\bSOC\s+low\b',
    ]

    return any(re.search(p, text_lower, re.IGNORECASE) for p in error_patterns)


def detect_pinout_query(text: str) -> bool:
    text_lower = text.lower()

    pinout_patterns = [
        # === Ukrainian ===
        r'\bрозпін\w*\b',
        r'\bпіноут\b',
        r'\bпінаут\b',
        r'\bяк\s+підключити\b',
        r'\bсхем\w*\s+підключен\w*\b',
        r'\bпідключен\w*\s+схем\w*\b',
        r'\bяк[і\w]*\s+дроти?\b',
        r'\b(який|яка|яке)\s+кабел\w*\b',
        r'\bяк\s+з[`\'ʼ]єднати\b',
        r'\bконектор\w*\b',
        r'\bпроводк\w*\b',
        r'\bщо\s+до\s+чого\s+(підключати|підключити)\b',
        r'\bяк\s+підпаяти\b',
        r'\bтемінал\w*\b',
        r'\bклем\w*\b',

        # === Russian ===
        r'\bраспин\w*\b',
        r'\bпиноут\b',
        r'\bкак\s+подключить\b',
        r'\bсхем\w*\s+подключен\w*\b',
        r'\bподключен\w*\s+схем\w*\b',
        r'\bкак\w*\s+провода?\b',
        r'\b(какой|какая|какое)\s+кабел\w*\b',
        r'\bкак\s+соединить\b',
        r'\bконнектор\w*\b',
        r'\bпроводк\w*\b',
        r'\bчто\s+к\s+чему\s+(подключать|подключить)\b',
        r'\bтерминал\w*\b',
        r'\bклемм\w*\b',

        # === English ===
        r'\bpinout\b',
        r'\bpin[-\s]?out\b',
        r'\bwiring\s+diagram\b',
        r'\bhow\s+to\s+(connect|wire|hook\s+up)\b',
        r'\bwiring\s+schema\w*\b',
        r'\bwhich\s+(cable|wire|connector|terminal)\b',
        r'\bconnect(ion)?\s+diagram\b',
        r'\bcable\s+diagram\b',
        r'\bterminal\w*\b',
        r'\bconnector\s+(pin\w*|layout|diagram)\b',
    ]

    return any(re.search(p, text_lower, re.IGNORECASE) for p in pinout_patterns)


def detect_documentation_query(text: str) -> bool:
    text_lower = text.lower()

    doc_patterns = [
        # === Ukrainian ===
        r'\bдокументац\w*\b',
        r'\bінструкц\w*\b',
        r'\bмануал\w*\b',
        r'\bпосібник\w*\b',
        r'\bкерівництво\b',
        r'\bдатащіт\w*\b',
        r'\bдата[-\s]?шіт\w*\b',
        r'\bдай\s+(документ|інструкц|мануал|посібник)\w*\b',
        r'\bзнайди\s+(документ|інструкц|мануал|посібник)\w*\b',
        r'\b(де\s+знайти|де\s+скачати)\s+(документ|інструкц|мануал)\w*\b',
        r'\bPDF\b',
        r'\bтехнічн\w*\s+документ\w*\b',
        r'\bспецифікац\w*\b',

        # === Russian ===
        r'\bдокументац\w*\b',
        r'\bинструкц\w*\b',
        r'\bмануал\w*\b',
        r'\bруководств\w*\b',
        r'\bдатащит\w*\b',
        r'\bдата[-\s]?шит\w*\b',
        r'\bдай\s+(документ|инструкц|мануал|руководств)\w*\b',
        r'\bнайди\s+(документ|инструкц|мануал|руководств)\w*\b',
        r'\b(где\s+найти|где\s+скачать)\s+(документ|инструкц|мануал)\w*\b',
        r'\bтехнич\w*\s+документ\w*\b',
        r'\bспецификац\w*\b',

        # === English ===
        r'\bdocumentation\b',
        r'\bdatasheet\b',
        r'\bdata[-\s]?sheet\b',
        r'\bmanual\b',
        r'\buser\s+guide\b',
        r'\binstallation\s+guide\b',
        r'\bspec\w*\s+sheet\b',
        r'\b(give|find|get|show|download)\s+(me\s+)?(the\s+)?(doc\w*|manual|datasheet)\b',
        r'\bwhere\s+(to\s+)?(find|download|get)\s+(doc\w*|manual|datasheet)\b',
        r'\btechnical\s+doc\w*\b',
        r'\bspecification\w*\b',
    ]

    return any(re.search(p, text_lower, re.IGNORECASE) for p in doc_patterns)


def detect_specific_error_code(text: str) -> bool:
    """
    Return True only when the text contains a *specific* fault/alarm/error code
    or recognised abbreviation — not just generic error language.

    Examples that match  : E0049, F04, OVP, BMS alarm, Err03, F-12, W001
    Examples that do NOT : "what does this error mean", "помилка на екрані"
    """
    specific_code_patterns = [
        # Alphanumeric codes: E0049, F04, W12, Err03, F-12, ERR_05
        r'\b[EeFfWwBb]\d{2,4}\b',
        r'\bErr\w*\d+\b',
        r'\b[EeFfWwBb]-\d{2,4}\b',
        r'\b[A-Z]{2,5}_\d{2,4}\b',

        # Well-known named fault abbreviations (add more as needed)
        r'\bOVP\b',       # Over-Voltage Protection
        r'\bUVP\b',       # Under-Voltage Protection
        r'\bOCP\b',       # Over-Current Protection
        r'\bOTP\b',       # Over-Temperature Protection
        r'\bBMS\s+alarm\b',
        r'\bBMS\s+fault\b',
        r'\bSOC\s+low\b',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in specific_code_patterns)


def needs_clarification(status: str, extracted_entities: Dict[str, Any], original_text: str) -> bool:
    """
    Determine whether the classified query needs clarification — i.e. it matched
    a keyword-based status but is missing information required to actually answer it.

    Rules (deterministic, per status):
      compat     → needs exactly 2 valid models; clarify if fewer than 2 are present
      error_code → needs a *specific* code/abbreviation in the text; clarify if only
                   generic error language was detected (no concrete code found)
      all others → clarify if no valid model is present at all

    Returns:
        True  → add  "clarification": true  to the result JSON
        False → add  "clarification": false
    """
    if status == "lifestyle":
        # Social exchanges never need clarification — no model/code is expected
        return False

    models = extracted_entities.get("model", [])
    valid_models = [m for m in models if m.get("value") is not None]
    num_valid_models = len(valid_models)

    if status == "compat":
        # Need both sides of the compatibility pair to be known
        return num_valid_models < 2

    if status == "error_code":
        # Need a concrete code, not just generic error vocabulary
        return not detect_specific_error_code(original_text)

    # For every other status: clarify when no model is present
    return num_valid_models == 0


def determine_status(extracted_entities: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Determine query status based on extracted entities and text analysis.

    STATUS logic (checked in order of priority):
    - "parallel": Parallel questions (detected by keywords)
    - "compat": Compatibility questions (detected by keywords)
    - "error_code": Error code / fault status questions (detected by keywords)
    - "pinout": Wiring / pinout / connection diagram questions (detected by keywords)
    - "documentation": Documentation / datasheet / manual requests (detected by keywords)
    - "simple": Direct parameter lookup from SQL/YAML
      * Has at least 1 VALID CANONICAL model (value != null)
      * Has parameters (datasheet specs)
      * Can be answered by SQL without interpretation/logic/combination
      * Allowed: 1-2 params + 1-2 models
      * Meaning: "Give me parameter value(s) for model(s)"
    - "complex": Everything else (calculations, >2 models, >2 params, no entities, invalid models, etc.)
    - "lifestyle": Social/meta queries (greetings, farewells, gratitude, small talk, reactions)
      * LOWEST PRIORITY - checked last

    Args:
        extracted_entities: Dict with manufacturer, model, equipment_type, parameters
        original_text: Original query text

    Returns:
        Dict with keys:
          "status"        — one of the status strings above
          "clarification" — bool; True when additional info is needed to answer the query
    """
    def _result(status: str) -> Dict[str, Any]:
        return {
            "status": status,
            "clarification": needs_clarification(status, extracted_entities, original_text),
        }

    if detect_parallel_query(original_text):
        logger.debug("Detected parallel query")
        return _result("parallel")

    if detect_compatibility_query(original_text):
        logger.debug("Detected compatibility query")
        return _result("compat")

    if detect_error_code_query(original_text):
        logger.debug("Detected error code query")
        return _result("error_code")

    if detect_pinout_query(original_text):
        logger.debug("Detected pinout query")
        return _result("pinout")

    if detect_documentation_query(original_text):
        logger.debug("Detected documentation query")
        return _result("documentation")

    models = extracted_entities.get("model", [])
    parameters = extracted_entities.get("parameters", [])

    valid_models = [m for m in models if m.get("value") is not None]

    num_valid_models = len(valid_models)
    num_params = len(parameters)

    logger.debug(f"Query entities: {num_valid_models} valid models, {num_params} parameters")

    if num_valid_models >= 1 and num_params >= 1:
        if num_valid_models <= 2 and num_params <= 2:
            logger.debug("Query classified as simple")
            return _result("simple")

    # Lifestyle is the lowest priority - only if NO technical entities at all
    if num_params > 0 or len(models) > 0:
        # Has parameters or models  = technical query
        logger.debug("Query classified as complex (has technical entities)")
        return _result("complex")

    if detect_lifestyle_query(original_text):
        logger.debug("Detected lifestyle query")
        return _result("lifestyle")

    logger.debug("Query classified as complex (default)")
    return _result("complex")


def build_param_bindings_logic(extracted_entities: Dict[str, Any]) -> list:
    """
    Build parameter bindings using direct logic.
    Maps parameters to the closest models by position in text.

    Only includes VALID canonical models (value != None).

    Args:
        extracted_entities: Dict with model and parameters lists

    Returns:
        List of bindings: [{"model": str, "parameters": [str]}]
    """
    models = extracted_entities.get("model", [])
    parameters = extracted_entities.get("parameters", [])

    # Filter out invalid models (where value is None/null)
    valid_models = [m for m in models if m.get("value") is not None]

    if not valid_models or not parameters:
        return []

    # Map each parameter to the closest valid model by position
    bindings_dict = {}  # model_value -> [param_keys]

    for param in parameters:
        param_pos = param.get("position", 0)

        # Find the closest valid model
        closest_model = min(
            valid_models,
            key=lambda m: abs(m.get("position", 0) - param_pos)
        )

        model_value = closest_model["value"]
        param_key = param["key"]

        if model_value not in bindings_dict:
            bindings_dict[model_value] = []

        if param_key not in bindings_dict[model_value]:
            bindings_dict[model_value].append(param_key)

    # Convert to list format
    param_bindings = []
    for model, params in bindings_dict.items():
        if not model: continue
        param_bindings.append({
            "model": model,
            "parameters": params
        })

    return param_bindings


def determine_intent_logic(status: str, extracted_entities: Dict[str, Any]) -> str:
    """
    Determine question intent based on status and entities.

    Args:
        status: "compat", "simple", or "complex"
        extracted_entities: Extracted entities

    Returns:
        Intent string
    """
    if status == "compat":
        return "compatibility_query"

    if status == "error_code":
        return "error_code_query"

    if status == "pinout":
        return "pinout_query"

    if status == "documentation":
        return "documentation_query"

    if status == "simple":
        return "sql_query"

    if status == "lifestyle":
        return "lifestyle_query"

    if extracted_entities.get("parameters"):
        return "multi_model_query"

    if extracted_entities.get("model"):
        return "uncertain"

    return "no_entities"


if __name__ == "__main__":
    from pprint import pprint

    # Test cases
    test_cases = [
        {
            "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.84, "position": 50}],
                "model": [{"value": "lxp_lb_eu_10k", "confidence": 0.88, "position": 60}],
                "equipment_type": [{"value": "inverter", "confidence": 0.88, "position": 45}],
                "parameters": [
                    {"key": "max_charge_current_a", "confidence": 0.95, "position": 10}
                ]
            },
            "expected": "simple"  # 1 model + 1 param
        },
        {
            "query": "Чи сумісний Pylontech US5000 з Victron MultiPlus?",
            "entities": {
                "manufacturer": [
                    {"value": "pylontech", "confidence": 0.9, "position": 15},
                    {"value": "victron", "confidence": 0.9, "position": 40}
                ],
                "model": [
                    {"value": "us5000", "confidence": 0.9, "position": 25},
                    {"value": "multiplus", "confidence": 0.9, "position": 50}
                ],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "compat"  # compatibility keywords
        },
        {
            "query": "Вага Dyness A48100 та максимальний струм для Pylontech US5000",
            "entities": {
                "manufacturer": [
                    {"value": "dyness", "confidence": 0.9, "position": 5},
                    {"value": "pylontech", "confidence": 0.9, "position": 40}
                ],
                "model": [
                    {"value": "a48100", "confidence": 0.9, "position": 12},
                    {"value": "us5000", "confidence": 0.9, "position": 55}
                ],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0},
                    {"key": "max_charge_current_a", "confidence": 0.90, "position": 25}
                ]
            },
            "expected": "simple"  # 2 models + 2 params - allowed
        },
        {
            "query": "Які є інвертори?",
            "entities": {
                "manufacturer": [],
                "model": [],
                "equipment_type": [{"value": "inverter", "confidence": 0.9, "position": 5}],
                "parameters": []
            },
            "expected": "complex"  # no models
        },
        {
            "query": "Вага Pylontech US5000",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 5}],
                "model": [{"value": "us5000", "confidence": 0.9, "position": 15}],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0}
                ]
            },
            "expected": "simple"  # 1 model + 1 param
        },
        {
            "query": "Порівняти 5 моделей батарей по ємності",
            "entities": {
                "manufacturer": [],
                "model": [
                    {"value": "model1", "position": 10},
                    {"value": "model2", "position": 20},
                    {"value": "model3", "position": 30},
                    {"value": "model4", "position": 40},
                    {"value": "model5", "position": 50}
                ],
                "equipment_type": [],
                "parameters": [
                    {"key": "capacity_ah", "confidence": 0.95, "position": 60}
                ]
            },
            "expected": "complex"  # 5 models > 2
        },
        {
            "query": "Вага, ємність, напруга для Pylontech US5000",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 30}],
                "model": [{"value": "us5000", "confidence": 0.9, "position": 40}],
                "equipment_type": [],
                "parameters": [
                    {"key": "weight_kg", "confidence": 0.95, "position": 0},
                    {"key": "capacity_ah", "confidence": 0.95, "position": 5},
                    {"key": "voltage_v", "confidence": 0.95, "position": 15}
                ]
            },
            "expected": "complex"  # 1 model + 3 params > 2
        },
        {
            "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.86, "position": 48}],
                "model": [{"value": None, "confidence": 0.95, "position": 57, "original_value": "LXP-LB-EU 10k"}],
                "equipment_type": [],
                "parameters": [
                    {"key": "max_charge_current_a", "confidence": 0.975, "position": 5}
                ]
            },
            "expected": "complex"  # model value is None - not canonical
        },
        # === NEW TYPES ===
        {
            "query": "Що робити якшо в мене такий статус E0049?",
            "entities": {
                "manufacturer": [],
                "model": [],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "error_code"
        },
        {
            "query": "What does error code E12 mean on my inverter?",
            "entities": {
                "manufacturer": [],
                "model": [],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "error_code"
        },
        {
            "query": "Що означає ця помилка на екрані?",
            "entities": {
                "manufacturer": [],
                "model": [],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "error_code"
        },
        {
            "query": "Яка розпіновка між інвертором та акумулятором Pylontech?",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 40}],
                "model": [],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "pinout"
        },
        {
            "query": "How to connect LuxPower to battery? Wiring diagram?",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 10}],
                "model": [],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "pinout"
        },
        {
            "query": "Схема підключення Victron MultiPlus",
            "entities": {
                "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 20}],
                "model": [{"value": "multiplus", "confidence": 0.9, "position": 30}],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "pinout"
        },
        {
            "query": "Дай документацію по моделі Pylontech US5000",
            "entities": {
                "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 25}],
                "model": [{"value": "us5000", "confidence": 0.9, "position": 35}],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "documentation"
        },
        {
            "query": "Give me the datasheet for LuxPower LXP-LB-EU",
            "entities": {
                "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 20}],
                "model": [{"value": "lxp_lb_eu", "confidence": 0.9, "position": 30}],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "documentation"
        },
        {
            "query": "Де знайти інструкцію до Dyness B4850?",
            "entities": {
                "manufacturer": [{"value": "dyness", "confidence": 0.9, "position": 15}],
                "model": [{"value": "b4850", "confidence": 0.9, "position": 25}],
                "equipment_type": [],
                "parameters": []
            },
            "expected": "documentation"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['query']}")
        print('=' * 70)

        result = determine_status(test['entities'], test['query'])
        status = result["status"]
        clarification = result["clarification"]
        intent = determine_intent_logic(status, test['entities'])
        param_bindings = build_param_bindings_logic(test['entities'])

        expected = test.get('expected', '?')
        status_mark = "✓" if status == expected else "✗"

        print(f"\nStatus: {status} (expected: {expected}) {status_mark}")
        print(f"Clarification needed: {clarification}")
        print(f"Intent: {intent}")
        print(f"Param bindings:")
        pprint(param_bindings)