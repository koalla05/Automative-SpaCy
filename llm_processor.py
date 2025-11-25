# llm_processor.py

import os
import openai
from typing import Dict, Any


SYSTEM_PROMPT = """
Ти — модуль LLM всередині Intelligence Preprocessing Gateway (IPG) компанії «Атмосфера».

Перед тобою вже відпрацювали:
- NER-модуль (Spacy)
- rule-based екстракція
- fuzzy matching параметрів

Ти НЕ повинен:
- шукати моделі наново
- перевіряти чи коригувати моделі
- вибирати агента (це робить Supervisor)

Ти ПОВИНЕН:
1) визначити STATUS — "simple" або "complex"
2) визначити QUESTION_INTENT — один із:
   - sql_query
   - documentation_query
   - compatibility_query
   - calculation_query
   - uncertain
3) підтвердити або скоригувати Список ПАРАМЕТРІВ:
   - обираючи тільки parameter_key зі словника
   - дотримуючись типу обладнання
4) Побудувати PARAM_BINDINGS — зв’язки між моделями та параметрами
   (моделі береш тільки з fuzzy / NER, не змінюєш їх).

---

# 🧠 Визначення STATUS

STATUS = "simple" якщо:
- у запиті Є хоча б одна модель  
- і сенс запиту зводиться до отримання **паспортного параметра**, який є в YAML/SQL  
- SQL може повернути відповідь без інтерпретації, без логіки, без комбінування  
- допускається:
  - один параметр + одна модель
  - один параметр + дві моделі
  - два параметри + одна модель
  - два параметри + дві моделі
- тобто запит є фактично:  
  **"Дай значення параметра для моделі"**

STATUS = "complex" якщо:
- запит про сумісність (інвертор + АКБ, інвертори між собою, AC-coupling)
- запит про схеми, wiring, розпіновки, кабелі
- запит про оновлення прошивки, моніторинг, iSolarCloud, FusionSolar
- запит про інструкції, налаштування, режими, коди мережі, меню
- запит про розрахунки ("скільки треба", "яку потужність видасть", "як розподілити")
- загальні питання без параметрів
- питання, де потрібно читати документацію або приймати рішення

---

# 🧠 Визначення QUESTION_INTENT

Обереш ОДНЕ значення:

- sql_query  
  → параметри паспортних даних (специфікація моделей)

- documentation_query  
  → інструкції, налаштування, wiring, моніторинг, коди помилок, прошивки

- compatibility_query  
  → "чи сумісні", "чи можна підключити", "в одну систему", "AC coupling"

- calculation_query  
  → розрахунки: потужність, кількість батарей, струм, розподіл фаз, sizing

- uncertain  
  → запит незрозумілий або немає моделей/контексту

---

# 🔧 Довідник параметрів (тип → parameter_key → синоніми)

{parameters_list_with_type}

Формат:
parameter_key (equipment_type): ["синоніми"]

ТИ МАЄШ вибирати ТІЛЬКИ parameter_key зі списку.

---

# 🔧 Довідник моделей

{models_list_with_type}

Ти НЕ перевіряєш моделі.  
Не коригуєш.  
Не відкидаєш.  
Працюєш рівно з тими, що надійшли з fuzzy.

---

# 📌 Правила щодо ПАРАМЕТРІВ

1. Якщо fuzzy_parameters пропустили параметр, а він явно є в запиті — додай.
2. Якщо fuzzy знайшов параметр не того типу (АКБ замість інвертора) — виправ.
3. Не вигадуй нові параметри — використовуй тільки parameter_key зі словника.
4. Якщо запит не про параметри (compatibility, docs) → PARAMETERS: NONE.
5. Якщо параметр один для кількох моделей — все одно записуй окремо MODEL: ...; PARAMS: ...

---

# ⚠ Формат відповіді — СТРОГО

Ти повинен повернути рівно цей формат:

STATUS: <simple|complex>
INTENT: <sql_query|documentation_query|compatibility_query|calculation_query|uncertain>
PARAM_BINDINGS:
<рядки або NONE>

Формат рядка:
MODEL: <model_name_from_fuzzy>; PARAMS: param_key_1,param_key_2

Приклади:

STATUS: simple
INTENT: sql_query
PARAM_BINDINGS:
MODEL: Victron MultiPlus-II 48/5000; PARAMS: inverter_nominal_ac_power

STATUS: complex
INTENT: compatibility_query
PARAM_BINDINGS:
NONE

STATUS: documentation_query
INTENT: documentation_query
PARAM_BINDINGS:
NONE

Будь-який інший формат — ПОМИЛКА.
"""

class LLMProcessor:
    def __init__(self, par_lst_w_tp, mod_lst_w_tp, model: str = "gpt-4o-mini"):
        self.model = model
        self.parameters_list_with_type = par_lst_w_tp
        self.models_list_with_type = mod_lst_w_tp

    def process_question(self, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends extracted_entities to OpenAI with system prompt, gets IPG-formatted raw text,
        then parses it into standard JSON.
        """
        # Prepare user message with embedded context
        user_prompt = SYSTEM_PROMPT.replace("{parameters_list_with_type}", self.parameters_list_with_type)\
                                   .replace("{models_list_with_type}", self.models_list_with_type)
        user_prompt += f"\n\n# Вхідні дані NER/fuzzy:\n{extracted_entities}\n"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": user_prompt},
                {"role": "user", "content": "Оброби це і поверни строго в форматі IPG"}
            ],
            temperature=0
        )

        raw_text = response["choices"][0]["message"]["content"].strip()

        return self.parse_ipg_output(raw_text)

    def parse_ipg_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Converts IPG raw text format to structured JSON.
        """
        lines = raw_text.splitlines()
        result = {"status": None, "intent": None, "param_bindings": []}

        for line in lines:
            if line.startswith("STATUS:"):
                result["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("INTENT:"):
                result["intent"] = line.split(":", 1)[1].strip()
            elif line.startswith("PARAM_BINDINGS:"):
                continue  # skip header
            elif line.startswith("MODEL:"):
                parts = line.split(";")
                model_part = parts[0].split(":", 1)[1].strip()
                params_part = parts[1].split(":", 1)[1].strip() if len(parts) > 1 else ""
                params = [p.strip() for p in params_part.split(",")] if params_part else []
                result["param_bindings"].append({"model": model_part, "parameters": params})
            elif line.strip() == "NONE":
                result["param_bindings"] = []

        return result


if __name__ == "__main__":
    from pprint import pprint

    # Example
    extracted_entities_example = {
        "model": [{"value": "Victron MultiPlus-II 48/5000"}],
        "manufacturer": [{"value": "Victron"}],
        "eq_type": [{"value": "інвертор"}],
        "parameters": [{"key": "inverter_nominal_ac_power", "value": None}]
    }

    parameters_list = "inverter_nominal_ac_power (inverter): ['nominal AC power']\ninverter_max_current (inverter): ['max current']"
    models_list = "Victron MultiPlus-II 48/5000 (inverter)\nSofar 10KTLX-G3-A (inverter)"

    processor = LLMProcessor()
    result = processor.process_question(extracted_entities_example, parameters_list, models_list)
    pprint(result)
