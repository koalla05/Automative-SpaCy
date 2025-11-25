# llm_processor.py

import os
import openai
from typing import Dict, Any

SYSTEM_PROMPT = """
You are the LLM module inside the Intelligence Preprocessing Gateway (IPG) for "Atmosfera" company.

Previous modules have already run:
- NER module (Spacy)
- Rule-based extraction
- Fuzzy matching for parameters

You must NOT:
- Search for models again
- Correct or validate models
- Choose agents (Supervisor does that)

You MUST:
1) Determine STATUS — "simple" or "complex"
2) Determine QUESTION_INTENT — one of:
   - sql_query
   - documentation_query
   - compatibility_query
   - calculation_query
   - uncertain
3) Confirm or correct the PARAMETER list:
   - Choose only parameter_key from the dictionary
   - Respect equipment type
4) Build PARAM_BINDINGS — links between models and parameters
   (take models ONLY from fuzzy/NER, don't change them)

⚠️ CRITICAL RULE FOR STATUS:
- If query is asking for specification parameters (ємність, вага, струм, напруга, потужність, etc.) for specific models
- AND each parameter can be found via SQL lookup
- THEN status = "simple" — REGARDLESS of how many models are in the query
- Having 3, 5, or 10 models does NOT make it "complex" if it's just listing specs
- "complex" means: compatibility checks, calculations, documentation, configuration, or vague queries

---

# 🧠 Determining STATUS

STATUS = "simple" if:
- Query contains at least one model
- Query meaning is to get **specification parameter** from YAML/SQL
- SQL can return answer without interpretation, logic, or combining
- Allowed:
  - 1 parameter + 1 model
  - 1 parameter + 2+ models
  - 2+ parameters + 1 model
  - 2+ parameters + 2+ models (as long as each parameter maps to a specific model)
- Query is essentially:
  **"Give me parameter value(s) for model(s)"**
- NUMBER OF MODELS DOESN'T MATTER - what matters is whether each parameter can be looked up directly

EXAMPLES of "simple":
- "What is max charge current for LuxPower LXP-LB-EU?"
- "Який максимальний струм заряджання для LuxPower LXP-LB-EU?"
- "Weight of Pylontech US5000"
- "Ємність та вага Dyness A48100"
- "Max AC power for Victron MultiPlus 48/5000 and Growatt SPF 5000"
- "Максимальний струм заряджання для Pylontech US5000 і ємність для Dyness A48100"
  → This is SIMPLE because:
     - Parameter 1 (max_charge_current_a) → Model 1 (us5000)
     - Parameter 2 (capacity_kwh) → Model 2 (a48100)
     - Each can be answered with a simple SQL lookup
- "Ємність та вага для Dyness A48100, максимальний струм для Pylontech US5000 і напруга для BYD HVS 7.7"
  → This is SIMPLE even with 3 models because:
     - capacity_kwh + weight_kg → a48100 (2 params, 1 model)
     - max_charge_current_a → us5000 (1 param, 1 model)
     - battery_voltage_range → hvs_7.7 (1 param, 1 model)
     - All are direct SQL lookups of specification data

STATUS = "complex" if:
- Compatibility query (inverter + battery, inverters together, AC-coupling)
- Wiring, schematics, pinouts, cables
- Firmware updates, monitoring, iSolarCloud, FusionSolar
- Instructions, configuration, modes, grid codes, menus
- Calculations ("how many", "what power", "how to distribute")
- General questions without parameters OR without models
- Questions requiring documentation reading or decision-making
- Vague or unclear queries without specific models
- Queries asking to COMPARE or CHOOSE between models (not just list specs)

EXAMPLES of "complex":
- "Can I connect Pylontech US5000 to Victron MultiPlus?"
- "How to configure Fronius Symo?"
- "Wiring diagram for Huawei SUN2000"
- "How many batteries do I need?"
- "Is Deye compatible with BYD?"

---

# 🧠 Determining QUESTION_INTENT

Choose ONE value:

- sql_query
  → specification parameters (model datasheets)
  → ANY query asking for parameter values from models
  → Examples: "вага", "ємність", "струм", "напруга", "потужність", "ККД"

- documentation_query
  → instructions, configuration, wiring, monitoring, error codes, firmware

- compatibility_query
  → "чи сумісні", "чи можна підключити", "в одну систему", "AC coupling"

- calculation_query
  → calculations: power, battery count, current, phase distribution, sizing
  → "скільки треба", "яку потужність видасть", "як розподілити"

- uncertain
  → ONLY use this when query is genuinely unclear/ambiguous/off-topic
  → DO NOT use if there are clear models + parameters (that's sql_query)
  → DO NOT use just because there are multiple models
  → Examples of uncertain: "hello", "what is solar?", "help me", no context

---

# 🔧 Parameter Dictionary (type → parameter_key → synonyms)

{parameters_list_with_type}

Format:
parameter_key (equipment_type): ["synonyms"]

You MUST choose ONLY parameter_key from this list.

---

# 🔧 Model Dictionary

{models_list_with_type}

You do NOT verify models.
Do NOT correct.
Do NOT reject.
Work with exactly what came from fuzzy.

---

# 📌 Rules for PARAMETERS

1. If fuzzy_parameters missed a parameter but it's clearly in query — add it.
2. If fuzzy found wrong type parameter (battery instead of inverter) — fix it.
3. Don't invent new parameters — use only parameter_key from dictionary.
4. If query is not about parameters (compatibility, docs) → PARAMETERS: NONE.
5. If one parameter for multiple models — still write separately MODEL: ...; PARAMS: ...

IMPORTANT for Ukrainian queries:
- "струм заряджання" OR "струм розряджання" → likely asking for BOTH:
  * max_charge_current_a
  * max_discharge_current_a
- "максимальний струм заряджання/розряджання" → BOTH parameters
- "ємність" → capacity_ah OR capacity_kwh (depends on context)
- Don't confuse "ККД заряджання" (efficiency) with "струм заряджання" (current)

# 📌 Rules for MODEL names in PARAM_BINDINGS

CRITICAL: Use the EXACT model value from extracted_entities["model"][]["value"]
- DO NOT concatenate manufacturer + model
- DO NOT normalize or change the model name
- Take it EXACTLY as provided by NER

Example:
If extracted_entities has:
  "model": [{"value": "us5000"}, {"value": "a48100"}]

Then PARAM_BINDINGS should use:
  MODEL: us5000; PARAMS: ...
  MODEL: a48100; PARAMS: ...

NOT:
  MODEL: pylontech_us5000  ❌
  MODEL: dyness_a48100     ❌

---

# ⚠ Response Format — STRICT

You must return exactly this format:

STATUS: <simple|complex>
INTENT: <sql_query|documentation_query|compatibility_query|calculation_query|uncertain>
PARAM_BINDINGS:
<lines or NONE>

Line format:
MODEL: <model_name_from_fuzzy>; PARAMS: param_key_1,param_key_2

IMPORTANT for multi-model queries:
- Analyze which parameter belongs to which model based on query structure
- Example: "Максимальний струм заряджання для Pylontech US5000 і ємність для Dyness A48100"
  → "струм заряджання" is FOR "Pylontech US5000"
  → "ємність" is FOR "Dyness A48100"
  → Create separate bindings:
    MODEL: us5000; PARAMS: max_charge_current_a
    MODEL: a48100; PARAMS: capacity_kwh

Examples:

STATUS: simple
INTENT: sql_query
PARAM_BINDINGS:
MODEL: lxp_lb; PARAMS: max_charge_current_a,max_discharge_current_a

STATUS: simple
INTENT: sql_query
PARAM_BINDINGS:
MODEL: us5000; PARAMS: max_charge_current_a
MODEL: a48100; PARAMS: capacity_kwh

STATUS: simple
INTENT: sql_query
PARAM_BINDINGS:
MODEL: a48100; PARAMS: capacity_kwh,weight_kg
MODEL: us5000; PARAMS: max_charge_current_a
MODEL: hvs_7.7; PARAMS: battery_voltage_range_full_load_v_min

STATUS: complex
INTENT: compatibility_query
PARAM_BINDINGS:
NONE

STATUS: complex
INTENT: documentation_query
PARAM_BINDINGS:
NONE

Any other format is an ERROR.
"""


class LLMProcessor:
    def __init__(self, par_lst, mod_lst, model: str = "gpt-4o-mini"):
        self.model = model
        self.parameters_list = par_lst
        self.models_list = mod_lst

    def process_question(self, extracted_entities: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """
        Sends extracted_entities + original text to OpenAI with system prompt,
        gets IPG-formatted raw text, then parses it into standard JSON.
        """
        # Prepare system prompt with embedded context
        system_content = SYSTEM_PROMPT.replace("{parameters_list_with_type}", self.parameters_list) \
            .replace("{models_list_with_type}", self.models_list)

        # User message with original query + extracted entities
        user_content = f"""Original query: {original_text}

Extracted entities from NER/fuzzy:
{extracted_entities}

Process this and return in the prescribed format."""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
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

        in_bindings = False
        for line in lines:
            line = line.strip()

            if line.startswith("STATUS:"):
                result["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("INTENT:"):
                result["intent"] = line.split(":", 1)[1].strip()
            elif line.startswith("PARAM_BINDINGS:"):
                in_bindings = True
                continue
            elif in_bindings:
                if line == "NONE":
                    result["param_bindings"] = []
                    break
                elif line.startswith("MODEL:"):
                    parts = line.split(";")
                    model_part = parts[0].split(":", 1)[1].strip()
                    params_part = parts[1].split(":", 1)[1].strip() if len(parts) > 1 else ""
                    params = [p.strip() for p in params_part.split(",")] if params_part else []
                    result["param_bindings"].append({"model": model_part, "parameters": params})

        return result


if __name__ == "__main__":
    from pprint import pprint

    # Example
    extracted_entities_example = {
        "manufacturer": [{"value": "LuxPower", "confidence": 0.84}],
        "model": [{"value": "lxp_lb", "confidence": 0.88, "original_value": "LXP-LB-EU"}],
        "equipment_type": [{"value": "inverter", "confidence": 0.88}],
        "parameters": [
            {"key": "max_charge_current_a", "confidence": 0.95},
            {"key": "max_discharge_current_a", "confidence": 0.90}
        ]
    }

    parameters_list = "max_charge_current_a (battery): ['максимальний струм заряджання']\nmax_discharge_current_a (battery): ['максимальний струм розряджання']"
    models_list = "lxp_lb (inverter)\nvictron_multiplus_ii_48_5000 (inverter)"

    processor = LLMProcessor(parameters_list, models_list)
    result = processor.process_question(
        extracted_entities_example,
        "Який максимальний струм заряджання/розряджання АКБ на інверторі LuxPower LXP-LB-EU 10k?"
    )
    pprint(result)