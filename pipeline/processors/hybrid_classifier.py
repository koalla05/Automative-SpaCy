"""
hybrid_classifier.py
──────────────────────────────────────────────────────────────────────────────
Hybrid query status classifier.

Flow:
    1. Run keyword/regex classifier (llm_processor.determine_status).
    2. If result == "complex"  →  call LLM with the query + entity context.
       The LLM may upgrade the status to a more specific label.
    3. If LLM returns "complex" or fails  →  keep "complex".

The LLM receives a summary of extracted entities so it can make the same
model/param-count-aware decisions that the keyword layer normally would.

Usage:
    export OPENAI_API_KEY="sk-..."
    python hybrid_classifier.py                    # demo / self-test
    python hybrid_classifier.py sk-your-key-here   # key as CLI arg

Importing:
    from hybrid_classifier import classify
    status, meta = classify(query, entities, client)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import textwrap
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found.  Run:  pip install openai")
    sys.exit(1)

sys.path.insert(0, "/mnt/user-data/uploads")
from llm_processor import determine_status

# ── try tabulate ──────────────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OPENAI_MODEL = "gpt-5.4-mini"

VALID_STATUSES = {
    "simple", "complex", "lifestyle", "compat",
    "parallel", "error_code", "pinout", "documentation",
}

# Labels the LLM is allowed to assign when KW returned "complex".
# "simple" is excluded: entity-count logic (≤2 models, ≤2 params) belongs to
# the keyword layer which already handles it correctly.  If KW said complex
# because of entity counts, that decision stays.
LLM_UPGRADEABLE = VALID_STATUSES - {"simple", "complex"}


# ══════════════════════════════════════════════════════════════════════════════
# LLM FALLBACK PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are a query classifier for a solar/battery technical support chatbot.

The keyword-based classifier already ran and returned "complex" — meaning it
could not match the query to a specific category.  Your job is to decide
whether the query actually belongs to a more specific label.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

error_code
  ANY fault, alarm, or device-misbehaviour question — with OR without an
  explicit error code number.  Use this when:
  • A specific code or abbreviation is shown  (E0049, F04, OVP, BMS alarm)
  • An indicator/LED is blinking, red, or flashing
  • The device is not working as expected: won't start, shuts down, beeps,
    won't charge, trips, overheats — i.e. troubleshooting / fault diagnosis
  Do NOT require a code number — symptom descriptions count.

pinout
  Any question about physical wiring, connections, or interface selection.
  Use this when:
  • How to connect device A to device B (cables, terminals, ports)
  • Which port to use (CAN, RS485, RS232, BMS port, communication port)
  • Wire colour, polarity, terminal labelling
  • Cable cross-section / cable sizing between two devices  ← this is pinout,
    NOT a calculation — it is a physical installation reference question
  • Wiring diagrams, connection schemes, pinouts
  Note: "RS485 чи RS232?" and "which interface for communication?" are pinout —
  they ask which physical connection method to use, not for analysis.

documentation
  Request for a document, file, or downloadable resource:
  datasheets, manuals, user guides, quick-start guides, brochures, spec
  sheets, firmware files, "where to download / find".

compat
  Whether two or more specific devices work together / are compatible.

parallel
  Multi-unit configuration: stacking, parallel operation, 3-phase setup,
  "how many units can I chain / combine / connect together", cascading.
  Ukrainian signals: об'єднати, каскад, підключити кілька, паралельно.
  English signals: chain, stack, how many units, multi-unit, scale up power.

lifestyle
  Greetings, farewells, thanks, acknowledgements, small talk.
  Examples: "ясно", "зрозуміло", "got it", "все зрозуміло".

complex
  Use ONLY when none of the above labels fit:
  • Open-ended browsing / catalogue queries ("what inverters do you have?")
  • Requests requiring genuine calculation or system design
  • Advisory / opinion questions ("which is better, lithium or lead-acid?")
  • Queries with >2 models or >2 parameters that need multi-row DB lookup
  • Unresolved model names (marked as unresolved in entity data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION SHORTCUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Device behaving unexpectedly, not starting, shutting down → error_code
• Code / alarm / indicator on screen → error_code
• Cable, wire, port, interface, connection, wiring → pinout
• How many units / can I stack / chain / combine → parallel
• Will X work with Y / is X compatible with Y → compat
• Give me the manual / datasheet / guide / file → documentation
• Acknowledgement / reaction / greeting → lifestyle
• Everything else → complex

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER return "simple" — that decision belongs to the entity layer.
- Respond ONLY with a JSON object, no markdown, no backticks:
  {"status": "<label>", "reason": "<one concise sentence>"}
- The query may be in Ukrainian, Russian, or English.
""").strip()


def _build_user_message(query: str, entities: dict) -> str:
    """
    Serialize the query + entity summary into the user turn so the LLM
    can apply the same model/param-count awareness as the keyword layer.
    """
    models = entities.get("model", [])
    params = entities.get("parameters", [])
    manufacturers = entities.get("manufacturer", [])
    equipment = entities.get("equipment_type", [])

    valid_models = [m["value"] for m in models if m.get("value")]
    null_models  = [m.get("original_value", "?") for m in models if not m.get("value")]
    param_keys   = [p["key"] for p in params]
    mfr_values   = [m["value"] for m in manufacturers if m.get("value")]
    eq_values    = [e["value"] for e in equipment if e.get("value")]

    ctx_lines = ["=== Query ===", query, "", "=== Extracted entities ==="]
    ctx_lines.append(f"valid_models   : {valid_models or 'none'}")
    if null_models:
        ctx_lines.append(f"unresolved_models: {null_models}  ← model name not in DB")
    ctx_lines.append(f"parameters     : {param_keys or 'none'}")
    ctx_lines.append(f"manufacturers  : {mfr_values or 'none'}")
    ctx_lines.append(f"equipment_type : {eq_values or 'none'}")
    ctx_lines.append(f"param_count    : {len(params)}")
    ctx_lines.append(f"valid_model_count: {len(valid_models)}")

    return "\n".join(ctx_lines)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def classify(
    query: str,
    entities: dict,
    client: Optional[OpenAI] = None,
    openai_model: str = OPENAI_MODEL,
) -> tuple[str, dict]:
    """
    Classify a query using the hybrid approach.

    Returns:
        (status, meta)  where meta contains timing and routing info.

    meta keys:
        kw_status       — result from keyword classifier
        final_status    — final status after hybrid resolution
        llm_called      — bool: was LLM invoked?
        llm_status      — LLM result (or None)
        llm_reason      — LLM explanation (or None)
        kw_ms           — keyword latency in ms
        llm_ms          — LLM latency in ms (or 0)
        total_ms        — total latency in ms
        upgraded        — bool: LLM changed the label away from "complex"
    """
    meta: dict = {
        "kw_status": None,
        "final_status": None,
        "llm_called": False,
        "llm_status": None,
        "llm_reason": None,
        "kw_ms": 0.0,
        "llm_ms": 0.0,
        "total_ms": 0.0,
        "upgraded": False,
    }

    # ── Step 1: keyword classifier ────────────────────────────────────────────
    t0 = time.perf_counter()
    kw_status = determine_status(entities, query)
    meta["kw_ms"] = (time.perf_counter() - t0) * 1000
    meta["kw_status"] = kw_status

    # ── Step 2: LLM fallback only when KW returned "complex" ─────────────────
    if kw_status != "complex" or client is None:
        meta["final_status"] = kw_status
        meta["total_ms"] = meta["kw_ms"]
        return kw_status, meta

    meta["llm_called"] = True
    t1 = time.perf_counter()
    try:
        user_msg = _build_user_message(query, entities)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        llm_status = parsed.get("status", "complex").lower().strip()
        llm_reason = parsed.get("reason", "")

        # Validate — reject unknown labels or forbidden "simple"
        if llm_status not in VALID_STATUSES or llm_status == "simple":
            llm_status = "complex"
            llm_reason = f"[rejected invalid label] {llm_reason}"

    except Exception as exc:
        llm_status = "complex"
        llm_reason = f"LLM error: {exc}"

    meta["llm_ms"] = (time.perf_counter() - t1) * 1000
    meta["llm_status"] = llm_status
    meta["llm_reason"] = llm_reason

    # ── Step 3: accept upgrade only for non-complex labels ───────────────────
    final = llm_status if llm_status in LLM_UPGRADEABLE else "complex"
    meta["final_status"] = final
    meta["upgraded"] = (final != "complex")
    meta["total_ms"] = meta["kw_ms"] + meta["llm_ms"]

    return final, meta


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST / DEMO
# ══════════════════════════════════════════════════════════════════════════════

DEMO_CASES = [
    # These are cases where KW returns "complex" — the interesting ones.
    {
        "query": "Що значить мигання індикатора на батареї?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "error_code",
        "note": "No explicit error keyword → KW=complex, LLM should upgrade",
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
                {"key": "voltage_v", "confidence": 0.95, "position": 15},
            ],
        },
        "expected": "complex",
        "note": "3 params > threshold → must stay complex (LLM should NOT upgrade)",
    },
    {
        "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.86, "position": 48}],
            "model": [{"value": None, "confidence": 0.95, "position": 57, "original_value": "LXP-LB-EU 10k"}],
            "equipment_type": [],
            "parameters": [{"key": "max_charge_current_a", "confidence": 0.975, "position": 5}],
        },
        "expected": "complex",
        "note": "Unresolved model (None value) → must stay complex",
    },
    {
        "query": "Які є інвертори?",
        "entities": {
            "manufacturer": [], "model": [],
            "equipment_type": [{"value": "inverter", "confidence": 0.9, "position": 5}],
            "parameters": [],
        },
        "expected": "complex",
        "note": "Catalogue browsing → genuinely complex",
    },
    {
        "query": "Порівняти 5 моделей батарей по ємності",
        "entities": {
            "manufacturer": [],
            "model": [
                {"value": "model1", "position": 10}, {"value": "model2", "position": 20},
                {"value": "model3", "position": 30}, {"value": "model4", "position": 40},
                {"value": "model5", "position": 50},
            ],
            "equipment_type": [],
            "parameters": [{"key": "capacity_ah", "confidence": 0.95, "position": 60}],
        },
        "expected": "complex",
        "note": "5 models > threshold → must stay complex",
    },
    # Non-complex cases — KW handles cleanly, LLM not called.
    {
        "query": "Вага Pylontech US5000",
        "entities": {
            "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 5}],
            "model": [{"value": "us5000", "confidence": 0.9, "position": 15}],
            "equipment_type": [],
            "parameters": [{"key": "weight_kg", "confidence": 0.95, "position": 0}],
        },
        "expected": "simple",
        "note": "KW handles — LLM not called",
    },
    {
        "query": "Схема підключення Victron MultiPlus",
        "entities": {
            "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 20}],
            "model": [{"value": "multiplus", "confidence": 0.9, "position": 30}],
            "equipment_type": [], "parameters": [],
        },
        "expected": "pinout",
        "note": "KW detects pinout — LLM not called",
    },
]


def run_demo(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key) if key else None

    if not client:
        print("⚠️  No OPENAI_API_KEY — LLM fallback disabled.  KW-only demo.\n")

    correct = 0
    rows = []

    print(f"\n{'─'*100}")
    print(f"  Hybrid classifier demo  ({len(DEMO_CASES)} cases)")
    print(f"{'─'*100}\n")

    for i, tc in enumerate(DEMO_CASES, 1):
        status, meta = classify(tc["query"], tc["entities"], client)
        ok = status == tc["expected"]
        if ok:
            correct += 1

        mark     = "✓" if ok else "✗"
        llm_tag  = f"→LLM={meta['llm_status']}" if meta["llm_called"] else "  (KW only)"
        upgrade  = " ⬆UPGRADED" if meta["upgraded"] else ""

        print(
            f"[{i:02d}] {mark} KW={meta['kw_status']:<14} {llm_tag:<22}{upgrade}"
            f"\n       final={status:<14} expected={tc['expected']:<14}"
            f"  kw={meta['kw_ms']:.2f}ms  llm={meta['llm_ms']:.0f}ms"
        )
        print(f"       {tc['note']}")
        if meta["llm_called"]:
            print(f"       llm_reason: {meta['llm_reason']}")
        if not ok:
            print(f"       ⚠ WRONG — got '{status}', expected '{tc['expected']}'")
        print()

        rows.append([
            i,
            tc["query"][:52] + ("…" if len(tc["query"]) > 52 else ""),
            meta["kw_status"],
            meta["llm_status"] or "—",
            status,
            tc["expected"],
            mark,
            f"{meta['total_ms']:.0f}ms",
        ])

    n = len(DEMO_CASES)
    print(f"\n{'═'*100}")
    print(f"  Result: {correct}/{n}  ({correct/n*100:.0f}%)")
    print(f"{'═'*100}")

    if HAS_TABULATE:
        headers = ["#", "Query", "KW", "LLM", "Final", "Expected", "OK", "Time"]
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    print()


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(api_key)