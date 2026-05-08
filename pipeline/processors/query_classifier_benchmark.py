"""
query_classifier_benchmark.py
──────────────────────────────────────────────────────────────────────────────
Compares two approaches for query status classification:
  1. Keyword/regex dictionary approach  (llm_processor.py)
  2. LLM approach via OpenAI GPT-4o-mini

Usage:
    pip install openai tabulate
    export OPENAI_API_KEY="sk-..."
    python query_classifier_benchmark.py

    # Or pass key inline:
    OPENAI_API_KEY=sk-... python query_classifier_benchmark.py
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import textwrap
import json
from typing import Optional

# ── Optional tabulate for pretty tables ──────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── OpenAI client ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found. Run:  pip install openai")
    sys.exit(1)

# ── Local keyword classifier ───────────────────────────────────────────────────
sys.path.insert(0, "/mnt/user-data/uploads")
from llm_processor import determine_status


# ══════════════════════════════════════════════════════════════════════════════
# VALID STATUS LABELS
# ══════════════════════════════════════════════════════════════════════════════
VALID_STATUSES = {
    "simple", "complex", "lifestyle", "compat",
    "parallel", "error_code", "pinout", "documentation",
}

# ══════════════════════════════════════════════════════════════════════════════
# TEST DATASET
# Each entry: query text + extracted_entities (for keyword approach) + expected
# ══════════════════════════════════════════════════════════════════════════════
TEST_CASES = [
    # ── simple ──────────────────────────────────────────────────────────────
    {
        "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.84, "position": 50}],
            "model": [{"value": "lxp_lb_eu_10k", "confidence": 0.88, "position": 60}],
            "equipment_type": [{"value": "inverter", "confidence": 0.88, "position": 45}],
            "parameters": [{"key": "max_charge_current_a", "confidence": 0.95, "position": 10}],
        },
        "expected": "simple",
    },
    {
        "query": "Вага Pylontech US5000",
        "entities": {
            "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 5}],
            "model": [{"value": "us5000", "confidence": 0.9, "position": 15}],
            "equipment_type": [],
            "parameters": [{"key": "weight_kg", "confidence": 0.95, "position": 0}],
        },
        "expected": "simple",
    },
    {
        "query": "Вага Dyness A48100 та максимальний струм для Pylontech US5000",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.9, "position": 5},
                {"value": "pylontech", "confidence": 0.9, "position": 40},
            ],
            "model": [
                {"value": "a48100", "confidence": 0.9, "position": 12},
                {"value": "us5000", "confidence": 0.9, "position": 55},
            ],
            "equipment_type": [],
            "parameters": [
                {"key": "weight_kg", "confidence": 0.95, "position": 0},
                {"key": "max_charge_current_a", "confidence": 0.90, "position": 25},
            ],
        },
        "expected": "simple",
    },
    # ── complex ──────────────────────────────────────────────────────────────
    {
        "query": "Які є інвертори?",
        "entities": {
            "manufacturer": [],
            "model": [],
            "equipment_type": [{"value": "inverter", "confidence": 0.9, "position": 5}],
            "parameters": [],
        },
        "expected": "complex",
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
    },
    # ── lifestyle ─────────────────────────────────────────────────────────────
    {
        "query": "Привіт!",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "lifestyle",
    },
    {
        "query": "Дякую за допомогу",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "lifestyle",
    },
    {
        "query": "ok",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "lifestyle",
    },
    # ── compat ────────────────────────────────────────────────────────────────
    {
        "query": "Чи сумісний Pylontech US5000 з Victron MultiPlus?",
        "entities": {
            "manufacturer": [
                {"value": "pylontech", "confidence": 0.9, "position": 15},
                {"value": "victron", "confidence": 0.9, "position": 40},
            ],
            "model": [
                {"value": "us5000", "confidence": 0.9, "position": 25},
                {"value": "multiplus", "confidence": 0.9, "position": 50},
            ],
            "equipment_type": [],
            "parameters": [],
        },
        "expected": "compat",
    },
    {
        "query": "Can I use Dyness battery with LuxPower inverter?",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.85, "position": 10},
                {"value": "luxpower", "confidence": 0.85, "position": 30},
            ],
            "model": [], "equipment_type": [], "parameters": [],
        },
        "expected": "compat",
    },
    # ── error_code ────────────────────────────────────────────────────────────
    {
        "query": "Що робити якшо в мене такий статус E0049?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "error_code",
    },
    {
        "query": "What does error code E12 mean on my inverter?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "error_code",
    },
    {
        "query": "Що означає ця помилка на екрані?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "error_code",
    },
    # ── pinout ────────────────────────────────────────────────────────────────
    {
        "query": "Яка розпіновка між інвертором та акумулятором Pylontech?",
        "entities": {
            "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 40}],
            "model": [], "equipment_type": [], "parameters": [],
        },
        "expected": "pinout",
    },
    {
        "query": "How to connect LuxPower to battery? Wiring diagram?",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 10}],
            "model": [], "equipment_type": [], "parameters": [],
        },
        "expected": "pinout",
    },
    {
        "query": "Схема підключення Victron MultiPlus",
        "entities": {
            "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 20}],
            "model": [{"value": "multiplus", "confidence": 0.9, "position": 30}],
            "equipment_type": [], "parameters": [],
        },
        "expected": "pinout",
    },
    # ── documentation ─────────────────────────────────────────────────────────
    {
        "query": "Дай документацію по моделі Pylontech US5000",
        "entities": {
            "manufacturer": [{"value": "pylontech", "confidence": 0.9, "position": 25}],
            "model": [{"value": "us5000", "confidence": 0.9, "position": 35}],
            "equipment_type": [], "parameters": [],
        },
        "expected": "documentation",
    },
    {
        "query": "Give me the datasheet for LuxPower LXP-LB-EU",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 20}],
            "model": [{"value": "lxp_lb_eu", "confidence": 0.9, "position": 30}],
            "equipment_type": [], "parameters": [],
        },
        "expected": "documentation",
    },
    {
        "query": "Де знайти інструкцію до Dyness B4850?",
        "entities": {
            "manufacturer": [{"value": "dyness", "confidence": 0.9, "position": 15}],
            "model": [{"value": "b4850", "confidence": 0.9, "position": 25}],
            "equipment_type": [], "parameters": [],
        },
        "expected": "documentation",
    },
    # ── parallel ──────────────────────────────────────────────────────────────
    {
        "query": "Чи можна зібрати 3-фазну систему з цих інверторів?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "parallel",
    },
    {
        "query": "Can I build a 3-phase stack with LuxPower inverters?",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 20}],
            "model": [], "equipment_type": [], "parameters": [],
        },
        "expected": "parallel",
    },
    # ── edge / tricky ─────────────────────────────────────────────────────────
    {
        "query": "Що значить мигання індикатора на батареї?",
        "entities": {"manufacturer": [], "model": [], "equipment_type": [], "parameters": []},
        "expected": "error_code",
        "note": "No explicit error code but flashing-indicator semantics",
    },
    {
        "query": "Яке напруга у Pylontech US3000C і яку вагу має Dyness A48100?",
        "entities": {
            "manufacturer": [
                {"value": "pylontech", "confidence": 0.9, "position": 10},
                {"value": "dyness", "confidence": 0.9, "position": 45},
            ],
            "model": [
                {"value": "us3000c", "confidence": 0.9, "position": 20},
                {"value": "a48100", "confidence": 0.9, "position": 55},
            ],
            "equipment_type": [],
            "parameters": [
                {"key": "voltage_v", "confidence": 0.9, "position": 0},
                {"key": "weight_kg", "confidence": 0.9, "position": 35},
            ],
        },
        "expected": "simple",
        "note": "2 models, 2 params — within simple threshold",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# LLM CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are a query classifier for a solar/battery technical support chatbot.
Classify each user query into exactly ONE of these status labels:

  simple        — Direct parameter lookup for 1–2 canonical models (1–2 params).
                  Answerable from a database without interpretation.
  complex       — Needs analysis, calculation, >2 models/params, vague, or no entities.
  lifestyle     — Greetings, farewells, thanks, small talk, reactions.
  compat        — Compatibility / integration question between two or more devices.
  parallel      — Parallel/stack / 3-phase configuration question.
  error_code    — Error code lookup, fault status, blinking/flashing indicator.
  pinout        — Wiring, pinout, connection diagram, terminal assignment.
  documentation — Request for datasheet, manual, PDF, specification sheet.

Rules:
- Respond with a JSON object and nothing else: {"status": "<label>", "reason": "<one sentence>"}
- Do NOT add markdown, backticks, or any extra text.
- "reason" is one concise sentence explaining your choice.
- The query may be in Ukrainian, Russian, or English.
""").strip()


def classify_with_llm(query: str, client: OpenAI, model: str = "gpt-5.4-mini") -> tuple[str, str, float]:
    """
    Returns (status, reason, latency_seconds).
    Falls back to 'unknown' on any error.
    """
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_completion_tokens=120,
        )
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        status = parsed.get("status", "unknown").lower().strip()
        reason = parsed.get("reason", "")
        if status not in VALID_STATUSES:
            status = "unknown"
    except Exception as e:
        status = "unknown"
        reason = f"ERROR: {e}"
    latency = time.perf_counter() - t0
    return status, reason, latency


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD CLASSIFIER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def classify_with_keywords(query: str, entities: dict) -> tuple[str, float]:
    """Returns (status, latency_seconds)."""
    t0 = time.perf_counter()
    status = determine_status(entities, query)
    latency = time.perf_counter() - t0
    return status, latency


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(api_key: Optional[str] = None, model: str = "gpt-5.4-mini"):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("❌  OPENAI_API_KEY not set. Export it or pass as argument.")
        sys.exit(1)

    client = OpenAI(api_key=key)

    rows = []
    kw_correct = 0
    llm_correct = 0
    kw_total_ms = 0.0
    llm_total_ms = 0.0

    print(f"\n{'─'*90}")
    print(f"  Running {len(TEST_CASES)} test cases   |   model: {model}")
    print(f"{'─'*90}\n")

    for i, tc in enumerate(TEST_CASES, 1):
        query = tc["query"]
        entities = tc["entities"]
        expected = tc["expected"]
        note = tc.get("note", "")

        # ── keyword approach ──────────────────────────────────────────────────
        kw_status, kw_lat = classify_with_keywords(query, entities)
        kw_ok = kw_status == expected
        if kw_ok:
            kw_correct += 1
        kw_total_ms += kw_lat * 1000

        # ── LLM approach ──────────────────────────────────────────────────────
        llm_status, llm_reason, llm_lat = classify_with_llm(query, client, model)
        llm_ok = llm_status == expected
        if llm_ok:
            llm_correct += 1
        llm_total_ms += llm_lat * 1000

        rows.append({
            "id": i,
            "query": query[:55] + ("…" if len(query) > 55 else ""),
            "expected": expected,
            "kw": kw_status,
            "kw_ok": "✓" if kw_ok else "✗",
            "llm": llm_status,
            "llm_ok": "✓" if llm_ok else "✗",
            "llm_ms": f"{llm_lat*1000:.0f}ms",
            "note": note,
            "llm_reason": llm_reason,
        })

        # progress line
        kw_mark = "✓" if kw_ok else "✗"
        llm_mark = "✓" if llm_ok else "✗"
        print(
            f"[{i:02d}] {kw_mark}KW={kw_status:<14} {llm_mark}LLM={llm_status:<14} "
            f"expected={expected:<14} ({llm_lat*1000:.0f}ms)"
        )
        if not kw_ok or not llm_ok:
            print(f"       ↳ query   : {query}")
            if not llm_ok:
                print(f"       ↳ llm why : {llm_reason}")
        if note:
            print(f"       ↳ note    : {note}")

    # ── Summary table ─────────────────────────────────────────────────────────
    n = len(TEST_CASES)
    kw_acc = kw_correct / n * 100
    llm_acc = llm_correct / n * 100
    kw_avg_ms = kw_total_ms / n
    llm_avg_ms = llm_total_ms / n

    print(f"\n{'═'*90}")
    print("  SUMMARY")
    print(f"{'═'*90}")

    summary = [
        ["", "Keyword/Regex", f"LLM ({model})"],
        ["Correct", f"{kw_correct}/{n}", f"{llm_correct}/{n}"],
        ["Accuracy", f"{kw_acc:.1f}%", f"{llm_acc:.1f}%"],
        ["Avg latency", f"{kw_avg_ms:.3f} ms", f"{llm_avg_ms:.0f} ms"],
        ["Total latency", f"{kw_total_ms:.1f} ms", f"{llm_total_ms:.0f} ms"],
    ]

    if HAS_TABULATE:
        print(tabulate(summary[1:], headers=summary[0], tablefmt="rounded_outline"))
    else:
        for row in summary:
            print("  " + "  |  ".join(str(c).ljust(20) for c in row))

    # ── Per-label breakdown ───────────────────────────────────────────────────
    labels = sorted(VALID_STATUSES)
    label_rows = []
    for label in labels:
        group = [tc for tc in TEST_CASES if tc["expected"] == label]
        if not group:
            continue
        result_rows_for_label = [r for r in rows if TEST_CASES[r["id"]-1]["expected"] == label]
        kw_c = sum(1 for r in result_rows_for_label if r["kw_ok"] == "✓")
        llm_c = sum(1 for r in result_rows_for_label if r["llm_ok"] == "✓")
        g = len(group)
        label_rows.append([label, f"{kw_c}/{g}", f"{llm_c}/{g}"])

    print("\n  Per-label accuracy:")
    if HAS_TABULATE:
        print(tabulate(label_rows, headers=["Label", "KW correct", "LLM correct"], tablefmt="rounded_outline"))
    else:
        for row in label_rows:
            print("  " + "  |  ".join(str(c).ljust(16) for c in row))

    # ── Mismatches detail ─────────────────────────────────────────────────────
    mismatches = [r for r in rows if r["kw_ok"] == "✗" or r["llm_ok"] == "✗"]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)} cases):")
        for r in mismatches:
            kw_flag = "" if r["kw_ok"] == "✓" else f"  ← KW got '{r['kw']}'"
            llm_flag = "" if r["llm_ok"] == "✓" else f"  ← LLM got '{r['llm']}' ({r['llm_reason']})"
            print(f"\n  [{r['id']:02d}] expected={r['expected']}")
            print(f"       query  : {r['query']}")
            if kw_flag:  print(f"       keyword: {kw_flag}")
            if llm_flag: print(f"       llm    : {llm_flag}")

    print(f"\n{'═'*90}")
    print("  Done.")
    print(f"{'═'*90}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PROS / CONS ANALYSIS (printed once at end)
# ══════════════════════════════════════════════════════════════════════════════

PROS_CONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               PROS & CONS: KEYWORD/REGEX  vs  LLM                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ KEYWORD / REGEX DICTIONARY                                                  ║
║  ✅ Pros                                                                     ║
║    • Deterministic — same input → same output, always                       ║
║    • Sub-millisecond latency, zero external API calls                       ║
║    • Free at runtime; no per-call cost                                      ║
║    • Fully offline; no dependency on third-party uptime                     ║
║    • Easy to audit, debug, and add specific patterns                        ║
║    • Works reliably on domain-specific abbreviations (E0049, LXP-LB-EU)    ║
║    • Uses entity data (model count, param count) not just raw text          ║
║                                                                              ║
║  ❌ Cons                                                                     ║
║    • Brittle for unseen phrasing / spelling variations / typos              ║
║    • Multilingual coverage requires hand-crafting per language              ║
║    • Patterns grow large and overlap — maintenance burden over time         ║
║    • Cannot reason about semantics ("мигання індикатора" → error_code)     ║
║    • False positives: a keyword hit can override correct entity signals     ║
║    • Adding new categories requires new pattern lists + careful ordering    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ LLM (GPT-5.4-mini)                                                           ║
║  ✅ Pros                                                                     ║
║    • Understands semantics; handles paraphrasing & implicit intent          ║
║    • Multilingual out-of-the-box (UA/RU/EN and mixed)                      ║
║    • No pattern maintenance — just update the system prompt                 ║
║    • Provides an explanation ("reason") useful for debugging                ║
║    • Handles edge cases naturally (flashing light → error_code)             ║
║    • Easy to extend: add a new label description in the prompt              ║
║                                                                              ║
║  ❌ Cons                                                                     ║
║    • ~200–2 000 ms latency per call (network round-trip)                   ║
║    • Per-call cost ($) — adds up at scale                                   ║
║    • Non-deterministic: temperature=0 helps but not 100% guaranteed        ║
║    • Fails if OpenAI is down; requires fallback strategy                    ║
║    • Cannot use extracted entity metadata (model count, param count)        ║
║      unless you inject it into the prompt                                   ║
║    • Harder to guarantee exact label compliance (may hallucinate labels)   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ RECOMMENDATION                                                               ║
║  Use a HYBRID approach:                                                      ║
║    1. Run keyword classifier first (free, instant).                         ║
║    2. If confidence is low (e.g. fell to "complex" default), call the LLM  ║
║       as a fallback to handle ambiguous/edge cases.                         ║
║    3. Cache LLM results for recurring query patterns.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(PROS_CONS)

    # Accept optional API key as first CLI argument
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    run_benchmark(api_key=api_key)