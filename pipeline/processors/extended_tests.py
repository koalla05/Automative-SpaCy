"""
extended_tests.py
──────────────────────────────────────────────────────────────────────────────
Extended benchmark — focused on cases where the keyword classifier is likely
to return "complex" while the correct label is something specific.

Each test has:
    query       — raw user text
    entities    — extracted entities dict (as produced by the NER pipeline)
    expected    — ground-truth label
    kw_blind    — True  → KW is expected to return "complex" here
                  False → KW should handle it correctly (regression guard)
    note        — why this case is interesting

Run:
    export OPENAI_API_KEY="sk-..."
    python extended_tests.py
    python extended_tests.py sk-your-key

Tests are organised into sections:
    A  KW blind spots → error_code
    B  KW blind spots → pinout
    C  KW blind spots → documentation
    D  KW blind spots → compat
    E  KW blind spots → lifestyle
    F  KW blind spots → parallel
    G  True complex — must NOT be upgraded
    H  Regression — KW handles these; LLM must agree
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
from typing import Optional

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

sys.path.insert(0, "/mnt/user-data/uploads")
sys.path.insert(0, os.path.dirname(__file__))

from llm_processor import determine_status
from hybrid_classifier import classify, OPENAI_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _e(models=None, params=None, manufacturers=None, equipment=None):
    """Shorthand entity builder."""
    def model_entry(v, pos=0):
        return {"value": v, "confidence": 0.9, "position": pos}
    def param_entry(k, pos=0):
        return {"key": k, "confidence": 0.9, "position": pos}
    def mfr_entry(v, pos=0):
        return {"value": v, "confidence": 0.9, "position": pos}
    def eq_entry(v, pos=0):
        return {"value": v, "confidence": 0.9, "position": pos}

    return {
        "manufacturer": [mfr_entry(v, p) for v, p in (manufacturers or [])],
        "model":        [model_entry(v, p) for v, p in (models or [])],
        "equipment_type": [eq_entry(v, p) for v, p in (equipment or [])],
        "parameters":   [param_entry(k, p) for k, p in (params or [])],
    }

def _empty():
    return _e()


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ══════════════════════════════════════════════════════════════════════════
    # A  KW BLIND SPOTS → error_code
    #    KW needs explicit error keywords; these use indirect semantics.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "A",
        "query": "Що значить мигання індикатора на батареї?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Flashing indicator = fault semantics, no explicit error keyword",
    },
    {
        "section": "A",
        "query": "Інвертор пищить і не запускається",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Beeping inverter = alarm/fault; no error code in text",
    },
    {
        "section": "A",
        "query": "На дисплеї горить червоний індикатор, що це означає?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Red LED = status/fault; indirect phrasing",
    },
    {
        "section": "A",
        "query": "Інвертор не заряджає батарею, в чому проблема?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Troubleshooting / fault diagnosis without error code",
    },
    {
        "section": "A",
        "query": "My battery shows F04 on the screen",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Fault code F04 — KW regex covers E\\d+ but not F\\d+",
    },
    {
        "section": "A",
        "query": "Inverter keeps shutting down after a few seconds",
        "entities": _e(manufacturers=[("luxpower", 0)]),
        "expected": "error_code",
        "kw_blind": True,
        "note": "Protection trip / fault behaviour — no code word",
    },
    {
        "section": "A",
        "query": "Що означає OVP на інверторі?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "OVP = over-voltage protection alarm abbreviation; not in KW dict",
    },
    {
        "section": "A",
        "query": "Alarm BMS — що це таке?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": True,
        "note": "BMS alarm phrasing not covered by KW patterns",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # B  KW BLIND SPOTS → pinout
    #    KW covers розпін/wiring/схема — these use other phrasing.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "B",
        "query": "До якого порту підключати CAN шину?",
        "entities": _empty(),
        "expected": "pinout",
        "kw_blind": True,
        "note": "CAN bus port — port-level connection question not in KW dict",
    },
    {
        "section": "B",
        "query": "Який колір дроту плюс, а який мінус?",
        "entities": _empty(),
        "expected": "pinout",
        "kw_blind": True,
        "note": "Wire colour polarity — specific but not a KW pattern",
    },
    {
        "section": "B",
        "query": "RS485 чи RS232 для зв'язку з інвертором?",
        "entities": _empty(),
        "expected": "pinout",
        "kw_blind": True,
        "note": "Communication interface selection — indirect pinout question",
    },
    {
        "section": "B",
        "query": "How do I wire the CT clamp on LuxPower?",
        "entities": _e(manufacturers=[("luxpower", 20)]),
        "expected": "pinout",
        "kw_blind": True,
        "note": "CT clamp wiring — 'wire' not in pinout pattern list",
    },
    {
        "section": "B",
        "query": "Де знаходиться порт BMS на Pylontech US5000?",
        "entities": _e(manufacturers=[("pylontech", 25)], models=[("us5000", 35)]),
        "expected": "pinout",
        "kw_blind": True,
        "note": "BMS port location question — physical layout / pin reference",
    },
    {
        "section": "B",
        "query": "Яку перерізу кабель потрібен між батареєю і інвертором?",
        "entities": _empty(),
        "expected": "pinout",
        "kw_blind": True,
        "note": "Cable cross-section selection — physical installation question",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # C  KW BLIND SPOTS → documentation
    #    KW covers документація/manual/datasheet — these use other phrasing.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "C",
        "query": "Можна отримати файл з технічними характеристиками?",
        "entities": _empty(),
        "expected": "documentation",
        "kw_blind": True,
        "note": "File request for tech specs — 'характеристики' not in KW doc dict",
    },
    {
        "section": "C",
        "query": "Чи є у вас брошура по Dyness B4850?",
        "entities": _e(manufacturers=[("dyness", 20)], models=[("b4850", 28)]),
        "expected": "documentation",
        "kw_blind": True,
        "note": "Brochure request — брошура not in KW pattern list",
    },
    {
        "section": "C",
        "query": "Send me the quick start guide for Victron",
        "entities": _e(manufacturers=[("victron", 25)]),
        "expected": "documentation",
        "kw_blind": True,
        "note": "Quick-start guide — 'quick start' not in KW doc patterns",
    },
    {
        "section": "C",
        "query": "Де завантажити прошивку та документи для LXP-LB-EU?",
        "entities": _e(manufacturers=[("luxpower", 35)]),
        "expected": "documentation",
        "kw_blind": True,
        "note": "Download firmware + docs — 'прошивку' mixed with docs request",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # D  KW BLIND SPOTS → compat
    #    KW covers сумісний/compatible — these use other phrasing.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "D",
        "query": "Чи підійде Pylontech US3000C до Victron MPII?",
        "entities": _e(
            manufacturers=[("pylontech", 10), ("victron", 30)],
            models=[("us3000c", 18), ("mpii", 38)],
        ),
        "expected": "compat",
        "kw_blind": True,
        "note": "підійде (will it fit/suit) = compatibility intent; not in KW compat dict",
    },
    {
        "section": "D",
        "query": "Victron і Dyness — вони разом працюють?",
        "entities": _e(manufacturers=[("victron", 0), ("dyness", 10)]),
        "expected": "compat",
        "kw_blind": True,
        "note": "Informal together-work question; KW looks for specific words",
    },
    {
        "section": "D",
        "query": "LuxPower LXP підтримує батареї Dyness?",
        "entities": _e(
            manufacturers=[("luxpower", 0), ("dyness", 20)],
            models=[("lxp", 8)],
        ),
        "expected": "compat",
        "kw_blind": True,
        "note": "підтримує (supports) = compatibility; not a KW compat pattern",
    },
    {
        "section": "D",
        "query": "Will a Dyness A48100 talk to a Victron Cerbo GX?",
        "entities": _e(
            manufacturers=[("dyness", 8), ("victron", 28)],
            models=[("a48100", 16), ("cerbo_gx", 36)],
        ),
        "expected": "compat",
        "kw_blind": True,
        "note": "Communication compatibility — informal 'talk to' phrasing",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # E  KW BLIND SPOTS → lifestyle
    #    KW catches common greetings — these are unusual reactions / phrases.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "E",
        "query": "Зрозумів, дякс",
        "entities": _empty(),
        "expected": "lifestyle",
        "kw_blind": True,
        "note": "'дякс' is in KW, but 'Зрозумів' alone would miss — mixed reaction",
    },
    {
        "section": "E",
        "query": "Ясно, зрозуміло",
        "entities": _empty(),
        "expected": "lifestyle",
        "kw_blind": True,
        "note": "Acknowledgement reaction — not in KW pattern list",
    },
    {
        "section": "E",
        "query": "Все зрозуміло, більше питань немає",
        "entities": _empty(),
        "expected": "lifestyle",
        "kw_blind": True,
        "note": "Closing statement — semantic lifestyle, no KW keyword",
    },
    {
        "section": "E",
        "query": "got it, thanks",
        "entities": _empty(),
        "expected": "lifestyle",
        "kw_blind": True,
        "note": "Short EN acknowledgement — KW catches 'thanks' but not 'got it' alone",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # F  KW BLIND SPOTS → parallel
    #    KW covers 3-phase/parallel explicitly — these use indirect phrasing.
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "F",
        "query": "Скільки інверторів LuxPower можна об'єднати в одну систему?",
        "entities": _e(manufacturers=[("luxpower", 15)]),
        "expected": "parallel",
        "kw_blind": True,
        "note": "об'єднати (join/combine) inverters = parallel config; no '3-phase/stack' keyword",
    },
    {
        "section": "F",
        "query": "Можна збільшити потужність, підключивши два інвертори?",
        "entities": _empty(),
        "expected": "parallel",
        "kw_blind": True,
        "note": "Scaling power with two inverters = parallel/stacking intent",
    },
    {
        "section": "F",
        "query": "How many Victron units can I chain together?",
        "entities": _e(manufacturers=[("victron", 10)]),
        "expected": "parallel",
        "kw_blind": True,
        "note": "Chain = stack/parallel intent; not in KW parallel patterns",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # G  TRUE COMPLEX — must NOT be upgraded by LLM
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "G",
        "query": "Які є інвертори?",
        "entities": _e(equipment=[("inverter", 5)]),
        "expected": "complex",
        "kw_blind": False,
        "note": "Catalogue browsing — genuinely complex, no upgrade allowed",
    },
    {
        "section": "G",
        "query": "Порівняти 5 моделей батарей по ємності",
        "entities": _e(
            models=[("m1",10),("m2",20),("m3",30),("m4",40),("m5",50)],
            params=[("capacity_ah", 60)],
        ),
        "expected": "complex",
        "kw_blind": False,
        "note": "5 models > threshold; entity layer must keep as complex",
    },
    {
        "section": "G",
        "query": "Вага, ємність, напруга для Pylontech US5000",
        "entities": _e(
            manufacturers=[("pylontech", 30)], models=[("us5000", 40)],
            params=[("weight_kg", 0), ("capacity_ah", 5), ("voltage_v", 15)],
        ),
        "expected": "complex",
        "kw_blind": False,
        "note": "3 params > threshold; entity layer must keep as complex",
    },
    {
        "section": "G",
        "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.86, "position": 48}],
            "model": [{"value": None, "confidence": 0.95, "position": 57, "original_value": "LXP-LB-EU 10k"}],
            "equipment_type": [],
            "parameters": [{"key": "max_charge_current_a", "confidence": 0.975, "position": 5}],
        },
        "expected": "complex",
        "kw_blind": False,
        "note": "Unresolved model (value=None) — cannot do SQL lookup, must stay complex",
    },
    {
        "section": "G",
        "query": "Розрахуй скільки сонячних панелей потрібно для мого будинку",
        "entities": _empty(),
        "expected": "complex",
        "kw_blind": True,
        "note": "Open calculation request — genuinely complex, no specific label",
    },
    {
        "section": "G",
        "query": "Що краще — літій або свинцевий акумулятор?",
        "entities": _empty(),
        "expected": "complex",
        "kw_blind": True,
        "note": "Advisory / opinion question — complex, not compat",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # H  REGRESSION — KW handles correctly; hybrid must not break these
    # ══════════════════════════════════════════════════════════════════════════

    {
        "section": "H",
        "query": "Вага Pylontech US5000",
        "entities": _e(manufacturers=[("pylontech",5)], models=[("us5000",15)], params=[("weight_kg",0)]),
        "expected": "simple",
        "kw_blind": False,
        "note": "Regression: simple — 1 model, 1 param",
    },
    {
        "section": "H",
        "query": "Який максимальний струм заряджання на інверторі LuxPower LXP-LB-EU 10k?",
        "entities": _e(
            manufacturers=[("luxpower",50)], models=[("lxp_lb_eu_10k",60)],
            equipment=[("inverter",45)], params=[("max_charge_current_a",10)],
        ),
        "expected": "simple",
        "kw_blind": False,
        "note": "Regression: simple — 1 model, 1 param, canonical model name",
    },
    {
        "section": "H",
        "query": "Привіт!",
        "entities": _empty(),
        "expected": "lifestyle",
        "kw_blind": False,
        "note": "Regression: lifestyle greeting",
    },
    {
        "section": "H",
        "query": "Чи сумісний Pylontech US5000 з Victron MultiPlus?",
        "entities": _e(
            manufacturers=[("pylontech",15),("victron",40)],
            models=[("us5000",25),("multiplus",50)],
        ),
        "expected": "compat",
        "kw_blind": False,
        "note": "Regression: compat",
    },
    {
        "section": "H",
        "query": "Що робити якшо в мене такий статус E0049?",
        "entities": _empty(),
        "expected": "error_code",
        "kw_blind": False,
        "note": "Regression: error_code with explicit code",
    },
    {
        "section": "H",
        "query": "Яка розпіновка між інвертором та акумулятором Pylontech?",
        "entities": _e(manufacturers=[("pylontech",40)]),
        "expected": "pinout",
        "kw_blind": False,
        "note": "Regression: pinout",
    },
    {
        "section": "H",
        "query": "Дай документацію по моделі Pylontech US5000",
        "entities": _e(manufacturers=[("pylontech",25)], models=[("us5000",35)]),
        "expected": "documentation",
        "kw_blind": False,
        "note": "Regression: documentation",
    },
    {
        "section": "H",
        "query": "Чи можна зібрати 3-фазну систему з цих інверторів?",
        "entities": _empty(),
        "expected": "parallel",
        "kw_blind": False,
        "note": "Regression: parallel / 3-phase",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key) if key else None

    if not client:
        print("⚠️  No OPENAI_API_KEY — running KW-only (no LLM fallback).\n")

    sections = sorted(set(tc["section"] for tc in TEST_CASES))
    section_labels = {
        "A": "KW blind spots → error_code",
        "B": "KW blind spots → pinout",
        "C": "KW blind spots → documentation",
        "D": "KW blind spots → compat",
        "E": "KW blind spots → lifestyle",
        "F": "KW blind spots → parallel",
        "G": "True complex — must NOT upgrade",
        "H": "Regression — KW must continue to handle",
    }

    total_correct  = 0
    hybrid_correct = 0
    kw_correct     = 0
    rows_all       = []
    n = len(TEST_CASES)

    for section in sections:
        cases = [tc for tc in TEST_CASES if tc["section"] == section]
        print(f"\n{'═'*95}")
        print(f"  Section {section}: {section_labels.get(section, '')}  ({len(cases)} cases)")
        print(f"{'═'*95}")

        for tc in cases:
            query    = tc["query"]
            entities = tc["entities"]
            expected = tc["expected"]

            # ── KW standalone ────────────────────────────────────────────────
            t0 = time.perf_counter()
            kw_status = determine_status(entities, query)
            kw_ms = (time.perf_counter() - t0) * 1000

            # ── Hybrid ───────────────────────────────────────────────────────
            hybrid_status, meta = classify(query, entities, client)

            kw_ok     = kw_status == expected
            hybrid_ok = hybrid_status == expected
            if kw_ok:     kw_correct     += 1
            if hybrid_ok: hybrid_correct += 1
            total_correct += 1  # denominator count

            kw_mark     = "✓" if kw_ok     else "✗"
            hybrid_mark = "✓" if hybrid_ok else "✗"
            blind_tag   = "[BLIND]" if tc["kw_blind"] else "       "
            llm_tag     = f"→LLM={meta['llm_status']:<12}" if meta["llm_called"] else "  (no LLM)      "
            upgrade_tag = " ⬆" if meta["upgraded"] else "  "

            print(
                f"  {blind_tag} {kw_mark}KW={kw_status:<14} {hybrid_mark}HYB={hybrid_status:<14}"
                f" {llm_tag}{upgrade_tag}  exp={expected}"
            )
            print(f"           query: {query}")
            print(f"           note : {tc['note']}")
            if meta["llm_called"] and meta["llm_reason"]:
                print(f"           why  : {meta['llm_reason']}")
            print()

            rows_all.append({
                "sec": section,
                "blind": "●" if tc["kw_blind"] else "○",
                "query": query[:48] + ("…" if len(query) > 48 else ""),
                "expected": expected,
                "kw": kw_status,
                "kw_ok": kw_mark,
                "hybrid": hybrid_status,
                "hyb_ok": hybrid_mark,
                "upgraded": "⬆" if meta["upgraded"] else "",
                "ms": f"{meta['total_ms']:.0f}",
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    blind_cases    = [r for r in rows_all if r["blind"] == "●"]
    n_blind        = len(blind_cases)
    kw_blind_ok    = sum(1 for r in blind_cases if r["kw_ok"]  == "✓")
    hyb_blind_ok   = sum(1 for r in blind_cases if r["hyb_ok"] == "✓")
    upgraded_count = sum(1 for r in rows_all if r["upgraded"] == "⬆")

    print(f"\n{'═'*95}")
    print("  SUMMARY")
    print(f"{'═'*95}")

    summary = [
        ["", "Keyword only", "Hybrid (KW + LLM fallback)"],
        ["Overall correct",    f"{kw_correct}/{n}",           f"{hybrid_correct}/{n}"],
        ["Overall accuracy",   f"{kw_correct/n*100:.1f}%",    f"{hybrid_correct/n*100:.1f}%"],
        ["Blind-spot correct", f"{kw_blind_ok}/{n_blind}",    f"{hyb_blind_ok}/{n_blind}"],
        ["Blind-spot accuracy",f"{kw_blind_ok/n_blind*100:.1f}%", f"{hyb_blind_ok/n_blind*100:.1f}%"],
        ["LLM upgrades",       "—",                            str(upgraded_count)],
    ]

    if HAS_TABULATE:
        print(tabulate(summary[1:], headers=summary[0], tablefmt="rounded_outline"))
    else:
        for row in summary:
            print("  " + "  |  ".join(str(c).ljust(28) for c in row))

    # ── Per-section breakdown ─────────────────────────────────────────────────
    sec_rows = []
    for sec in sections:
        group = [r for r in rows_all if r["sec"] == sec]
        g = len(group)
        kw_s   = sum(1 for r in group if r["kw_ok"]  == "✓")
        hyb_s  = sum(1 for r in group if r["hyb_ok"] == "✓")
        sec_rows.append([
            f"{sec}: {section_labels.get(sec,'')[:40]}",
            f"{kw_s}/{g}",
            f"{hyb_s}/{g}",
        ])

    print("\n  Per-section accuracy:")
    if HAS_TABULATE:
        print(tabulate(sec_rows, headers=["Section", "KW", "Hybrid"], tablefmt="rounded_outline"))
    else:
        for row in sec_rows:
            print("  " + "  |  ".join(str(c).ljust(44) for c in row))

    print(f"\n{'═'*95}\n")


if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    run(api_key)