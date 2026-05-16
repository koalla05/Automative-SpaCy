"""
model_benchmark.py
──────────────────────────────────────────────────────────────────────────────
Benchmark multiple OpenAI models against the hybrid_classifier.

Tests 17 queries covering all classification categories:
  • Multi-param + multi-model specs
  • Compatibility (missing models, needs clarification)
  • Parallel configuration
  • Pinout / physical connection
  • Documentation / manual requests
  • Complex synthesis queries
  • Real-world support queries from production dataset

Metrics per model:
  • Accuracy  — % of correct status labels
  • Avg latency (ms)
  • P95 latency (ms)
  • LLM upgrade rate — % of KW-complex queries that were upgraded
  • Clarification accuracy

Usage:
    export OPENAI_API_KEY="sk-..."
    python model_benchmark.py

    # or pass key directly
    python model_benchmark.py sk-your-key-here

    # skip specific models
    python model_benchmark.py --skip gpt-4o
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import json
import statistics
import argparse
from typing import Optional
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not found. Run: pip install openai")
    sys.exit(1)

# ── local imports ─────────────────────────────────────────────────────────────
# Adjust the import path to match your project structure.
# The hybrid_classifier imports from pipeline.processors.llm_processor,
# so we patch sys.path so the benchmark can be run from the project root.
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try direct import first (if running from project root)
    from hybrid_classifier import classify
except ImportError:
    try:
        # Try with pipeline package structure
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from hybrid_classifier import classify
    except ImportError:
        print("Cannot import hybrid_classifier. Make sure the file is in the same directory or PYTHONPATH is set.")
        sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ══════════════════════════════════════════════════════════════════════════════
# MODELS TO BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

MODELS_TO_TEST = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5",
    "gpt-4o-mini",
]

# ══════════════════════════════════════════════════════════════════════════════
# TEST DATASET
# ══════════════════════════════════════════════════════════════════════════════
# 17 test cases covering all 6 thematic categories + real production queries.

TEST_DATASET = [

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 1: Multi-model + multi-parameter spec lookup
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C1-01",
        "category": "multi_param_spec",
        "query": "Дай вагу, ємність і діапазон напруги Powentech US5000",
        "entities": {
            "manufacturer": [{"value": "powentech", "confidence": 0.85, "position": 4}],
            "model": [{"value": "us5000", "confidence": 0.88, "position": 14}],
            "equipment_type": [],
            "parameters": [
                {"key": "weight_kg", "confidence": 0.95, "position": 4},
                {"key": "nominal_capacity_ah", "confidence": 0.95, "position": 10},
                {"key": "battery_voltage_range_v", "confidence": 0.95, "position": 16},
            ],
        },
        "expected_status": "complex",
        "expected_clarification": False,
        "note": "1 model + 3 params → complex (>2 params threshold)",
    },
    {
        "id": "C1-02",
        "category": "multi_param_spec",
        "query": "Дай ємність для Victron MultiPlus 245000120 та вагу і ємність для Dyness DL5.0C",
        "entities": {
            "manufacturer": [
                {"value": "victron", "confidence": 0.9, "position": 40},
                {"value": "dyness", "confidence": 0.9, "position": 60},
            ],
            "model": [
                {"value": "multiplus_24_5000_120", "confidence": 0.9, "position": 50},
                {"value": "dl5_0c", "confidence": 0.9, "position": 70},
            ],
            "equipment_type": [],
            "parameters": [
                {"key": "nominal_capacity_ah", "confidence": 0.95, "position": 4},
                {"key": "weight_kg", "confidence": 0.95, "position": 47},
                {"key": "nominal_capacity_ah", "confidence": 0.95, "position": 54},
            ],
        },
        "expected_status": "complex",
        "expected_clarification": False,
        "note": "2 models + 3 params → complex synthesis query",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 2: Compatibility + missing models (needs clarification)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C2-01",
        "category": "compat_missing_models",
        "query": "Чи працюють разом Victron і Dyness?",
        "entities": {
            "manufacturer": [
                {"value": "victron", "confidence": 0.9, "position": 15},
                {"value": "dyness", "confidence": 0.9, "position": 25},
            ],
            "model": [],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "compat",
        "expected_clarification": True,
        "note": "Compat keyword + only manufacturers (no models) → clarification needed",
    },
    {
        "id": "C2-02",
        "category": "compat_missing_models_complex",
        "query": "Підкажіть чи працює Victron Energy MultiPlus-II 48/10000/140-100/100 з генератором",
        "entities": {
            "manufacturer": [
                {"value": "victron", "confidence": 0.9, "position": 0},
                {"value": "dyness", "confidence": 0.9, "position": 10},
            ],
            "model": [],
            "equipment_type": [],
            "parameters": [
                {"key": "generator_support", "confidence": 0.95, "position": 65},
            ],
        },
        "expected_status": "complex",
        "expected_clarification": True,
        "note": "compat + missing models",
    },
    {
        "id": "C2-03",
        "category": "compat_missing_models",
        "query": "Dyness DL5.0C співпрацює із LuxPower SNA5000?",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.9, "position": 0},
                {"value": "luxpower", "confidence": 0.9, "position": 20},
            ],
            "model": [
                {"value": "dl5_0c", "confidence": 0.9, "position": 6},
                {"value": "sna5000", "confidence": 0.88, "position": 30},
            ],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "compat",
        "expected_clarification": False,
        "note": "compat with 2 valid models",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 3: Parallel configuration (NOT compatibility)
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C3-01",
        "category": "parallel",
        "query": "Скільки інверторів LuxPower можна об'єднати в одну систему?",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 20}],
            "model": [],
            "equipment_type": [{"value": "inverter", "confidence": 0.9, "position": 8}],
            "parameters": [],
        },
        "expected_status": "parallel",
        "expected_clarification": True,
        "note": "Parallel keyword — multi-unit config, not compatibility",
    },
    {
        "id": "C3-02",
        "category": "parallel",
        "query": "Яку максимальну кількість Victron 15 кВА можна зібрати в паралель на одній фазі? Система 220В.",
        "entities": {
            "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 30}],
            "model": [{"value": None, "confidence": 0.75, "position": 38, "original_value": "Victron 15 кВА"}],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "parallel",
        "expected_clarification": True,
        "note": "parallel + unresolved model",
    },
    {
        "id": "C3-03",
        "category": "parallel",
        "query": "Скільки можна максимум приєднати Dyness A4810 і Dyness B4850",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.9, "position": 25},
                {"value": "dyness", "confidence": 0.9, "position": 40},
            ],
            "model": [
                {"value": "a4810", "confidence": 0.9, "position": 32},
                {"value": "b4850", "confidence": 0.9, "position": 47},
            ],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "parallel",
        "expected_clarification": True,
        "note": "parallel with 1 known and 1 not canon model",
    },
    {
        "id": "C3-04",
        "category": "parallel",
        "query": "Скільки можна максимум приєднати Dyness A48100 і Dyness B4850",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.9, "position": 25},
                {"value": "dyness", "confidence": 0.9, "position": 40},
            ],
            "model": [
                {"value": "a48100", "confidence": 0.9, "position": 32},
                {"value": "b4850", "confidence": 0.9, "position": 47},
            ],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "parallel",
        "expected_clarification": False,
        "note": "parallel with 2 known models",
        },

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 4: Pinout / physical connection
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C4-01",
        "category": "pinout",
        "query": "До якого порту підключати акумулятор Dyness до LuxPower?",
        "entities": {
            "manufacturer": [
                {"value": "dyness", "confidence": 0.9, "position": 30},
                {"value": "luxpower", "confidence": 0.9, "position": 42},
            ],
            "model": [],
            "equipment_type": [{"value": "battery", "confidence": 0.9, "position": 22}],
            "parameters": [],
        },
        "expected_status": "pinout",
        "expected_clarification": True,
        "note": "Pinout keyword — port connection question",
    },
    {
        "id": "C4-02",
        "category": "pinout",
        "query": "Підкажіть номенклатуру трансформатора струму для LuxPower SNA 5000.",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 42}],
            "model": [{"value": "sna5000", "confidence": 0.85, "position": 51}],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "complex",
        "expected_clarification": False,
        "note": "Real production query #7 — current transformer pinout/accessory",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 5: Documentation / manual requests
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C5-01",
        "category": "documentation",
        "query": "Перешли мені Quick start guide для Victron",
        "entities": {
            "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 38}],
            "model": [],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "documentation",
        "expected_clarification": True,
        "note": "Doc request — ambiguous (dozens of Victron models), clarification needed",
    },
    {
        "id": "C5-02",
        "category": "documentation",
        "query": "А що це таке LAN dongle для LuxPower? Є на нього інструкція?",
        "entities": {
            "manufacturer": [{"value": "luxpower", "confidence": 0.9, "position": 30}],
            "model": [{"value": None, "confidence": 0.6, "position": 20, "original_value": "LAN dongle"}],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "documentation",
        "expected_clarification": True,
        "note": "doc request, unresolved model",
    },
    {
        "id": "C5-03",
        "category": "documentation",
        "query": "Я правильний мануал знайшов? https://support.huawei.com/...",
        "entities": {
            "manufacturer": [],
            "model": [],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "documentation",
        "expected_clarification": True,
        "note": "doc verification, no entities",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 6: Complex synthesis query
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "C6-01",
        "category": "complex_synthesis",
        "query": "Чи сумісний Victron MultiPlus з Dyness, як їх підключити паралельно, які потрібні кабелі і скинь документацію?",
        "entities": {
            "manufacturer": [
                {"value": "victron", "confidence": 0.9, "position": 13},
                {"value": "dyness", "confidence": 0.9, "position": 28},
            ],
            "model": [{"value": "multiplus", "confidence": 0.9, "position": 21}],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "complex",
        "expected_clarification": True,
        "note": "Multi-intent query: compat + parallel + pinout + docs. Complex should be final output",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # REAL-WORLD PRODUCTION QUERIES
    # ─────────────────────────────────────────────────────────────────────────
    {
        "id": "P-01",
        "category": "real_world",
        "query": "Підкажіть чи працює Victron Energy MultiPlus-II 48/10000/140-100/100 з генератором. І де знайти схему підключення якщо працює?",
        "entities": {
            "manufacturer": [{"value": "victron", "confidence": 0.9, "position": 15}],
            "model": [{"value": "multiplus_ii_48_10000_140_100", "confidence": 0.85, "position": 28}],
            "equipment_type": [{"value": "inverter_charger", "confidence": 0.9, "position": 22}],
            "parameters": [],
        },
        "expected_status": "complex",
        "expected_clarification": True,
        "note": "Complex as we have several requests. We don't know the model of generator",
    },
    {
        "id": "P-02",
        "category": "real_world",
        "query": "Які ще гібридні інвертори можуть працювати без АКБ, як мережеві?",
        "entities": {
            "manufacturer": [],
            "model": [],
            "equipment_type": [{"value": "hybrid_inverter", "confidence": 0.9, "position": 12}],
            "parameters": [],
        },
        "expected_status": "complex",
        "expected_clarification": True,
        "note": "catalogue/advisory query, no specific models → complex with clarification",
    },
    {
        "id": "P-03",
        "category": "real_world",
        "query": "Що означає ця помилка на екрані?",
        "entities": {
            "manufacturer": [],
            "model": [],
            "equipment_type": [],
            "parameters": [],
        },
        "expected_status": "complex",
        "expected_clarification": True,
        "note": "error keyword but no specific code → complex + clarification",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    expected_status: str
    expected_clarification: bool
    final_status: str
    final_clarification: bool
    kw_status: str
    llm_status: Optional[str]
    llm_reason: Optional[str]
    llm_called: bool
    upgraded: bool
    kw_ms: float
    llm_ms: float
    total_ms: float
    status_correct: bool
    clarification_correct: bool
    fully_correct: bool


@dataclass
class ModelReport:
    model: str
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.results)

    @property
    def status_accuracy(self):
        return sum(r.status_correct for r in self.results) / self.n if self.n else 0

    @property
    def full_accuracy(self):
        return sum(r.fully_correct for r in self.results) / self.n if self.n else 0

    @property
    def latencies(self):
        return [r.total_ms for r in self.results]

    @property
    def avg_ms(self):
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def p95_ms(self):
        if not self.latencies:
            return 0
        sorted_lats = sorted(self.latencies)
        idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        return sorted_lats[idx]

    @property
    def llm_calls(self):
        return sum(r.llm_called for r in self.results)

    @property
    def upgrades(self):
        return sum(r.upgraded for r in self.results)

    @property
    def upgrade_rate(self):
        called = self.llm_calls
        return self.upgrades / called if called else 0


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_case(tc: dict, client: OpenAI, model: str) -> CaseResult:
    """Run a single test case against a given model."""
    status, meta = classify(
        query=tc["query"],
        entities=tc["entities"],
        client=client,
        openai_model=model,
    )
    clarification = meta["final_clarification"]

    status_ok = status == tc["expected_status"]
    clarification_ok = clarification == tc["expected_clarification"]

    return CaseResult(
        case_id=tc["id"],
        category=tc["category"],
        query=tc["query"],
        expected_status=tc["expected_status"],
        expected_clarification=tc["expected_clarification"],
        final_status=status,
        final_clarification=clarification,
        kw_status=meta["kw_status"],
        llm_status=meta.get("llm_status"),
        llm_reason=meta.get("llm_reason"),
        llm_called=meta["llm_called"],
        upgraded=meta["upgraded"],
        kw_ms=meta["kw_ms"],
        llm_ms=meta["llm_ms"],
        total_ms=meta["total_ms"],
        status_correct=status_ok,
        clarification_correct=clarification_ok,
        fully_correct=status_ok and clarification_ok,
    )


def run_model_benchmark(model: str, client: OpenAI, verbose: bool = True) -> ModelReport:
    """Run all test cases against a model and return a ModelReport."""
    report = ModelReport(model=model)

    if verbose:
        print(f"\n{'━' * 90}")
        print(f"  Model: {model}  ({len(TEST_DATASET)} test cases)")
        print(f"{'━' * 90}")

    for tc in TEST_DATASET:
        try:
            result = run_case(tc, client, model)
            report.results.append(result)

            if verbose:
                mark = "✓" if result.fully_correct else ("~" if result.status_correct else "✗")
                llm_tag = f"→LLM={result.llm_status}" if result.llm_called else "(KW-only)"
                upgrade_tag = " ⬆" if result.upgraded else ""
                print(
                    f"  [{tc['id']:<7}] {mark}  "
                    f"KW={result.kw_status:<14} {llm_tag:<22}{upgrade_tag}\n"
                    f"           final={result.final_status:<14} clarif={str(result.final_clarification):<6} "
                    f"exp={result.expected_status:<14} clarif_exp={str(result.expected_clarification):<6} "
                    f"{result.total_ms:.0f}ms"
                )
                if not result.status_correct:
                    print(f"           ⚠ STATUS WRONG — got '{result.final_status}', expected '{result.expected_status}'")
                if not result.clarification_correct:
                    print(f"           ⚠ CLARIF WRONG — got {result.final_clarification}, expected {result.expected_clarification}")
                if result.llm_reason and not result.status_correct:
                    print(f"           LLM reason: {result.llm_reason}")

        except Exception as exc:
            report.errors.append({"id": tc["id"], "error": str(exc)})
            if verbose:
                print(f"  [{tc['id']:<7}] ERROR: {exc}")

    if verbose:
        print(f"\n  → Status accuracy : {report.status_accuracy * 100:.1f}%")
        print(f"  → Full accuracy   : {report.full_accuracy * 100:.1f}%  (status + clarification)")
        print(f"  → Avg latency     : {report.avg_ms:.0f}ms")
        print(f"  → P95 latency     : {report.p95_ms:.0f}ms")
        print(f"  → LLM calls       : {report.llm_calls}/{report.n}")
        print(f"  → LLM upgrades    : {report.upgrades} ({report.upgrade_rate * 100:.0f}% of LLM calls)")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(reports: list[ModelReport]):
    """Print a ranked comparison table across all tested models."""
    print(f"\n\n{'═' * 100}")
    print("  BENCHMARK SUMMARY — Model Comparison")
    print(f"{'═' * 100}\n")

    # Sort by full accuracy desc, then avg latency asc
    sorted_reports = sorted(reports, key=lambda r: (-r.full_accuracy, r.avg_ms))

    rows = []
    for rank, rpt in enumerate(sorted_reports, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"  {rank}."
        rows.append([
            medal,
            rpt.model,
            f"{rpt.status_accuracy * 100:.1f}%",
            f"{rpt.full_accuracy * 100:.1f}%",
            f"{rpt.avg_ms:.0f}ms",
            f"{rpt.p95_ms:.0f}ms",
            f"{rpt.llm_calls}/{rpt.n}",
            f"{rpt.upgrade_rate * 100:.0f}%",
            len(rpt.errors),
        ])

    headers = [
        "Rank", "Model",
        "Status\nAccuracy", "Full\nAccuracy",
        "Avg\nLatency", "P95\nLatency",
        "LLM\nCalls", "Upgrade\nRate", "Errors",
    ]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # Fallback plain text table
        col_w = [6, 22, 10, 10, 10, 10, 10, 10, 8]
        header_row = "  ".join(h.replace("\n", " ").ljust(w) for h, w in zip(headers, col_w))
        print(header_row)
        print("  " + "-" * (sum(col_w) + 2 * len(col_w)))
        for row in rows:
            print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    # Per-category accuracy breakdown
    print(f"\n{'─' * 100}")
    print("  Per-category accuracy breakdown\n")

    categories = sorted(set(tc["category"] for tc in TEST_DATASET))
    cat_rows = []
    for cat in categories:
        cat_cases = [tc["id"] for tc in TEST_DATASET if tc["category"] == cat]
        row = [cat, len(cat_cases)]
        for rpt in sorted_reports:
            cat_results = [r for r in rpt.results if r.category == cat]
            if cat_results:
                acc = sum(r.status_correct for r in cat_results) / len(cat_results)
                row.append(f"{acc * 100:.0f}%")
            else:
                row.append("—")
        cat_rows.append(row)

    cat_headers = ["Category", "N"] + [r.model for r in sorted_reports]
    if HAS_TABULATE:
        print(tabulate(cat_rows, headers=cat_headers, tablefmt="simple"))
    else:
        for row in cat_rows:
            print("  " + "  ".join(str(v).ljust(22) for v in row))

    # Recommendation
    print(f"\n{'─' * 100}")
    best = sorted_reports[0]
    fastest = min(reports, key=lambda r: r.avg_ms)
    print(f"  🏆  Best accuracy : {best.model} ({best.full_accuracy * 100:.1f}% full, {best.avg_ms:.0f}ms avg)")
    if fastest.model != best.model:
        print(f"  ⚡  Fastest       : {fastest.model} ({fastest.avg_ms:.0f}ms avg, {fastest.full_accuracy * 100:.1f}% full accuracy)")

    # Find best accuracy/speed tradeoff (normalize both axes)
    if len(reports) > 1:
        max_acc = max(r.full_accuracy for r in reports)
        min_acc = min(r.full_accuracy for r in reports)
        max_lat = max(r.avg_ms for r in reports)
        min_lat = min(r.avg_ms for r in reports)

        def score(r):
            norm_acc = (r.full_accuracy - min_acc) / (max_acc - min_acc + 1e-9)
            norm_lat = 1 - (r.avg_ms - min_lat) / (max_lat - min_lat + 1e-9)
            return 0.7 * norm_acc + 0.3 * norm_lat

        best_tradeoff = max(reports, key=score)
        if best_tradeoff.model not in (best.model, fastest.model):
            print(f"  ⚖️  Best tradeoff  : {best_tradeoff.model} ({best_tradeoff.full_accuracy * 100:.1f}% acc, {best_tradeoff.avg_ms:.0f}ms avg)")

    print(f"\n{'═' * 100}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS TO JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_results(reports: list[ModelReport], path: str = "benchmark_results.json"):
    """Persist full results to JSON for later analysis."""
    output = []
    for rpt in reports:
        output.append({
            "model": rpt.model,
            "summary": {
                "status_accuracy": round(rpt.status_accuracy, 4),
                "full_accuracy": round(rpt.full_accuracy, 4),
                "avg_ms": round(rpt.avg_ms, 2),
                "p95_ms": round(rpt.p95_ms, 2),
                "llm_calls": rpt.llm_calls,
                "upgrades": rpt.upgrades,
                "upgrade_rate": round(rpt.upgrade_rate, 4),
                "errors": len(rpt.errors),
            },
            "cases": [
                {
                    "id": r.case_id,
                    "category": r.category,
                    "query": r.query,
                    "expected_status": r.expected_status,
                    "expected_clarification": r.expected_clarification,
                    "final_status": r.final_status,
                    "final_clarification": r.final_clarification,
                    "kw_status": r.kw_status,
                    "llm_status": r.llm_status,
                    "llm_reason": r.llm_reason,
                    "llm_called": r.llm_called,
                    "upgraded": r.upgraded,
                    "kw_ms": round(r.kw_ms, 2),
                    "llm_ms": round(r.llm_ms, 2),
                    "total_ms": round(r.total_ms, 2),
                    "status_correct": r.status_correct,
                    "clarification_correct": r.clarification_correct,
                    "fully_correct": r.fully_correct,
                }
                for r in rpt.results
            ],
            "errors": rpt.errors,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI models with hybrid_classifier")
    parser.add_argument("api_key", nargs="?", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_TEST,
                        help="Models to test (space-separated)")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Models to skip")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-case output, show only summary")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Path for JSON results output")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save results to JSON")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided.\n"
              "  Set OPENAI_API_KEY environment variable or pass as first argument.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    models = [m for m in args.models if m not in args.skip]
    verbose = not args.quiet

    print(f"\n{'═' * 100}")
    print(f"  Hybrid Classifier — OpenAI Model Benchmark")
    print(f"  Models  : {', '.join(models)}")
    print(f"  Queries : {len(TEST_DATASET)}")
    print(f"{'═' * 100}")

    reports = []
    for model in models:
        try:
            report = run_model_benchmark(model, client, verbose=verbose)
            reports.append(report)
        except Exception as exc:
            print(f"\n  ✗ FAILED to run model '{model}': {exc}")

    if reports:
        print_summary(reports)
        if not args.no_save:
            save_results(reports, args.output)
    else:
        print("\n  No results collected — check your API key and model availability.")


if __name__ == "__main__":
    main()