"""
Test phi3.5:latest con prompt avanzado de análisis financiero NVIDIA.
Dataset: nvidia_real_news.csv (title + summary, sin ground truth)
Genera: predictions_csv + metadata_json
"""
import os, time, csv, json, sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from collections import Counter
import ollama

MODEL       = "phi3.5:latest"
TEMPERATURE = 0.0
CSV_PATH    = "../../data/raw/nvidia_real_news.csv"
RESULTS_DIR = "../../results/ollama/phi3.5-advanced"
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """═══════════════════════════════════════════════════════════
SENTIMENT CLASSIFICATION RULES
═══════════════════════════════════════════════════════════
POSITIVE  → credible catalyst for NVDA stock appreciation
NEUTRAL   → no material expected impact on NVDA price, OR no connection
            to NVIDIA or its ecosystem whatsoever
NEGATIVE  → credible risk factor for NVDA stock decline

⚠️  RELEVANCE GATE — apply before any sentiment reasoning:
If the headline refers to an entity, sector, or event with NO
plausible connection to NVIDIA's business, financials, or ecosystem,
classify immediately as:
  sentiment: "neutral", intensity: "weak", relevance: "none"
Do NOT force a connection where none exists.

Examples of headlines that must be NEUTRAL by this rule:
- "Fed raises interest rates by 25bps"         ← macro, indirect at best
- "Apple announces new iPhone model"           ← unrelated sector
- "Oil prices surge amid Middle East tensions" ← no NVDA exposure
- "Amazon reports strong retail quarter"       ← retail segment, not AWS capex

═══════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES FOR DIFFICULT CASES
═══════════════════════════════════════════════════════════
"AMD stock falls due to NVIDIA competitive pressure"
→ sentiment: positive, intensity: moderate
REASON: Competitor weakening = positive for NVDA

"Alphabet's TPU expanding beyond Google's own cloud"
→ sentiment: negative, intensity: moderate
REASON: Competitor gaining ground in AI compute = negative for NVDA

"BNP says NVIDIA offers better profile than Arista"
→ sentiment: positive, intensity: moderate
REASON: Analyst explicitly endorses NVIDIA over competitor

"Company X transitions from NVIDIA to AMD"
→ sentiment: negative, intensity: moderate
REASON: Direct customer loss, even if mentioned as a side note

"Jensen Huang cancels public appearance"
→ sentiment: negative, intensity: weak
REASON: CEO no-show at announced events signals reputational or operational friction
"""

USER_PROMPT_TEMPLATE = """Analyze the following financial news headline and summary for its sentiment impact on NVIDIA (NVDA) stock.

HEADLINE: {title}
SUMMARY: {summary}

STEP 1 — ENTITY MAPPING
─────────────────────────────────────────────
Identify all entities mentioned:
- Companies, products, technologies, regulators, macro factors
- Map each to its relationship with NVIDIA (direct / ecosystem / unrelated)

STEP 2 — RELEVANCE ASSESSMENT
─────────────────────────────────────────────
Based on entity mapping:
- What is the closest NVIDIA business segment affected?
- What is the revenue materiality of that segment?
- Is the impact direct (NVIDIA named) or inferred (ecosystem effect)?

⚠️  HARD STOP — ask yourself explicitly:
"Is there ANY credible mechanism by which this headline
affects NVIDIA's revenue, margins, competitive position,
or investor sentiment?"
If the answer is NO → relevance = "none"
                    → sentiment = "neutral"
                    → intensity = "weak"
                    → stop here, skip steps 3-5
                    → reasoning = "No plausible connection to NVIDIA's business or ecosystem identified."

Output: relevance = high | medium | low | none

STEP 3 — SENTIMENT DIRECTION (only if relevance ≠ none)
─────────────────────────────────────────────
Apply classification rules:
POSITIVE  → credible catalyst for NVDA stock appreciation
NEUTRAL   → no material expected impact on NVDA price
NEGATIVE  → credible risk factor for NVDA stock decline

Output: sentiment = positive | neutral | negative

STEP 4 — INTENSITY (only if relevance ≠ none)
─────────────────────────────────────────────
Rate the magnitude of the expected impact:
- strong:   likely >3% move in NVDA
- moderate: likely 1-3% move
- weak:     likely <1% move or uncertain

Output: intensity = strong | moderate | weak

STEP 5 — FINAL OUTPUT
─────────────────────────────────────────────
Respond ONLY with this exact JSON (no extra text):
{{
  "sentiment": "<positive|neutral|negative>",
  "intensity": "<strong|moderate|weak>",
  "relevance": "<high|medium|low|none>",
  "reasoning": "<one sentence max>"
}}"""


def load_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                "date":    row.get("date", "").strip(),
                "title":   row.get("title", "").strip(),
                "summary": row.get("summary", "").strip(),
            })
    return rows


def parse_response(raw: str) -> dict:
    """Extrae JSON de la respuesta del modelo, con fallback robusto."""
    import re
    raw = raw.strip()
    # Eliminar bloques markdown ```json ... ``` o ``` ... ```
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # Intentar parsear JSON — primero tal cual, luego limpiando comillas dobles escapadas
    for text in (raw, raw.replace('""', '"')):
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end])
                sentiment = str(data.get("sentiment", "neutral")).lower().strip()
                intensity = str(data.get("intensity", "weak")).lower().strip()
                relevance = str(data.get("relevance", "none")).lower().strip()
                reasoning = str(data.get("reasoning", "")).strip()
                if sentiment not in ("positive", "negative", "neutral"): sentiment = "neutral"
                if intensity not in ("strong", "moderate", "weak"): intensity = "weak"
                if relevance not in ("high", "medium", "low", "none"): relevance = "none"
                return {"sentiment": sentiment, "intensity": intensity,
                        "relevance": relevance, "reasoning": reasoning}
            except (json.JSONDecodeError, ValueError):
                continue
    # Fallback regex: extraer campos individuales del texto crudo
    def extract(pattern, default):
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).lower().strip() if m else default
    sentiment = extract(r'"sentiment"\s*:\s*"(\w+)"', "neutral")
    intensity = extract(r'"intensity"\s*:\s*"(\w+)"', "weak")
    relevance = extract(r'"relevance"\s*:\s*"(\w+)"', "none")
    reasoning = extract(r'"reasoning"\s*:\s*"([^"]{5,})"', raw[:120])
    if sentiment not in ("positive", "negative", "neutral"): sentiment = "neutral"
    if intensity not in ("strong", "moderate", "weak"): intensity = "weak"
    if relevance not in ("high", "medium", "low", "none"): relevance = "none"
    return {"sentiment": sentiment, "intensity": intensity,
            "relevance": relevance, "reasoning": reasoning}


# ─── Main ─────────────────────────────────────────────────────────────────────

try:
    ollama.list()
except Exception as e:
    print(f"⚠️  No se pudo conectar con Ollama: {e}")
    sys.exit(1)

rows = load_csv(CSV_PATH)
print(f"Dataset: {CSV_PATH} — {len(rows)} noticias")
print(f"\n{'='*65}\nMODELO: {MODEL} — prompt avanzado NVIDIA\n{'='*65}")

# Precalentar el modelo y cachear el system prompt enviando un mensaje dummy.
# Ollama reutiliza el KV-cache del prefix cuando el system prompt es idéntico
# entre llamadas consecutivas del mismo proceso (keep_alive mantiene el modelo cargado).
print("Precalentando modelo y cacheando system prompt...")
try:
    ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "ready"},
        ],
        options={"temperature": TEMPERATURE, "num_predict": 1, "keep_alive": "10m"},
    )
    print("  Modelo cargado y system prompt cacheado.\n")
except Exception as e:
    print(f"  Advertencia en precalentamiento: {e}\n")

results_data = []
inf_times_ms = []
errors = 0

for idx, row in enumerate(rows, 1):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=row["title"],
        summary=row["summary"][:600]
    )
    try:
        t0 = time.time()
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": TEMPERATURE, "num_predict": 200, "keep_alive": "10m"},
        )
        elapsed_ms = (time.time() - t0) * 1000
        raw_text = response["message"]["content"]
        parsed = parse_response(raw_text)
    except Exception as e:
        print(f"  Error muestra {idx}: {e}")
        elapsed_ms = 0.0
        parsed = {"sentiment": "neutral", "intensity": "weak",
                  "relevance": "none", "reasoning": f"ERROR: {e}"}
        errors += 1

    results_data.append({**row, **parsed, "tiempo_ms": round(elapsed_ms, 1)})
    inf_times_ms.append(elapsed_ms)

    if idx % 10 == 0 or idx == len(rows):
        dist = Counter(r["sentiment"] for r in results_data)
        print(f"  [{idx}/{len(rows)}] pos={dist['positive']} neu={dist['neutral']} neg={dist['negative']} | {np.mean(inf_times_ms):.0f}ms/avg")

# ─── Export ───────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

csv_out = f"{RESULTS_DIR}/phi3.5_advanced_nvidia_real_predictions.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "date", "title", "summary", "sentiment",
                                           "intensity", "relevance", "reasoning", "tiempo_ms"])
    writer.writeheader()
    for i, r in enumerate(results_data, 1):
        writer.writerow({
            "id": i, "date": r["date"], "title": r["title"], "summary": r["summary"],
            "sentiment": r["sentiment"], "intensity": r["intensity"],
            "relevance": r["relevance"], "reasoning": r["reasoning"],
            "tiempo_ms": r["tiempo_ms"],
        })
print(f"\n  CSV: {csv_out}")

dist_sentiment  = Counter(r["sentiment"]  for r in results_data)
dist_intensity  = Counter(r["intensity"]  for r in results_data)
dist_relevance  = Counter(r["relevance"]  for r in results_data)

json_out = f"{RESULTS_DIR}/phi3.5_advanced_nvidia_real_metadata.json"
meta = {
    "model": MODEL,
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": CSV_PATH,
    "n_samples": len(results_data),
    "approach": "advanced financial prompt — chain-of-thought with relevance gate",
    "temperature": TEMPERATURE,
    "prompt_version": "v2 — relevance gate + 5-step CoT + few-shot difficult cases",
    "distribution": {
        "sentiment":  dict(dist_sentiment),
        "intensity":  dict(dist_intensity),
        "relevance":  dict(dist_relevance),
    },
    "efficiency": {
        "inference_ms_mean": round(float(np.mean(inf_times_ms)), 2),
        "inference_ms_std":  round(float(np.std(inf_times_ms)), 2),
        "connection_errors": errors,
    },
}
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"  JSON: {json_out}")
print(f"\nDistribución: {dict(dist_sentiment)}")
print(f"Relevancia:   {dict(dist_relevance)}")
print(f"\n{'='*65}")
print(f"RESULTADOS GUARDADOS EN:")
print(f"  CSV  → {os.path.abspath(csv_out)}")
print(f"  JSON → {os.path.abspath(json_out)}")
print(f"{'='*65}")
