"""
Singleton analyzer para qwen2.5:7b — se inicializa una sola vez por proceso.
"""
import json, re, time
from typing import Optional
import ollama

MODEL       = "qwen2.5:7b"
TEMPERATURE = 0.0
KEEP_ALIVE  = "30m"  # más tiempo evita recargas entre scripts

# ─── Prompts ──────────────────────────────────────────────────────────────────
# Añade esto en qwen_sentiment.py junto a las otras constantes
USER_PROMPT_TEMPLATE = "HEADLINE: {title}\nSUMMARY: {summary}\nJSON:"

SYSTEM_PROMPT = """You are a financial analyst for NVIDIA (NVDA).

RELEVANCE GATE — evaluate first, always:
If no direct connection to NVIDIA → sentiment=neutral, intensity=weak, relevance=none. STOP.

NVIDIA KEY CONNECTIONS:
  POSITIVE: TSMC/SK Hynix good news, hyperscaler AI capex up, NVDA earnings beat,
            CUDA adoption, Blackwell demand, competitor weakening
  NEGATIVE: AMD/Intel/TPU/Arm gaining AI share, export restrictions,
            NVDA earnings miss, customer loss to competitor, AI bubble fears
  NEUTRAL:  Pharma, utilities, aviation, packaging, gold, REITs, generic ETFs,
            power semiconductors, OLED, sensors, unrelated microcontrollers

COMPETITOR LOGIC — critical, apply carefully:
  COMPETITOR WEAKENING = POSITIVE for NVDA:
    - A competing AI chip company loses market share to NVDA → positive
    - An analyst prefers NVDA over a networking competitor → positive
    - A rival GPU maker misses AI revenue targets → positive

  COMPETITOR STRENGTHENING = NEGATIVE for NVDA:
    - A CPU architecture company gains inference market share → negative
    - A cloud provider expands its own AI chip beyond internal use → negative
    - A rival chip company hires AI accelerator engineers → negative
    - A competing chip wins major AI lab contracts → negative

  KEY RULE: Ask "who benefits from this news?"
    If NVDA's competitor benefits → negative for NVDA
    If NVDA's competitor is hurt → positive for NVDA

SECTOR CONNECTIONS — explicit rules:
  Sole GPU manufacturer news → always relevant, high
  HBM memory supplier news → relevant, high (supplies NVDA chips)
  CPU architecture company gaining AI share → relevant as competitor, evaluate direction
  Rival GPU/AI accelerator company news → relevant as competitor, evaluate direction
  Industrial manufacturer adopting NVDA technology → positive/weak/medium
  Supply chain partner transitioning away from NVDA → negative/moderate/high
  NVDA executive cancels announced public event → negative/weak/medium
  Major hyperscaler increasing AI infrastructure spend → positive/moderate/high
  Major hyperscaler cutting AI infrastructure spend → negative/moderate/high
  Named company using NVIDIA platform/technology → positive/weak/medium
  CPU architecture company gaining AI inference share → negative/moderate/high
  AI sector volatility index rising on ROI doubts → negative/moderate/high
  Supply chain partner transitioning away from NVDA → negative/moderate/high
  Hyperscaler spending more on AI → positive/moderate/high
  Hyperscaler spending less or losing AI war → evaluate: if total AI spend still rising → positive
  Article directly about NVDA stock (buying/selling/forecasts) → always high relevance
  Intel CPU news → neutral/none (CPUs do not compete with NVIDIA GPUs)
  Rival chip company gaining AI accelerator market share → negative/moderate/high

MIXED SIGNALS — when headline contains both positive and negative:
  Record revenue BUT growth visibly slowing AND market skeptical → neutral/weak/medium
  CEO defends outlook BUT stock has not recovered losses → neutral/weak/medium
  Rule: if positive and negative signals roughly cancel → neutral

OUTPUT FORMAT — respond ONLY with this JSON, nothing else:
{"sentiment":"<positive|neutral|negative>","intensity":"<strong|moderate|weak>","relevance":"<high|medium|low|none>","reasoning":"<max 10 words>"}

EXAMPLES — these illustrate reasoning patterns, not specific cases:

Pattern: competitor weakens because of NVDA pressure
{"sentiment":"positive","intensity":"moderate","relevance":"high","reasoning":"Rival chip company losing share due to NVDA dominance"}

Pattern: analyst endorses NVDA over a competitor
{"sentiment":"positive","intensity":"moderate","relevance":"high","reasoning":"Analyst rates NVDA superior to networking rival"}

Pattern: competitor gains AI compute ground
{"sentiment":"negative","intensity":"moderate","relevance":"high","reasoning":"CPU architecture firm expanding AI inference market share"}

Pattern: NVDA named customer switches to rival
{"sentiment":"negative","intensity":"moderate","relevance":"high","reasoning":"Named NVDA customer moving workloads to competing chip"}

Pattern: NVDA direct strong positive event
{"sentiment":"positive","intensity":"strong","relevance":"high","reasoning":"NVDA reports record data center revenue above estimates"}

Pattern: NVDA stock explicitly falling on macro fear
{"sentiment":"negative","intensity":"moderate","relevance":"high","reasoning":"NVDA stock declining on AI infrastructure ROI concerns"}

Pattern: key supply chain partner bullish on AI demand
{"sentiment":"positive","intensity":"moderate","relevance":"high","reasoning":"Sole GPU manufacturer raises outlook on AI chip demand"}

Pattern: memory supplier invests heavily in AI chips
{"sentiment":"positive","intensity":"moderate","relevance":"high","reasoning":"HBM supplier expanding capacity for AI accelerator demand"}

Pattern: NVDA executive no-show at announced event
{"sentiment":"negative","intensity":"weak","relevance":"medium","reasoning":"NVDA CEO withdraws from previously announced public summit"}

Pattern: industrial company adopts NVDA technology
{"sentiment":"positive","intensity":"weak","relevance":"medium","reasoning":"Manufacturing firm deploys NVDA platform for production AI"}

Pattern: mixed signals cancel each other out
{"sentiment":"neutral","intensity":"weak","relevance":"medium","reasoning":"Record sales offset by slowing growth and investor doubt"}

Pattern: unrelated sector, no NVDA connection
{"sentiment":"neutral","intensity":"weak","relevance":"none","reasoning":"No plausible connection to NVIDIA business or ecosystem"}"""
# ─── Parser ───────────────────────────────────────────────────────────────────

# Compilar regex una sola vez al importar el módulo, no en cada llamada
_RE_JSON    = re.compile(r'\{[^{}]+\}', re.DOTALL)
_RE_STRIP   = re.compile(r'```(?:json)?\s*|\s*```')
_VALID_SENT = frozenset(("positive", "negative", "neutral"))
_VALID_INT  = frozenset(("strong", "moderate", "weak"))
_VALID_REL  = frozenset(("high", "medium", "low", "none"))

def _parse(raw: str) -> dict:
    raw = _RE_STRIP.sub("", raw).strip()
    match = _RE_JSON.search(raw)
    if match:
        try:
            data      = json.loads(match.group())
            sentiment = str(data.get("sentiment", "neutral")).lower().strip()
            intensity = str(data.get("intensity", "weak")).lower().strip()
            relevance = str(data.get("relevance", "none")).lower().strip()
            reasoning = str(data.get("reasoning", "")).strip()
            return {
                "sentiment": sentiment if sentiment in _VALID_SENT else "neutral",
                "intensity":  intensity if intensity  in _VALID_INT  else "weak",
                "relevance":  relevance if relevance  in _VALID_REL  else "none",
                "reasoning":  reasoning,
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"sentiment": "neutral", "intensity": "weak",
            "relevance": "none",    "reasoning": "parse error"}


# ─── Analyzer ─────────────────────────────────────────────────────────────────

# Mensajes base pre-construidos como constante — no se reconstruyen en cada llamada
_BASE_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]

# Opciones de inferencia pre-construidas como constante
_INFER_OPTIONS = {
    "temperature": TEMPERATURE,
    "num_predict": 80,    # suficiente para el JSON corto — antes era 200
    "keep_alive":  KEEP_ALIVE,
    "stop":        ["\n\n", "HEADLINE:"],  # corta en cuanto termina el JSON
}

_WARMUP_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 1,
    "keep_alive":  KEEP_ALIVE,
}


class QwenSentimentAnalyzer:
    """Wrapper sobre qwen2.5:7b vía Ollama. Instanciar una sola vez por proceso."""

    def __init__(self):
        self._warmup()

    def _warmup(self):
        print(f"[QwenSentimentAnalyzer] Cargando {MODEL}...")
        ollama.chat(
            model=MODEL,
            messages=_BASE_MESSAGES + [{"role": "user", "content": "ready"}],
            options=_WARMUP_OPTIONS,
        )
        print("[QwenSentimentAnalyzer] Listo.\n")

    def analyze(self, title: str, summary: str = "") -> dict:
        # Truncar summary a 400 chars — suficiente para clasificar,
        # menos tokens = menos tiempo de procesamiento
        user_content = USER_PROMPT_TEMPLATE.format(
            title=title,
            summary=summary[:400],
        )
        t0 = time.perf_counter()  # más preciso que time.time()
        response = ollama.chat(
            model=MODEL,
            messages=_BASE_MESSAGES + [{"role": "user", "content": user_content}],
            options=_INFER_OPTIONS,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        result = _parse(response["message"]["content"])
        result["tiempo_ms"] = elapsed_ms
        return result


# ─── Singleton ────────────────────────────────────────────────────────────────

_analyzer: Optional[QwenSentimentAnalyzer] = None

def get_analyzer() -> QwenSentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = QwenSentimentAnalyzer()
    return _analyzer