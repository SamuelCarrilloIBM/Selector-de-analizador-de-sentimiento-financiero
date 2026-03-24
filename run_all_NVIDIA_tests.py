"""
Ejecuta todos los tests de manera secuencial.
Registra éxitos y fallos, y muestra un resumen al final.

Uso:
    ~/envs/sentimiento/bin/python run_all_NVIDIA_tests.py
"""
import subprocess
import sys
import os
import time
from datetime import datetime

VENV_PYTHON = os.path.expanduser("~/envs/sentimiento/bin/python")

# Si no estamos corriendo desde el venv, relanzar con él
if sys.executable != VENV_PYTHON and os.path.exists(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

PYTHON = sys.executable  # ya somos el venv

TESTS = [
    # (descripcion, directorio, script)
    ("BERT",           "tests/BERT",           "test_bert_nvidia.py"),
    ("FinBERT",        "tests/FinBERT",         "test_finbert_nvidia.py"),
    ("DistilRoBERTa",  "tests/DistilRoBERTa",   "test_distilroberta_nvidia.py"),
    ("DeBERTa v3",     "tests/DeBERTa v3",      "test_deberta_nvidia.py"),
    ("FinancialBERT",  "tests/FinancialBERT",   "test_financialbert_nvidia.py"),
    ("llama3.1",       "tests/llama3.1",        "test_llama3.1_nvidia.py"),
    ("llama3.2",       "tests/llama3.2",        "test_llama3.2_nvidia.py"),
    ("phi3.5",         "tests/phi3.5",          "test_phi3.5_nvidia.py")
]

results = []

print("=" * 65)
print(f"EJECUTANDO {len(TESTS)} TESTS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 65)

for name, cwd, script in TESTS:
    print(f"\n▶  {name}")
    print("-" * 65)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, "-u", script],
            cwd=cwd,
            timeout=3600,  # 1h max por test
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            results.append((name, "✓ OK", round(elapsed, 1), None))
            print(f"\n   → Completado en {elapsed:.1f}s")
        else:
            results.append((name, "✗ FALLO (exit code)", round(elapsed, 1), f"exit code {proc.returncode}"))
            print(f"\n   → Falló con exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        results.append((name, "✗ TIMEOUT", round(elapsed, 1), "superó 3600s"))
        print(f"\n   → TIMEOUT tras {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        results.append((name, "✗ ERROR", round(elapsed, 1), str(e)))
        print(f"\n   → Error inesperado: {e}")

# --- Resumen final ---
print("\n")
print("=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print(f"{'Test':<30} {'Estado':<25} {'Tiempo'}")
print("-" * 65)
for name, status, elapsed, detail in results:
    print(f"{name:<30} {status:<25} {elapsed}s")
    if detail:
        print(f"  {'':30} {detail}")

ok    = sum(1 for _, s, _, _ in results if s.startswith("✓"))
fail  = len(results) - ok
print("-" * 65)
print(f"Total: {ok}/{len(results)} completados correctamente, {fail} fallidos")
print("=" * 65)
