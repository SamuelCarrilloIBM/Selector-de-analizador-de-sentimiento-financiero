La alternativa a Unsloth: MLX Swift / MLX Examples
Apple creó la librería mlx-lm que permite cargar, ejecutar y entrenar modelos (LoRA/QLoRA) de forma increíblemente eficiente aprovechando la memoria unificada del Mac.

Requisitos mínimos:

Chip M1 o superior (M2/M3/M4 Pro o Max son ideales).

Al menos 16GB de RAM (aunque con 8GB podrías intentar modelos muy pequeños de 4-bit).

2. Instalación en Mac
Abre tu terminal y crea un entorno virtual para no ensuciar tu sistema:

Bash
python -m venv mlx_env
source mlx_env/bin/activate
pip install mlx-lm
3. Preparación de los datos
Para MLX, necesitas tres archivos .jsonl en una carpeta llamada data:

train.jsonl (Tus datos de entrenamiento)

valid.jsonl (Unos pocos datos para validar)

test.jsonl (Para probar al final)

El formato debe ser simple:

JSON
{"text": "<|user|>\nAnaliza: NVIDIA lanza nueva GPU.<|end|>\n<|assistant|>\nPositive, 0.9<|end|>"}
4. Ejecutar el Fine-Tuning (Comando Mágico)
MLX tiene un script ya preparado que hace todo el trabajo pesado. Solo tienes que ejecutar esto en tu terminal:

Bash
python -m mlx_lm.lora \
  --model microsoft/Phi-3.5-mini-instruct \
  --train \
  --data ./data \
  --batch-size 4 \
  --iters 500 \
  --checkpoint lora_checkpoints/phi35_finetuned.safetensors
--model: Descarga automáticamente el modelo de Hugging Face.

--iters: Número de pasos de entrenamiento.

--checkpoint: Donde se guardará tu "conocimiento financiero".

5. ¿Por qué es mejor usar MLX en Mac?
Memoria Unificada: A diferencia de un PC donde la RAM y la VRAM están separadas, en tu Mac el modelo puede usar casi toda la RAM disponible para entrenar.

Eficiencia Energética: Tu Mac no sonará como un avión despegando (a diferencia de una PC con NVIDIA).

Velocidad Real: Phi-3.5 es un modelo pequeño; en un Mac M2/M3, el entrenamiento de 500 iteraciones puede tardar apenas unos minutos.