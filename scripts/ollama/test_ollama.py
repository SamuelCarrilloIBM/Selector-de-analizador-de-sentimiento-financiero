#!/usr/bin/env python3
"""
Script de prueba rápida para verificar la configuración de Ollama
"""

import ollama

def test_ollama():
    print("=== Prueba de Ollama ===\n")
    
    try:
        # Listar modelos disponibles
        print("1. Verificando modelos disponibles...")
        models = ollama.list()
        available_models = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
        
        if available_models:
            print(f"✓ Modelos encontrados: {available_models}")
        else:
            print("✗ No se encontraron modelos instalados")
            print("   Ejecuta: ollama pull llama3.2-vision")
            return False
        
        # Verificar si llama3.2-vision está disponible
        print("\n2. Verificando llama3.2-vision...")
        if any('llama3.2-vision' in m for m in available_models):
            print("✓ llama3.2-vision está instalado")
        else:
            print("✗ llama3.2-vision no está instalado")
            print("   Ejecuta: ollama pull llama3.2-vision")
            return False
        
        # Hacer una prueba simple
        print("\n3. Probando análisis de sentimiento...")
        test_text = "The company reported strong quarterly earnings, exceeding expectations."
        
        prompt = f"""Analiza el sentimiento del siguiente texto financiero y responde SOLO con una de estas palabras: positive, negative, o neutral.

Texto: {test_text}

Sentimiento:"""
        
        response = ollama.generate(
            model='llama3.2-vision',
            prompt=prompt,
            options={'temperature': 0.1}
        )
        
        sentiment = response['response'].strip()
        print(f"✓ Respuesta del modelo: {sentiment}")
        
        print("\n=== ✓ Ollama está configurado correctamente ===")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nAsegúrate de que:")
        print("1. Ollama está instalado (https://ollama.ai)")
        print("2. El servicio de Ollama está corriendo")
        print("3. Has descargado el modelo: ollama pull llama3.2-vision")
        return False

if __name__ == "__main__":
    test_ollama()