# Agente de IA — Tutor Académico

Aplicación web en Streamlit que implementa un agente de IA capaz de responder preguntas académicas sobre **tecnología, programación e inteligencia artificial**. Cada respuesta incluye una explicación del concepto, un ejemplo, una aplicación práctica y una pregunta de comprobación.

## Requisitos

- Python 3.10+
- Streamlit
- Una API compatible con OpenAI para el modelo:
  - **Hugging Face Inference Providers** (configuración por defecto de este proyecto), o
  - **LM Studio** ejecutándose en local (`http://localhost:1234/v1`)

## Instalación

1. Crear el entorno virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Linux / macOS
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Copiar `.env.example` a `.env` y completar las credenciales:
   ```bash
   copy .env.example .env     # Windows
   cp .env.example .env       # Linux / macOS
   ```

## Configuración del modelo

### Opción A — Hugging Face (configuración por defecto)
1. Genera un token en https://huggingface.co/settings/tokens con el permiso **"Make calls to Inference Providers"**.
2. En tu `.env`:
   ```
   BASE_URL=https://router.huggingface.co/v1
   API_KEY=hf_tu_token_aqui
   MODEL=Qwen/Qwen2.5-3B-Instruct
   ```

### Opción B — LM Studio en local
1. Abre LM Studio, descarga y carga un modelo Instruct liviano (Llama 3.2 1B/3B, Qwen2.5 1.5B/3B, Phi-3 Mini o similar).
2. Ve a **Developer → Server Settings** e inicia el servidor.
3. En tu `.env`:
   ```
   BASE_URL=http://localhost:1234/v1
   API_KEY=lm-studio
   MODEL=nombre-exacto-del-modelo-cargado
   ```

## Ejecución

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

## Uso

1. Escribe una consulta relacionada con tecnología, programación o inteligencia artificial en el campo de texto.
2. Presiona **"Consultar al Agente"**.
3. El agente responderá con: explicación, ejemplo, aplicación práctica y una pregunta de comprobación.

**Ejemplo de consulta:** *¿Qué es la recursividad en programación?*

## Manejo de errores

La aplicación muestra mensajes claros cuando:
- La consulta está vacía.
- No hay conexión con el servidor del modelo (LM Studio cerrado / sin internet).
- La API key es inválida (error 401).
- El modelo configurado no existe o no está disponible (error 404).

## Autor

Jorge Abraham Fajardo López — 20231900189
