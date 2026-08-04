"""
Examen Práctico: Agente de IA con Streamlit
--------------------------------------------
Agente que responde preguntas académicas sobre tecnología, programación
e inteligencia artificial, conectado a un modelo de lenguaje mediante una
API compatible con OpenAI (Hugging Face Inference Providers por defecto;
también compatible con LM Studio local cambiando las variables de entorno).
"""

import os
import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE CONEXIÓN
# ----------------------------------------------------------------------
# Por defecto apunta a Hugging Face Inference Providers.
# Para usar LM Studio en local, cambia estas variables en tu .env a:
#   BASE_URL=http://localhost:1234/v1
#   API_KEY=lm-studio
#   MODEL=<nombre exacto del modelo cargado en LM Studio>
def obtener_config(clave: str, valor_por_defecto: str = "") -> str:
    """Busca la configuración primero en Streamlit Secrets (para el despliegue
    en la nube) y, si no existe, en las variables de entorno / .env (para
    ejecución local)."""
    try:
        if clave in st.secrets:
            return st.secrets[clave]
    except Exception:
        pass  # No hay archivo de secrets (ej. ejecución local sin secrets.toml)
    return os.getenv(clave, valor_por_defecto)


BASE_URL = obtener_config("BASE_URL", "https://router.huggingface.co/v1")
API_KEY = obtener_config("API_KEY", "")
MODEL = obtener_config("MODEL", "Qwen/Qwen2.5-3B-Instruct")

SYSTEM_PROMPT = (
    "Eres un agente tutor académico especializado en tecnología, programación "
    "e inteligencia artificial. Cuando el usuario pregunte sobre un concepto, "
    "responde SIEMPRE siguiendo esta estructura, en español:\n\n"
    "1. **Explicación**: describe el concepto de forma clara y precisa.\n"
    "2. **Ejemplo**: da un ejemplo concreto que ilustre el concepto.\n"
    "3. **Aplicación práctica**: menciona un caso real donde se use.\n"
    "4. **Pregunta de comprobación**: termina con una pregunta breve para "
    "verificar que el usuario entendió el tema.\n\n"
    "Si la consulta no está relacionada con tecnología, programación o "
    "inteligencia artificial, indícalo amablemente y pide al usuario que "
    "reformule su pregunta dentro de esos temas."
)

st.set_page_config(page_title="Agente de IA - Tutor Académico", page_icon="🤖", layout="centered")


def crear_cliente():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def consultar_agente(pregunta: str):
    """Envía la pregunta al modelo. Retorna (respuesta, None) o (None, error)."""
    try:
        cliente = crear_cliente()
        respuesta = cliente.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": pregunta},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        contenido = respuesta.choices[0].message.content
        if not contenido or not contenido.strip():
            return None, "El modelo devolvió una respuesta vacía. Intenta de nuevo."
        return contenido, None

    except APIConnectionError:
        return None, (
            "⚠️ No fue posible conectarse con el servidor del modelo.\n\n"
            "Verifica lo siguiente:\n"
            "- Si usas LM Studio: que esté abierto, el modelo cargado y el "
            "servidor iniciado (Developer → Start Server).\n"
            "- Si usas Hugging Face: que tengas conexión a internet y que "
            "el token configurado sea válido."
        )
    except APIStatusError as error:
        if error.status_code == 401:
            return None, "⚠️ Error de autenticación (401). Verifica tu API key en el archivo .env."
        if error.status_code == 404:
            return None, (
                f"⚠️ El modelo '{MODEL}' no fue encontrado. Verifica que el nombre "
                "coincida exactamente con el modelo cargado/disponible."
            )
        return None, f"⚠️ Error del servidor ({error.status_code}): {error.message}"
    except Exception as error:
        return None, f"⚠️ Error inesperado: {error}"


def main():
    st.title("🤖 Agente de IA — Tutor Académico")
    st.write(
        "Agente conversacional que responde preguntas sobre **tecnología, "
        "programación e inteligencia artificial**. Cada respuesta incluye "
        "una explicación, un ejemplo, una aplicación práctica y una "
        "pregunta de comprobación."
    )

    with st.sidebar:
        st.subheader("⚙️ Configuración de conexión")
        st.caption(f"**Servidor:** {BASE_URL}")
        st.caption(f"**Modelo:** {MODEL}")

    pregunta = st.text_area(
        "Escribe tu consulta:",
        placeholder="Ejemplo: ¿Qué es la recursividad en programación?",
        height=100,
    )

    if st.button("Consultar al Agente", type="primary"):
        if not pregunta or not pregunta.strip():
            st.error("⚠️ Debes escribir una consulta antes de continuar.")
        else:
            with st.spinner("Consultando al agente..."):
                respuesta, error = consultar_agente(pregunta.strip())

            if error:
                st.error(error)
            else:
                st.markdown("### 💬 Respuesta del agente")
                st.markdown(respuesta)


if __name__ == "__main__":
    main()
