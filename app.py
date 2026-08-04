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
from datetime import datetime
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
MODEL = obtener_config("MODEL", "meta-llama/Llama-3.1-8B-Instruct")

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

# Preguntas preestablecidas que el usuario puede disparar con un clic
PREGUNTAS_PREESTABLECIDAS = [
    {"icono": "🔁", "texto": "¿Qué es la recursividad en programación?"},
    {"icono": "🧠", "texto": "¿Qué es el machine learning y en qué se diferencia de la IA tradicional?"},
    {"icono": "🗂️", "texto": "¿Qué es una base de datos relacional?"},
    {"icono": "☁️", "texto": "¿Qué es la computación en la nube?"},
    {"icono": "🔒", "texto": "¿Qué es la criptografía y por qué es importante en ciberseguridad?"},
    {"icono": "🧩", "texto": "¿Qué es la programación orientada a objetos?"},
]

st.set_page_config(
    page_title="Agente de IA - Tutor Académico",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# ESTILOS PERSONALIZADOS
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f1220 0%, #171b2e 100%);
        }
        .main-header {
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #a855f7 100%);
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.35);
            margin-bottom: 1.4rem;
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 1.9rem;
            margin: 0 0 0.3rem 0;
        }
        .main-header p {
            color: rgba(255,255,255,0.9);
            font-size: 0.98rem;
            margin: 0;
        }
        .badge-row span {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            color: #fff;
            border-radius: 999px;
            padding: 3px 12px;
            font-size: 0.78rem;
            margin-right: 6px;
            margin-top: 8px;
        }
        div.stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(124, 58, 237, 0.35);
            padding: 0.55rem 0.8rem;
            font-weight: 500;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            border-color: #7c3aed;
            color: #7c3aed;
            transform: translateY(-1px);
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
            border: none !important;
        }
        .chat-card {
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(124, 58, 237, 0.18);
        }
        .chat-user {
            background: rgba(124, 58, 237, 0.10);
        }
        .chat-agent {
            background: rgba(255, 255, 255, 0.03);
        }
        .chat-meta {
            font-size: 0.75rem;
            opacity: 0.6;
            margin-bottom: 0.35rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(124, 58, 237, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def procesar_pregunta(pregunta: str):
    """Guarda la pregunta del usuario y dispara la consulta al agente."""
    pregunta = pregunta.strip()
    if not pregunta:
        return
    hora = datetime.now().strftime("%H:%M")
    st.session_state.historial.append({"rol": "usuario", "contenido": pregunta, "hora": hora})
    st.session_state.pendiente = pregunta


def main():
    if "historial" not in st.session_state:
        st.session_state.historial = []
    if "pendiente" not in st.session_state:
        st.session_state.pendiente = None

    # ---------------- HEADER ----------------
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 Agente de IA — Tutor Académico</h1>
            <p>Resuelve tus dudas de <b>tecnología, programación e inteligencia artificial</b>
            con explicaciones estructuradas: concepto, ejemplo, aplicación práctica y pregunta de comprobación.</p>
            <div class="badge-row">
                <span>💡 Explicación clara</span>
                <span>🧪 Ejemplo práctico</span>
                <span>🏭 Caso real</span>
                <span>✅ Verificación</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.subheader("⚙️ Configuración de conexión")
        st.caption(f"**Servidor:** {BASE_URL}")
        st.caption(f"**Modelo:** {MODEL}")
        st.divider()
        st.subheader("📊 Estadísticas")
        total_preguntas = sum(1 for m in st.session_state.historial if m["rol"] == "usuario")
        st.metric("Preguntas realizadas", total_preguntas)
        st.divider()
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.historial = []
            st.rerun()

    # ---------------- PREGUNTAS PREESTABLECIDAS ----------------
    if not st.session_state.historial:
        st.markdown("#### ⚡ Preguntas rápidas")
        st.caption("Haz clic para enviarlas directamente al agente.")
        columnas = st.columns(3)
        for i, item in enumerate(PREGUNTAS_PREESTABLECIDAS):
            with columnas[i % 3]:
                if st.button(f"{item['icono']} {item['texto']}", key=f"preset_{i}", use_container_width=True):
                    procesar_pregunta(item["texto"])
                    st.rerun()
        st.divider()

    # ---------------- HISTORIAL DE CONVERSACIÓN (estilo chat) ----------------
    if st.session_state.historial:
        for mensaje in st.session_state.historial:
            if mensaje["rol"] == "usuario":
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(mensaje["contenido"])
                    st.caption(mensaje["hora"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    if mensaje.get("es_error"):
                        st.error(mensaje["contenido"])
                    else:
                        st.markdown(mensaje["contenido"])
                        st.caption(mensaje["hora"])
    else:
        st.info("👋 Aún no hay conversación. Usa una pregunta rápida o escribe la tuya abajo para comenzar.")

    # Si hay una pregunta pendiente de responder, la procesamos y mostramos
    # la respuesta del agente justo debajo de la pregunta del usuario.
    if st.session_state.pendiente:
        pregunta = st.session_state.pendiente
        st.session_state.pendiente = None
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Pensando..."):
                respuesta, error = consultar_agente(pregunta)
            hora = datetime.now().strftime("%H:%M")
            if error:
                st.error(error)
                st.session_state.historial.append(
                    {"rol": "agente", "contenido": error, "hora": hora, "es_error": True}
                )
            else:
                st.markdown(respuesta)
                st.caption(hora)
                st.session_state.historial.append(
                    {"rol": "agente", "contenido": respuesta, "hora": hora, "es_error": False}
                )

    # ---------------- BARRA DE ENTRADA FIJA ABAJO ----------------
    pregunta_nueva = st.chat_input("Escribe tu consulta sobre tecnología, programación o IA...")
    if pregunta_nueva:
        procesar_pregunta(pregunta_nueva)
        st.rerun()


if __name__ == "__main__":
    main()
