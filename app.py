import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Silenciar advertencia de symlinks en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
load_dotenv()

# --- CONFIGURACIÓN DE LA PÁGINA WEB ---
st.set_page_config(
    page_title="ProfeBot - Tutor MEP",
    page_icon="🤖",
    layout="centered"
)

# --- INICIALIZACIÓN DE COMPONENTES ---
@st.cache_resource
def inicializar_sistema():
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        return None, None, None
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return vector_store, ai_client

vector_store, ai_client = inicializar_sistema()

# --- DISEÑO DE LA INTERFAZ ---
st.title("🤖 ProfeBot: Tu Tutor MEP")
st.subheader("Preparación integral para tus materias de bachillerato")
st.write("¡Pura vida! Estoy listo para ayudarte con Biología, Cívica, Español, Estudios Sociales, Física, Francés, Inglés, Matemáticas y Química.")
st.divider()

# Verificar que la base de datos exista
if vector_store is None:
    st.error("❌ No se encontró la base de datos en './chroma_db'. Por favor, ejecuta primero 'procesar_pdf.py' en tu terminal.")
    st.stop()

# --- MEMORIA DEL CHAT (Session State) ---
if "historial" not in st.session_state:
    st.session_state.historial = [
        {"role": "assistant", "content": "¡Hola, futuro bachiller! 🇨🇷 ¿Qué materia o tema vamos a repasar hoy?"}
    ]

# Mostrar los mensajes anteriores en la pantalla
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# --- LÓGICA DE ENTRADA DEL USUARIO ---
if pregunta := st.chat_input("Escribe tu duda aquí..."):
    
    # 1. Mostrar la pregunta del usuario en la pantalla
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.historial.append({"role": "user", "content": pregunta})

    # 2. Generar la respuesta con el motor RAG
    with st.chat_message("assistant"):
        contenedor_respuesta = st.empty()
        
        with st.spinner("🔍 Analizando el temario y preparando la clase..."):
            try:
                # Buscar en ChromaDB
                resultados = vector_store.similarity_search(pregunta, k=4)
                contexto_pdf = "\n\n".join([f"[Página {doc.metadata.get('page')}]: {doc.page_content}" for doc in resultados])
                
                # Prompt multi-tutor dinámico
                prompt_rag = f"""
                Eres "ProfeBot", un tutor experto, apasionado y dinámico en el sistema educativo del MEP de Costa Rica.
                Tu base de conocimientos incluye: Biología, Educación Cívica, Español, Estudios Sociales, Física, Francés, Inglés, Matemáticas y Química.

                TU MISIÓN:
                1. Analiza el "CONTEXTO DEL TEMARIO" proporcionado y la "PREGUNTA DEL ESTUDIANTE".
                2. Determina a qué materia pertenece la consulta.
                3. Explica la materia a fondo, de forma educativa, estructurada y fácil de entender. Usa viñetas, definiciones claras y ejemplos relevantes.
                4. Si el estudiante te pide algo muy general, tú debes desarrollar los conceptos clave y datos técnicos que el MEP suele evaluar en esas áreas.

                CONTEXTO DEL TEMARIO (Contiene información de diversas materias):
                {contexto_pdf}

                PREGUNTA DEL ESTUDIANTE:
                {pregunta}
                
                INSTRUCCIÓN FINAL: Saluda de forma personalizada según la materia y mantén un tono de apoyo constante hacia el estudiante.
                """
                
                # Consultar a Gemini
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_rag,
                )
                
                respuesta_final = response.text.strip()
                
                # Dibujar la respuesta en pantalla
                contenedor_respuesta.markdown(respuesta_final)
                st.session_state.historial.append({"role": "assistant", "content": respuesta_final})
                
            except Exception as e:
                contenedor_respuesta.error(f"Hubo un error al conectar con Gemini: {e}")