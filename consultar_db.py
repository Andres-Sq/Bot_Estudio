import os
# Silenciar la advertencia de symlinks de HuggingFace en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def consultar_sistema_rag():
    persist_directory = "./chroma_db"
    
    if not os.path.exists(persist_directory):
        print(f"[Error] No se encontró la base de datos en '{persist_directory}'. Ejecuta primero 'procesar_pdf.py'.")
        return

    print(" Cargando motor de embeddings locales...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(" Conectando con la base de datos vectorial ChromaDB...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print(" Inicializando cliente de Google Gemini...")
    ai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    print("\n Base de datos conectada. ¡Tu tutor de Estudios Sociales del MEP está listo!")
    print("Escribe tu pregunta sobre el temario de Bachillerato (o escribe 'salir' para terminar):\n")

    while True:
        pregunta = input("👤 Tu pregunta: ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego y éxitos en el estudio!")
            break
        
        if not pregunta.strip():
            continue

        print("\n🔍 Analizando los contenidos del MEP...")
        # Buscamos los 4 fragmentos más relevantes del temario
        resultados = vector_store.similarity_search(pregunta, k=4)

        contexto_pdf = "\n\n".join([f"[Página {doc.metadata.get('page')}]: {doc.page_content}" for doc in resultados])

        print("🤖 ProfeBot está redactando tu explicación explicativa...")
        
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

        try:
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_rag,
            )
            
            print("\n✨ RESPUESTA DEL TUTOR (MEP):")
            print("-" * 60)
            print(response.text.strip())
            print("-" * 60)
            
        except Exception as e:
            print(f"\n[Error con Gemini]: {e}")
            
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    consultar_sistema_rag()