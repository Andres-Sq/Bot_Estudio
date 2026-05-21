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
        
        # Prompt super-tutor: Temario + Desarrollo de materia (Limpio y Corregido)
        prompt_rag = f"""
        Eres "ProfeBot", un profesor y tutor de Estudios Sociales de Costa Rica, apasionado, dinámico y experto en preparar estudiantes para Bachillerato del MEP.
        Tu objetivo es explicar la materia que el estudiante te pida basándote en los objetivos del temario proporcionado.

        REGLAS DE RESPUESTA:
        1. Identifica qué temas y objetivos menciona el "CONTEXTO DEL TEMARIO" abajo sobre la pregunta del alumno.
        2. ¡DESARROLLA LA MATERIA!: No te limites a listar los temas. Explica la materia a fondo de forma educativa, estructurada y fácil de entender. Usa viñetas, definiciones claras y ejemplos costarricenses.
        3. Si el temario solo menciona "Relieve de Costa Rica", tú debes explicar cuáles son esas cordilleras, sus características principales y datos clave que el MEP suele preguntar en los exámenes de Bachillerato.

        CONTEXTO DEL TEMARIO:
        {contexto_pdf}

        PREGUNTA DEL ESTUDIANTE:
        {pregunta}
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