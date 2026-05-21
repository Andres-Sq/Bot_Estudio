import os
# Silenciar la advertencia de symlinks de HuggingFace en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from google import genai  # <- Nuevo SDK de Google
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

    # Inicializamos el cliente moderno de Gemini para el texto
    print(" Inicializando cliente de Google Gemini...")
    ai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    print("\n Base de datos conectada y Gemini listo.")
    print("Escribe tu pregunta sobre el manual de Git (o escribe 'salir' para terminar):\n")

    while True:
        pregunta = input("👤 Tu pregunta: ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        if not pregunta.strip():
            continue

        print("\n🔍 Buscando fragmentos en el PDF...")
        resultados = vector_store.similarity_search(pregunta, k=3)

        # Unimos los fragmentos encontrados para dárselos como contexto a la IA
        contexto_pdf = "\n\n".join([f"[Página {doc.metadata.get('page')}]: {doc.page_content}" for doc in resultados])

        print("🤖 Gemini redactando respuesta basada en tu manual...")
        
        # Estructuramos el Prompt para obligar a Gemini a usar SOLO tu PDF
        prompt_rag = f"""
        Eres un asistente experto en Git. Responde la pregunta del usuario utilizando ÚNICAMENTE el contexto extraído del manual proporcionado abajo. 
        Si la respuesta no se encuentra en el contexto, di amigablemente que esa información no viene en el manual.

        CONTEXTO DEL MANUAL:
        {contexto_pdf}

        PREGUNTA DEL USUARIO:
        {pregunta}
        """

        try:
            # Llamamos a gemini-2.5-flash (disponible e ideal para el tier gratuito)
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_rag,
            )
            
            print("\n✨ RESPUESTA DEL ASISTENTE:")
            print("-" * 60)
            print(response.text.strip())
            print("-" * 60)
            
        except Exception as e:
            print(f"\n[Error con Gemini]: {e}")
            
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    consultar_sistema_rag()