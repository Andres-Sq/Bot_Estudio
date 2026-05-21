import os
# Silenciar la advertencia de symlinks de HuggingFace en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def consultar_sistema_rag():
    persist_directory = "./chroma_db"
    
    # 1. Verificar si la base de datos existe
    if not os.path.exists(persist_directory):
        print(f"[Error] No se encontró la base de datos en '{persist_directory}'. Ejecuta primero 'procesar_pdf.py'.")
        return

    # 2. Cargar el motor de embeddings locales
    print(" Cargando motor de embeddings locales...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Conectarse a la base de datos ChromaDB existente
    print(" Conectando con la base de datos vectorial ChromaDB...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print("\n Base de datos conectada correctamente.")
    print("Escribe tu pregunta sobre el manual de Git (o escribe 'salir' para terminar):\n")

    # 4. Bucle interactivo de consulta
    while True:
        pregunta = input("👤 Tu pregunta: ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        
        if not pregunta.strip():
            continue

        print("\n🔍 Buscando los fragmentos más relevantes en el PDF...")
        # Buscamos los 3 fragmentos de texto más cercanos matemáticamente a la pregunta
        resultados = vector_store.similarity_search(pregunta, k=3)

        print(f"\n--- ENCONTRADOS {len(resultados)} FRAGMENTOS RELEVANTES ---")
        for i, doc in enumerate(resultados, 1):
            print(f"\n📄 [Fragmento {i}] - Página {doc.metadata.get('page', 'Desconocida')}:")
            print("-" * 50)
            print(doc.page_content.strip())
            print("-" * 50)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    consultar_sistema_rag()