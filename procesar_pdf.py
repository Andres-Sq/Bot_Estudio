import os
# Silenciar la advertencia de symlinks de HuggingFace en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma # Usamos la versión moderna langchain_chroma

load_dotenv()

def procesar_todos_los_pdfs(carpeta):
    todos_los_chunks = []
    
    # Lista todos los archivos PDF en la carpeta
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith(".pdf")]
    
    if not archivos_pdf:
        print(f"No se encontraron archivos PDF en '{carpeta}'.")
        return

    for archivo in archivos_pdf:
        ruta_completa = os.path.join(carpeta, archivo)
        print(f"\n--- Procesando: {archivo} ---")
        
        loader = PyPDFLoader(ruta_completa)
        paginas = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(paginas)
        
        # Añadir metadato para saber de qué materia viene
        for chunk in chunks:
            chunk.metadata["materia"] = archivo.replace(".pdf", "")
            
        todos_los_chunks.extend(chunks)
        print(f"Fragmentos de '{archivo}': {len(chunks)}")

    print(f"\nTotal de fragmentos acumulados: {len(todos_los_chunks)}")
    
    print(" Generando embeddings y guardando en ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    persist_directory = "./chroma_db"
    
    # Guardar todo junto
    Chroma.from_documents(
        documents=todos_los_chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("\n--- ¡Procesamiento Masivo Completado con Éxito! ---")
    print(f"Tu base de datos ahora contiene todas tus materias en: {persist_directory}")

if __name__ == "__main__":
    carpeta_documentos = "documentos"
    
    if os.path.exists(carpeta_documentos):
        procesar_todos_los_pdfs(carpeta_documentos)
    else:
        print(f"\n[Aviso] La carpeta '{carpeta_documentos}' no existe. Por favor créala y coloca ahí tus 9 PDFs.")