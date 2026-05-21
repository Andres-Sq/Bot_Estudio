import os
# Silenciar la advertencia de symlinks de HuggingFace en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <- Importación moderna
from langchain_community.vectorstores import Chroma

load_dotenv()

def procesar_documento_pdf(ruta_pdf):
    print(f" Cargando el archivo: {ruta_pdf}...")
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()
    
    print(" Dividiendo el texto en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(paginas)
    print(f" Generados {len(chunks)} fragmentos de texto.")
    
    print(" Conectando con HuggingFace Embeddings locales (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(" Guardando fragmentos en la base de datos vectorial (ChromaDB)...")
    persist_directory = "./chroma_db"
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("\n--- ¡Procesamiento Completado con Éxito! ---")
    print(f"Tu base de datos vectorial se guardó en: {persist_directory}")
    return vector_store

if __name__ == "__main__":
    archivo_prueba = "documentos/manual_git.pdf" 
    
    if os.path.exists(archivo_prueba):
        procesar_documento_pdf(archivo_prueba)
    else:
        print(f"\n[Aviso] No encontré el archivo en '{archivo_prueba}'. Verifica el nombre de la carpeta y del archivo.")