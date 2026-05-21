import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar la API Key desde el archivo .env
load_dotenv()

# Define la version a utilizar para el mdoelo 'gemini-2.5-flash'
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Hacer una pregunta de prueba
try:
    respuesta = llm.invoke("Hola, estoy configurando un proyecto RAG contigo. ¿Puedes oírme?")
    print("\n--- ¡Conexión Exitosa con Google AI Studio! ---")
    print(respuesta.content)
    print("---------------------------------------------\n")
except Exception as e:
    print(f"\nHubo un error en la configuración: {e}\n")