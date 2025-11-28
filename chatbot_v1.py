import os
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# -------------------------------
# 1. Definir directorio de PDFs
# -------------------------------
pdf_dir = "/home/carbajal/Documents/SaludMental/docs_chatbot/"  # cambia esto al directorio que quieras

# -------------------------------
# 2. Cargar todos los PDFs
# -------------------------------
loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)
docs = loader.load()
# Mostrar los nombres de archivo cargados
#for i, doc in enumerate(docs):  # muestro los primeros 5
#    print(f"{i+1}. {doc.metadata['source']}")

# -------------------------------
# 3. Partir documentos en chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# -------------------------------
# 4. Crear embeddings y vectorstore
# -------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # persistencia local
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 5. Instanciar modelo LLaMA-3.1 en Ollama
# -------------------------------
llm = ChatOllama(model="llama3", temperature=0)

# -------------------------------
# 6. Prompt para RAG
# -------------------------------
prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto. Por favor responde en español y sigue estas reglas:
- Sé claro y conciso
- Mantén un tono formal pero cercano
- Siempre incluye ejemplos si es necesario
- No inventes información

Contexto:
{context}

Pregunta:
{question}
""")

# -------------------------------
# 7. Pipeline de RAG
# -------------------------------
def rag_pipeline(question: str):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    return response.content

# -------------------------------
# 8. Loop interactivo de chat
# -------------------------------
chat_history = []

print("💬 Chatbot listo. Escribe 'salir' para terminar.\n")

while True:
    query = input("❓ Pregunta: ")
    if query.lower() in ["salir", "exit", "quit"]:
        break

    # Obtenemos los documentos relevantes
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    formatted_prompt = prompt.format(context=context, question=query)

    # Llamamos al LLM
    response = llm.invoke(formatted_prompt)
    answer = response.content

    # Mostramos respuesta
    print("\n💡 Respuesta:\n", answer, "\n")

    # Guardamos historial
    chat_history.append((query, answer))

    # Mostramos documentos fuente
    print("📚 Documentos usados:")
    for doc in docs:
        print(" -", doc.metadata.get("source", "desconocido"))
    print("\n" + "-"*40 + "\n")
