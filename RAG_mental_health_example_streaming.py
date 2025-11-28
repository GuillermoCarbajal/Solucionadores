from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

# ----------------------------
# 1. Cargar PDFs desde directorio
# ----------------------------
pdf_dir = "/home/carbajal/Documents/SaludMental/docs_chatbot/"
loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)
docs = loader.load()
print(f"✅ Se cargaron {len(docs)} documentos.")

# ----------------------------
# 2. Dividir documentos en chunks (más pequeños para velocidad)
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,   # más pequeño → más rápido
    chunk_overlap=100
)
docs_split = text_splitter.split_documents(docs)

# ----------------------------
# 3. Crear embeddings con Ollama
# ----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ----------------------------
# 4. Guardar en vectorstore Chroma
# ----------------------------
vectorstore = Chroma.from_documents(docs_split, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # solo el más relevante para acelerar

# ----------------------------
# 5. Memoria de conversación
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ----------------------------
# 6. Prompt personalizado
# ----------------------------
prompt_template = """Eres un asistente experto en salud mental de la OMS.
Usa el contexto recuperado para contestar la pregunta de manera clara, empática y precisa.
Si no sabes la respuesta, indica que no la encuentras en los documentos. Responde únicamente la pregunta en español. No traduzcas, no agregues explicaciones intermedias.

Contexto:
{context}

Historial de conversación:
{chat_history}

Pregunta:
{question}

Respuesta:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template,
)

# ----------------------------
# 7. Handler para streaming
# ----------------------------
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# ----------------------------
# 8. LLM Ollama con streaming
# ----------------------------
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    disable_streaming=False,
    callbacks=[StreamHandler()]
)

# ----------------------------
# 9. Pipeline RAG interactivo
# ----------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# ----------------------------
# 10. Chat interactivo
# ----------------------------
print("🤖 Chatbot OMS + PDFs. Escribe 'salir' para terminar.\n")

while True:
    query = input("Tú: ")
    if query.lower() in ["salir", "exit", "quit"]:
        print("👋 Adiós!")
        break

    # Invocar RAG
    result = rag_chain.invoke({"question": query})

    print("\n💡 Respuesta generada:\n")
    
    # Mostrar fuentes
    print("📚 Fuente(s):")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Desconocido"))
    print("\n" + "-"*50 + "\n")
