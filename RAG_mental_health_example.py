from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, SeleniumURLLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler


# 1. Cargar documentos (PDFs + URLs)
pdf_dir = "/home/carbajal/Documents/SaludMental/docs_chatbot/"  # 📂 ruta al directorio con tus PDFs
loader_pdfs = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)

urls = [
    "https://www.who.int/news-room/questions-and-answers/item/stress",
    "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
]
loader_urls = SeleniumURLLoader(urls=urls)

docs = loader_pdfs.load() #+ loader_urls.load()
print(f" Se cargaron {len(docs)} documentos en total.")



# Dividir en chunks de 2048 caracteres con solapamiento de 200
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=100
)

# for i, doc in enumerate(docs):
#     print(f"{i} === Documento:", doc.metadata.get("source"), "===")
#     print(doc.page_content, "...")  # Muestra solo los primeros 500 caracteres
#     print("\n")

docs_split = text_splitter.split_documents(docs)



# 2. Crear embeddings con Ollama
# (asegurate de tener el modelo: ollama pull nomic-embed-text)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Guardar en base vectorial
#vectorstore = Chroma.from_documents(docs_split, embeddings, persist_directory="./chroma_db_1024") # primera vez
vectorstore = Chroma(embedding_function=embeddings,  persist_directory="./chroma_db_1024") # cuando ya se generaron los embeddings

# 4. Configurar el retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Definir memoria de conversación
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer", 
    k=5, # últimas 5 interacciones
)

# 6. Prompt personalizado
# Prompt con las variables esperadas
prompt_template = """Eres un asistente experto en salud mental de la OMS.
Usa el contexto recuperado para contestar la pregunta de manera clara, empática y precisa.
Si no sabes la respuesta, indica que no la encuentras en los documentos. Responde siempre en español y se breve.

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

# 7. Instanciar modelo Ollama (LLaMA 3)

# --- Handler para imprimir tokens a medida que llegan ---
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    disable_streaming=False,
    max_tokens=256  # máximo tamaño de la respuesta
    #callbacks=[StreamHandler()]  # <- imprime en vivo
)

# 8. Construir el pipeline RAG con memoria y prompt
rag_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
)


# 9. Modo chat interactivo
print("🤖 Chatbot de salud mental (OMS + PDFs). Escribe 'salir' para terminar.\n")

while True:
    query = input("Tú: ")
    if query.lower() in ["salir", "exit", "quit"]:
        print("👋 Adiós!")
        break
    
    result = rag_chain.invoke({"question": query})
    print("\n💡 Respuesta:", result["answer"], "\n")
    
    # Mostrar fuentes
    print("📚 Fuente(s):")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Desconocido"))
    print("\n" + "-"*50 + "\n")










