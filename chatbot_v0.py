import os
import torch
# Para PDF y TXT
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Para Word (DOCX)
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# Vectorstore
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore


from langchain.chains import RetrievalQA
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.prompts import PromptTemplate


from langchain_ollama import ChatOllama


prompt_template = """
Eres un asistente experto. Por favor responde en español y sigue estas reglas:
- Sé claro y conciso
- Mantén un tono formal pero cercano
- Siempre incluye ejemplos si es necesario
- No inventes información

Contexto del documento: {context}

Pregunta del usuario: {question}

Respuesta:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)


# ----------------------------
# 1. Cargar documentos
# ----------------------------
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import os

def load_documents(file_paths):
    all_docs = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Extensión no soportada: {ext}")
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

# Ejemplo
folder = "/home/carbajal/Documents/SaludMental/docs_chatbot/MSP/"
file_list = [os.path.join(folder,f) for f in os.listdir(folder)]
#file_list = ["documento1.pdf", "documento2.docx", "documento3.txt"]
docs= load_documents(file_list)


#docs = load_documents("/home/carbajal/Documents/SaludMental/docs_chatbot/MSP/UY_MSP_OF_DOC_Estrategia nacional de prevención de suicidio 2021 a 2025_2021.pdf")  # Cambiar al path de tu documento

# ----------------------------
# 2. Token-based chunking
# ----------------------------
from transformers import AutoTokenizer
tokenizer_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,          # Aproximadamente la mitad del max_length
    chunk_overlap=50,
    length_function=lambda text: len(tokenizer.encode(text))
)

texts = text_splitter.split_documents(docs)

print(f"Se generaron {len(texts)} sub-documentos.")

# ----------------------------
# 3. Generar embeddings
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=tokenizer_model,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

#vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore = InMemoryVectorStore.from_documents(texts, embeddings)
# ----------------------------
# 4. Preparar modelo de generación
# ----------------------------
#llm_model = "google/flan-t5-large"  # Podés cambiar a flan-t5-large para mayor max_length
#llm_model = "google/long-t5-tglobal-base"  # necesita torch 2.6 o superior
#llm_model = "google/flan-t5-xl" # no da la memoria para la GPU de la laptop
#tokenizer_llm = AutoTokenizer.from_pretrained(llm_model)
#model = AutoModelForSeq2SeqLM.from_pretrained(llm_model).to(
#    "cuda" if torch.cuda.is_available() else "cpu"
#)


llm_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer_llm = AutoTokenizer.from_pretrained(llm_model, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_model,
    device_map="auto",
    torch_dtype="auto",
    use_auth_token=True
)


#llm_model = ChatOllama(
#    model="llama3.1",
#    temperature=0,
    # other params...
#)

llm_pipeline = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer_llm,
    #device='cuda' if torch.cuda.is_available() else -1
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ----------------------------
# 5. Crear chain RAG
# ----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True
# )

# LLM ya definido previamente como llm
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # o "map_reduce" para documentos largos
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# ----------------------------
# 6. Ejecutar consulta
# ----------------------------
# query = "¿Cuál es la conclusión principal del documento?"
# result = rag_chain.invoke({"query": query})

# print("Respuesta:", result['result'])
# for doc in result['source_documents']:
#     print("Fragmento fuente:", doc.page_content[:200], "...\n")

# -------------------
# 7️⃣ Chat interactivo
# -------------------
print("📚 Sistema cargado. Escribí 'salir' para terminar.")
while True:
    query = input("\nTu pregunta: ")
    if query.lower() in ["salir", "exit"]:
        break
    result = rag_chain.invoke({"query": query})
    print("\n💡 Respuesta:\n", result['result'], '\n')

    for i, doc in enumerate(result['source_documents']):
        print(f"Fragmento fuente {i}: ", doc.page_content[:512], "...\n")
