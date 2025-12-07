# app/vectorstore/load_vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_vectorstore(persist_dir, model_name="Qwen/Qwen3-Embedding-4B"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
