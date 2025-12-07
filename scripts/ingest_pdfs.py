"""Ingest PDFs, add metadata 'filename' and 'intent' (heuristic), split using semantic chunking, and build FAISS index."""
import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract"
def guess_intent_from_filename(filename: str):
    """
    Heuristic to guess document intent from filename.
    """
    name = filename.lower()
    if 'hr' in name:
        return 'HR_Policy'
    if 'it' in name:
        return 'IT_guidelines'
    # fallback: place in HR by default (change as needed)
    return 'HR_Policy'


def ingest_folder(
    folder_path: str, 
    persist_path: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-4B",
    breakpoint_threshold_type: str = "percentile"
):
    """
    Ingest documents from folder, chunk them semantically, and create FAISS vector store.
    
    Args:
        folder_path: Path to folder containing documents (e.g., 'data/references')
        persist_path: Where to save the FAISS index (e.g., 'data/vector_db')
        embedding_model: HuggingFace embedding model name
        breakpoint_threshold_type: Threshold type for semantic chunking
    
    Returns:
        FAISS vectorstore object
    """
    
    # Step 1: Load all documents
    print(f"📂 Loading documents from: {folder_path}")
    all_docs = []
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    for file in os.listdir(folder_path):
        file_lower = file.lower()
        
        if file_lower.endswith(('.pdf', '.md', '.txt')):
            path = os.path.join(folder_path, file)
            print(f"📄 Loading: {path}")
            
            # Load PDF with hi_res strategy
            if file_lower.endswith('.pdf'):
                loader = UnstructuredFileLoader(
                    path, 
                    unstructured_kwargs={'strategy': 'hi_res'}
                )
                docs = loader.load()
            
            # Load markdown/text files
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs = [Document(page_content=content, metadata={})]
            
            # Add metadata to each document
            for d in docs:
                d.metadata['filename'] = file.replace('.pdf', '').replace('.md', '').replace('.txt', '')
                d.metadata['source'] = path  # Full path for reference
                d.metadata['intent'] = guess_intent_from_filename(file)
            
            all_docs.extend(docs)
    
    if not all_docs:
        raise ValueError(f"No documents found in {folder_path}")
    
    print(f"✅ Loaded {len(all_docs)} documents")
    
    # Step 2: Initialize embeddings
    print(f"🔧 Initializing embeddings: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Step 3: Semantic chunking
    print(f"✂️  Using SemanticChunker with breakpoint_threshold_type='{breakpoint_threshold_type}'")
    semantic_chunker = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type=breakpoint_threshold_type
    )
    
    # Create chunks while preserving metadata
    chunks = []
    for doc in all_docs:
        # Split the document content semantically
        doc_chunks = semantic_chunker.create_documents([doc.page_content])
        
        # Add original metadata to each chunk
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)
        
        chunks.extend(doc_chunks)
    
    print(f"✅ Created {len(chunks)} semantic chunks")
    
    # Step 4: Create FAISS vector store
    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Step 5: Save to disk
    os.makedirs(persist_path, exist_ok=True)
    vectorstore.save_local(persist_path)
    print(f"💾 Saved vectorstore to {persist_path}")
    print(f"📊 Total chunks indexed: {len(chunks)}")
    
    return vectorstore


if __name__ == '__main__':
    # Default paths for standalone execution
    folder_path = "data/references"
    persist_path = "data/vector_db"
    
    ingest_folder(
        folder_path=folder_path,
        persist_path=persist_path
    )