import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def process_sec_filing(pdf_path: str, persist_dir: str = "./chroma_db"):
    print(f"[*] Ingesting SEC Filing: {pdf_path}")
    
    # Load PDF using PyMuPDF for speed and accuracy
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Financial text requires tight chunking to preserve multi-row table context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[+] Generated {len(chunks)} financial text chunks.")
    
    # Local embedding model (Runs completely offline on your machine)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store locally in ChromaDB (Zero external cloud dependency)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vector_store.persist()
    print(f"[+] Vector database successfully persisted at {persist_dir}")

if __name__ == "__main__":
    # Point this to your downloaded SEC 10-K PDF (e.g., Apple, Tesla, or Microsoft 10-K)
    sample_pdf = "sample_10k.pdf" 
    if os.path.exists(sample_pdf):
        process_sec_filing(sample_pdf)
    else:
        print(f"[-] Error: Place a valid SEC 10-K PDF named '{sample_pdf}' in your root directory.")
        