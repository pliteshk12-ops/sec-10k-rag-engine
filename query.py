import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def query_sec_filing(question: str, persist_dir: str = "./chroma_db"):
    print(f"[*] Loading local vector database from {persist_dir}...")
    
    # Initialize the same local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the persisted Chroma database
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    print(f"[*] Searching filing for: '{question}'\n" + "="*50)
    
    # Retrieve top 3 most relevant financial chunks
    docs = vector_store.similarity_search(question, k=3)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n[Evidence Match {i}]")
        print(f"Content Snippet:\n{doc.page_content[:400]}...\n")
        print("-" * 50)

if __name__ == "__main__":
    # Test query targeting financial performance or risk factors
    sample_question = "What are the primary risk factors and total revenue highlights?"
    query_sec_filing(sample_question)