import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rich.console import Console

console = Console()

def get_hybrid_retriever(persist_dir: str = "./chroma_db"):
    if not os.path.exists(persist_dir):
        console.print("[bold red][!] Vector database not found. Run ingest.py first.[/bold red]")
        return None

    console.print("[bold cyan][*] Initializing Hybrid Retriever (BM25 + Dense)...[/bold cyan]")
    
    # 1. Load Dense Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 2. Extract documents for BM25 sparse retriever
    # Fetch all stored documents from Chroma collection to build BM25 index
    all_data = vector_store.get()
    texts = all_data["documents"]
    metadatas = all_data["metadatas"]
    
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    
    if not docs:
        console.print("[bold red][!] No documents found in vector store to build BM25 index.[/bold red]")
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    # 3. Combine into Ensemble Retriever (weights: 50% BM25 keyword, 50% Dense semantic)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    
    console.print("[bold green][+] Hybrid Ensemble Retriever ready successfully![/bold green]")
    return ensemble_retriever

if __name__ == "__main__":
    retriever = get_hybrid_retriever()
    if retriever:
        query = "What are the primary risk factors?"
        results = retriever.invoke(query)
        console.print(f"\n[bold yellow]Hybrid Query Results for: '{query}'[/bold yellow]")
        for i, doc in enumerate(results, 1):
            console.print(f"{i}. [Page {doc.metadata.get('page')}] {doc.page_content[:150]}...")