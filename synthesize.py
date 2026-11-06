import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from rich.console import Console

console = Console()

def extract_numbers(text: str) -> set:
    # Extract digits, currency figures, and percentages from text
    return set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text))

def validate_answer_evidence(answer: str, retrieved_chunks: list) -> bool:
    """
    Ensures numeric integrity: every number in the answer must appear 
    in at least one of the retrieved source chunks.
    """
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return True # No numbers to verify
        
    combined_source_text = " ".join([doc.page_content for doc in retrieved_chunks])
    source_numbers = extract_numbers(combined_source_text)
    
    # Check if all numbers in the answer exist in the source text
    unverified_numbers = answer_numbers - source_numbers
    if unverified_numbers:
        console.print(f"[bold red][!] Unverified numbers detected in output: {unverified_numbers}[/bold red]")
        return False
        
    return True

def run_synthesis(query: str):
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        console.print("[bold red][!] Vector database not found. Run ingest.py first.[/bold red]")
        return

    console.print(f"[bold cyan][*] Querying SEC 10-K database for: '{query}'[/bold cyan]")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    docs = retriever.invoke(query)
    
    # Simulated generation output or local model response
    # For demonstration with flan-t5-base / local synthesis:
    synthesized_answer = "The filing reports significant operational metrics and revenue highlights."
    
    # Run the validation guardrail check
    is_valid = validate_answer_evidence(synthesized_answer, docs)
    
    if not is_valid:
        synthesized_answer = "Not enough evidence or unverified numeric claims detected."

    console.print(f"\n[bold green]Final Verified Output:[/bold green] {synthesized_answer}")
    console.print("\n[bold cyan]Source Citations:[/bold cyan]")
    for doc in docs:
        console.print(f"• File: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")

if __name__ == "__main__":
    run_synthesis("What are the primary risk factors and revenue highlights?")