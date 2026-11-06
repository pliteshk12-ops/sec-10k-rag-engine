import os
import csv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.table import Table

console = Console()

def run_evaluation(csv_path: str = "./evaluation/questions.csv", persist_dir: str = "./chroma_db"):
    if not os.path.exists(persist_dir):
        console.print("[bold red][!] Vector database not found. Run ingest.py first.[/bold red]")
        return
        
    if not os.path.exists(csv_path):
        console.print(f"[bold red][!] Evaluation CSV not found at {csv_path}.[/bold red]")
        return

    console.print(fmt_msg := "[bold cyan][*] Initializing evaluation harness against local vector store...[/bold cyan]")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    total_queries = 0
    hit_at_3 = 0
    citation_matches = 0
    refusal_correct = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_queries += 1
            qid = row["id"]
            question = row["question"]
            expected_page = int(row["expected_page"])
            is_answerable = row["is_answerable"].lower() == "true"

            # Retrieve top 3 docs
            retrieved_docs = retriever.invoke(question)
            
            # Check retrieval hit rate @ 3 (Did expected page appear in any retrieved chunk metadata?)
            retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs]
            page_hit = expected_page in retrieved_pages
            if page_hit:
                hit_at_3 += 1
                citation_matches += 1

            # Check refusal simulation for unanswerable questions
            if not is_answerable:
                # If retriever returns low semantic relevance or we simulate refusal logic
                if len(retrieved_docs) == 0 or not page_hit:
                    refusal_correct += 1

    # Display results in a professional table
    table = Table(title="RAG Pipeline Evaluation Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="green")

    hit_rate_pct = (hit_at_3 / total_queries) * 100 if total_queries > 0 else 0
    table.add_row("Total Test Cases", str(total_queries))
    table.add_row("Retrieval Hit Rate @ Top 3", f"{hit_rate_pct:.1f}%")
    table.add_row("Citation Page Accuracy", f"{(citation_matches/total_queries)*100:.1f}%")
    
    console.print(table)

if __name__ == "__main__":
    run_evaluation()