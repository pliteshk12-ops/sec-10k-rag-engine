import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from rich.console import Console
from rich.panel import Panel

console = Console()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_financial_synthesis(question: str, persist_dir: str = "./chroma_db"):
    console.print(f"[bold cyan][*] Loading local vector database from '{persist_dir}'...[/bold cyan]")
    
    # 1. Initialize local embeddings & Chroma vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Load local offline synthesis model
    console.print(f"[yellow][*] Loading local offline synthesis model (google/flan-t5-base)...[/yellow]")
    model_id = "google/flan-t5-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    local_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False
    )
    
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    
    # 3. Define Prompt Template
    template = """Use the following pieces of SEC 10-K context to answer the financial question at the end. 
If you do not know the answer, state that the context lacks sufficient evidence. Keep the summary precise.

Context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    # 4. Modern LCEL Chain (Replaces RetrievalQA)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    console.print(f"\n[bold green][+] Executing query against SEC 10-K filing:[/bold green] [italic]{question}[/italic]\n")
    
    # Execute query
    result = rag_chain.invoke(question)
    
    # Retrieve retrieved documents separately for evidence auditing
    retrieved_docs = retriever.invoke(question)
    
    # 5. Display professional output using Rich
    console.print(Panel(result, title="[bold green]AI Financial Synthesis Output[/bold green]", expand=False))
    
    console.print("\n[bold underline]Evidence & Source Tracking:[/bold underline]")
    for i, doc in enumerate(retrieved_docs, 1):
        source_meta = doc.metadata.get('source', 'sample_10k.pdf')
        page_no = doc.metadata.get('page', 'N/A')
        console.print(f"[dim]• Evidence Chunk {i} | File: {source_meta} | Page: {page_no}[/dim]")

if __name__ == "__main__":
    financial_query = "What are the primary risk factors and revenue highlights?"
    run_financial_synthesis(financial_query)