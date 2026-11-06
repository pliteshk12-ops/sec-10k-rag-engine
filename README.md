# SEC 10-K Evidence-First RAG Pipeline

An offline, table-aware Retrieval-Augmented Generation (RAG) system designed to parse, embed, and query dense SEC 10-K financial filings with strict page-level source tracking and zero cloud dependencies.

## 🚀 Key Architectural Features
- **Smart Financial Chunking**: Preserves multi-row table context and financial disclosures using PyMuPDF and recursive character splitting.
- **Local Vector Persistence**: Stores embeddings locally using ChromaDB for high-speed, secure similarity search.
- **Evidence-First Retrieval**: Enforces strict citation tracking (file name and page number references) to eliminate financial hallucinations.
- **100% Offline Execution**: Runs entirely on local machine resources using HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) and local LLM synthesis (`flan-t5-base`).

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Orchestration**: LangChain (LCEL)
- **Vector Database**: ChromaDB
- **Embeddings & LLM**: HuggingFace Transformers, Sentence-Transformers
- **CLI Interface**: Rich (Python TUI library)

## ⚙️ Quick Start
1. Clone the repository and set up a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate