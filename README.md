# SEC 10-K Evidence-First RAG Pipeline

An offline, table-aware Retrieval-Augmented Generation (RAG) system designed to parse, embed, and query dense SEC 10-K financial filings with evidence-first source attribution and zero cloud dependencies.

## 🚀 Key Architectural Features
- **Smart Financial Chunking**: Preserves multi-row table context and financial disclosures using PyMuPDF and recursive character splitting.
- **Local Vector Persistence**: Stores embeddings locally using ChromaDB for high-speed, secure similarity search.
- **Evidence-First Retrieval**: Maps retrieved context directly to source file names and page-level metadata.
- **100% Offline Execution**: Runs entirely on local machine resources using HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) and local LLM synthesis (`flan-t5-base`).
- **Live Market Verification**: Integrates `pandas` and `yfinance` to pull live stock telemetry and financial metrics for cross-referencing.

## 🛠️ Tech Stack
- **Language**: Python 3.12
- **Orchestration**: LangChain (LCEL)
- **Vector Database**: ChromaDB
- **Embeddings & LLM**: HuggingFace Transformers, Sentence-Transformers
- **Data & Market Analysis**: Pandas, yfinance
- **CLI Interface**: Rich (Python TUI library)

## ⚙️ Quick Start
1. Clone the repository and set up a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate