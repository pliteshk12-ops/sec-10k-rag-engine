# SEC 10-K RAG Engine

A production-hardened, evidence-first Retrieval-Augmented Generation (RAG) system built to query and analyze complex SEC 10-K filings with strict numeric integrity, automated page-level citation tracking, and hybrid search capabilities.

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |     SEC EDGAR 10-K Filings       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  Ingestion & PyMuPDF Parser      |
               | (Metadata: Page Numbers, Source) |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Vector Indexing (ChromaDB)       |
               | Embeddings: all-MiniLM-L6-v2     |
               +----------------------------------+
                                |
           +--------------------+--------------------+
           |                                         |
           v                                         v
+-----------------------+                 +-----------------------+
| BM25 Keyword Search   |                 | Dense Vector Search   |
| (Alphanumeric Recall) |                 | (Semantic Context)    |
+-----------------------+                 +-----------------------+
           |                                         |
           +--------------------+--------------------+
                                |
                                v
               +----------------------------------+
               |    Hybrid Ensemble Retriever     |
               |      (Reciprocal Rank Fusion)     |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Answer Validation Guardrail      |
               | (Numeric Cross-Referencing)      |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Verified Answer + Page Citation  |
               +----------------------------------+