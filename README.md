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
               ================================================================================
Query: "What are the primary risk factors and revenue highlights?"
--------------------------------------------------------------------------------
[Retrieved Context Chunk 1 (Page 71)]:
"Disclose any future changes in the relationship between costs and revenue if events 
are known that are reasonably likely to cause a material change..."

Synthesized Answer: 
"Based on the SEC filing disclosures, the primary risk factors relate to operational 
expense volatility and forward-looking material changes in cost structures..."

Evidence Citations:
  • File: sample_10k.pdf | Page Number: 71
  • Validation Check: Passed (100% Numeric & Fact Integrity Verified)
================================================================================