# 🛰️ MOSDAC AI Helpbot – NLP & RAG Pipeline (Member B)

This module implements the NLP preprocessing, embedding generation, vector similarity search, and the Retrieval-Augmented Generation (RAG) pipeline using a free open-source LLM.

---

## ✅ Responsibilities

- Clean and chunk scraped documents
- Generate embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- Store and retrieve documents using ChromaDB
- Build a RAG pipeline:
  - Query understanding
  - Vector similarity search
  - Knowledge graph fallback
  - Prompt creation
  - LLM-based response generation using `google/flan-t5-base`

---

## 🗂️ Key Files

- `utils/text_utils.py` – Text cleaning & chunking
- `src/generate_embeddings.py` – Convert documents to vector embeddings
- `src/query.py` – Full RAG pipeline logic
- `llm/llm_client.py` – Flan-T5-based response generation
- `llm/prompt_templates.py` – RAG prompt format
- `utils/kg_utils.py` – Dummy knowledge graph search
- `tests/` – Unit tests for core components

---

## 🧪 How to Test

Run unit tests:

```bash
pytest
