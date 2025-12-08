# Business Service RAG Demo

This project is a small end‑to‑end Retrieval-Augmented Generation (RAG) pipeline focused on **InterSystems Business Services** documentation.

It:

1. **Chunks and embeds documentation** using OpenAI embeddings (`embed.py`).
2. **Uploads embeddings to InterSystems IRIS** as a vector table (`upload_embeddings.py`).
3. **Searches the vector table** for relevant chunks (`search.py`).
4. **Provides an interactive chat** that uses OpenAI’s Responses API and your vector search as a tool (`business_service_chat.py`).

---

## High‑Level Architecture

1. **Source docs**: `.txt` / `.md` files in `Documentation/BusinessService`.
2. **Chunking & embedding** (`embed.py`):
   - Split docs into overlapping chunks (paragraph/sentence-aware).
   - Save chunks to `BusinessService.parquet`.
   - Generate embeddings with `text-embedding-3-small`.
   - Save embedded chunks to `BusinessService_embedded.parquet`.
3. **Vector DB (InterSystems IRIS)** (`upload_embeddings.py`):
   - Create a vector table (`AIDemo.Embeddings`).
   - Upload `chunk_text` + `embedding` as `VECTOR(FLOAT, 1536)`.
4. **Search API** (`search.py`):
   - Given a query, generate its embedding.
   - Run a similarity search using `VECTOR_DOT_PRODUCT` and `TO_VECTOR`.
   - Return top‑k matching chunks.
5. **RAG Chat** (`business_service_chat.py`):
   - CLI chat using OpenAI Responses API.
   - Exposes `search_embeddings` as a tool (`search_business_docs`).
   - Model decides when to call the search tool to answer business-service questions.

---

## Repository Structure

Assuming a flat structure:

- `embed.py`
- `upload_embeddings.py`
- `search.py`
- `business_service_chat.py`
- `Documentation/BusinessService/`  
  (your `.txt` / `.md` docs go here)
- `venv/dev.env`  
  (your environment variables – notably `OPENAI_API_KEY`)

---

## Prerequisites

### 1. Python & Packages

- **Python**: 3.10+ recommended
- Install the following (adjust as needed):

```bash
pip install \
  openai \
  python-dotenv \
  pandas \
  pyarrow \
  tiktoken \
  langchain-text-splitters \
  numpy \
  iris
```

> Note: `pyarrow` (or `fastparquet`) is needed for Parquet support; `pandas` will use one if available.

### 2. OpenAI Account & API Key

- You need an OpenAI API key with access to:
  - `text-embedding-3-small`
  - `gpt-5-nano` (or your preferred Responses-capable model)

Set it either as a regular environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

or in `.env` (see below).

### 3. InterSystems IRIS with Vector Support

You need an IRIS instance with:

- A namespace (in code: `VECTOR`)
- The ability to:
  - Create tables
  - Use `VECTOR` column types
  - Use `VECTOR_DOT_PRODUCT` and `TO_VECTOR` functions

The defaults in code:

- `IRIS_HOST = "localhost"`
- `IRIS_PORT = 8881`
- `IRIS_NAMESPACE = "VECTOR"`
- `IRIS_USERNAME = "superuser"`
- `IRIS_PASSWORD = "sys"`
- `TABLE_NAME = "AIDemo.Embeddings"`
- `EMBEDDING_DIMENSIONS = 1536` (matches `text-embedding-3-small`)

Adjust these in the scripts if needed.

---

## Environment Configuration

The scripts assume a `.env` file at:

- `venv/dev.env`

At a minimum, put your OpenAI key there:

```env
OPENAI_API_KEY=sk-...
# Optional: override default chat model
# OPENAI_RESPONSES_MODEL=gpt-5-nano
```

Both `embed.py` and `search.py` load it via:

```python
from dotenv import load_dotenv
load_dotenv(dotenv_path="venv/dev.env")
```

If you prefer not to use `.env`, just export `OPENAI_API_KEY` directly in your shell.

---

## Step 1: Prepare Documentation

Place your Business Service documentation as `.txt` or `.md` files in:

```text
Documentation/BusinessService/
  ├── intro.md
  ├── settings.md
  ├── examples.txt
  └── ...