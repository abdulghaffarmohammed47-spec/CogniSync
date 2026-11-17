# CogniSync

An AI-powered Retrieval-Augmented Generation (RAG) document intelligence platform. Upload your files, retrieve relevant passages, and ask questions with answers grounded in your own sources.

## Features

- Ingest PDF, DOCX, TXT, Markdown, and CSV documents.
- Extract and chunk content into an in-memory knowledge base.
- Retrieve passages with local BM25 keyword search, or switch to OpenAI embeddings for semantic retrieval.
- Generate source-grounded answers with OpenAI; citations refer to the displayed retrieved passages.
- Review and remove indexed files in a focused Streamlit workspace.
- Run local retrieval without an API key. Documents and chat history are held only in the current app session.

## Run locally

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Add documents from the sidebar, then ask a question in the chat.

## Enable semantic search and AI answers

Provide an OpenAI API key in the sidebar, or set `OPENAI_API_KEY` in your environment before starting the app. Existing documents are re-embedded when semantic search is enabled. The app uses `text-embedding-3-small` and `gpt-4o-mini` by default.

When configured, document chunks are sent to the OpenAI embeddings API and retrieved excerpts are sent to the chat completions API. Do not upload documents you are not permitted to share with that provider. The API key is not stored by the app.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Project layout

- `app.py` — Streamlit interface and session workflow.
- `rag_engine.py` — document parsing, chunking, BM25/vector retrieval, and OpenAI API client.
- `tests/` — focused tests for parsing, chunking, and retrieval behavior.
