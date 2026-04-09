# Legal Document Reviewer

A Streamlit web application that uses LangChain and OpenAI to help legal professionals review, query, and summarize legal PDF documents.

## Live Demo

**[Launch the app](https://legal-document-reviewer-22c4dymz4vkh8mtndkchng.streamlit.app/)** — hosted on Streamlit Community Cloud. No installation required; bring your own OpenAI API key.

## Features

- **PDF Upload & Text Extraction** — Extract text from legal PDFs using PyPDF2
- **RAG Question Answering** — Ask natural language questions about the document; get answers backed by relevant excerpts
- **Document Summarization** — Generate a concise legal summary covering obligations, rights, key clauses, and risks
- **Session State Management** — Vector store persists across reruns within a session
- **Error Handling** — Graceful handling of encrypted, scanned, or corrupted PDFs

## Tech Stack

| Component    | Library                            |
| ------------ | ---------------------------------- |
| Frontend     | Streamlit                          |
| LLM          | OpenAI GPT-4o Mini (via LangChain) |
| Embeddings   | OpenAI `text-embedding-3-small`    |
| Vector Store | FAISS (in-memory)                  |
| PDF Parsing  | PyPDF2                             |
| RAG Pipeline | LangChain LCEL                     |

## Prerequisites

- Python 3.9 or higher
- An **OpenAI API key** — get one at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## Local Setup

1. **Clone or download this project**

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key** (optional — you can also enter it in the sidebar)

   ```bash
   cp .env.example .env
   # Edit .env and add your actual OPENAI_API_KEY
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

## Usage

1. Enter your **OpenAI API key** in the left sidebar (if not set via `.env`)
2. **Upload a PDF** in Section 1 — the app extracts text and builds a vector index
3. **Ask questions** in Section 2, e.g.:
   - _"What are the terms for termination?"_
   - _"What is the duration of the confidentiality obligation?"_
   - _"Who are the parties in this agreement?"_
4. Click **Generate Summary** in Section 3 for a structured overview of the document

## Notes

- Only **text-based PDFs** are supported. Scanned (image-based) PDFs cannot be processed.
- The API key entered in the sidebar is never stored persistently — it only lives in the current session.
- For large documents, indexing may take 10–30 seconds depending on document size.
