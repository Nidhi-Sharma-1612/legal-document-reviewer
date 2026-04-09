import os
import streamlit as st
from dotenv import load_dotenv

from core import (
    extract_text_from_pdf,
    create_vector_store,
    answer_question,
    summarize_document,
)

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Document Reviewer",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Document Reviewer")
st.markdown("Upload a legal PDF document to extract key information, ask questions, and generate summaries using AI.")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        placeholder="Enter your OpenAI API key...",
        help="Get your API key from https://platform.openai.com/api-keys",
    )
    api_key = api_key_input.strip() if api_key_input.strip() else env_key

    st.divider()
    st.subheader("Supported Files")
    st.markdown(
        "- **Text-based PDFs** (contracts, NDAs, terms of service)\n"
        "- Does **not** support scanned/image PDFs\n"
        "- File size: up to 200 MB"
    )
    st.divider()
    st.caption("Powered by LangChain + OpenAI GPT-4o Mini")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# ──────────────────────────────────────────────
# Section 1: Upload & Extract
# ──────────────────────────────────────────────

st.header("1. Upload Legal Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a text-based legal PDF (contracts, NDAs, service agreements, etc.)",
)

if uploaded_file is not None:
    is_new_file = (
        "last_filename" not in st.session_state
        or st.session_state.last_filename != uploaded_file.name
    )

    if is_new_file:
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            try:
                result = extract_text_from_pdf(uploaded_file)
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
                st.stop()

        for warning in result.page_warnings:
            st.warning(warning)

        st.session_state.raw_text = result.text
        st.session_state.last_filename = uploaded_file.name
        st.session_state.vector_store = None

        # Build vector index
        with st.spinner("Building vector index (this may take a moment)..."):
            try:
                vector_store = create_vector_store(result.text, api_key)
            except (ValueError, RuntimeError) as e:
                st.error(str(e))
                st.stop()

        st.session_state.vector_store = vector_store
        st.success(
            f"Document processed successfully! "
            f"Extracted {len(result.text):,} characters "
            f"({len(result.text.split())} words)."
        )

    if st.session_state.get("raw_text"):
        with st.expander("Preview extracted text (first 1,500 characters)", expanded=False):
            preview = st.session_state.raw_text[:1500]
            if len(st.session_state.raw_text) > 1500:
                preview += "..."
            st.text(preview)

# ──────────────────────────────────────────────
# Section 2: Question Answering
# ──────────────────────────────────────────────

st.header("2. Ask a Question")

if not st.session_state.get("vector_store"):
    st.info("Upload and process a document above to enable question answering.")
else:
    question = st.text_input(
        "Enter your question about the document:",
        placeholder='"What are the terms for termination?" or "What is the confidentiality duration?"',
    )

    if st.button("Get Answer", type="primary", disabled=not question.strip()):
        with st.spinner("Searching document and generating answer..."):
            try:
                answer = answer_question(question, st.session_state.vector_store, api_key)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.subheader("Answer")
        st.markdown(answer)

# ──────────────────────────────────────────────
# Section 3: Summarization
# ──────────────────────────────────────────────

st.header("3. Document Summary")

if not st.session_state.get("vector_store"):
    st.info("Upload and process a document above to generate a summary.")
else:
    if st.button("Generate Summary", type="secondary"):
        with st.spinner("Analyzing document and generating summary..."):
            try:
                summary = summarize_document(st.session_state.vector_store, api_key)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.subheader("Document Summary")
        st.markdown(summary)
