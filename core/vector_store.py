"""Text chunking, embedding, and FAISS vector store creation."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_vector_store(text: str, api_key: str) -> FAISS:
    """
    Split text into chunks, embed them with OpenAI, and return a FAISS index.

    Raises ValueError if the text produces no chunks.
    Raises RuntimeError for embedding or indexing failures.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("No text chunks were created. The document may be empty.")

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        return FAISS.from_texts(chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to build vector index: {e}") from e
