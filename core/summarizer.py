"""Document summarization chain using LangChain LCEL and OpenAI GPT-4o Mini."""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

_SUMMARY_QUERY = (
    "key terms obligations parties rights termination payment confidentiality"
)

_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are an expert legal document analyst. Based on the following excerpts from a legal document, provide a concise and structured summary.

The summary should cover:
- Type of document and parties involved
- Key obligations and rights of each party
- Important terms, conditions, and restrictions
- Termination clauses (if present)
- Payment or compensation terms (if present)
- Confidentiality or non-disclosure terms (if present)
- Any notable risks or red flags

Document excerpts:
{context}

Provide the summary in clear, plain English suitable for a legal professional:"""
)

TOP_K = 6


def summarize_document(vector_store: FAISS, api_key: str) -> str:
    """
    Retrieve the most representative chunks and generate a structured legal summary.

    Raises RuntimeError on chain or API failure.
    """
    try:
        docs = vector_store.similarity_search(_SUMMARY_QUERY, k=TOP_K)
        context = "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3,
        )

        chain = _SUMMARY_PROMPT | llm | StrOutputParser()
        return chain.invoke({"context": context})

    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {e}") from e
