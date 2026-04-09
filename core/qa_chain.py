"""RAG question-answering chain using LangChain LCEL and OpenAI GPT-4o Mini."""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_QA_PROMPT = PromptTemplate.from_template(
    """You are an expert legal document analyst. Use the following excerpts from a legal document to answer the question accurately and concisely.
If the answer is not found in the provided context, say "I couldn't find relevant information in the document for this question."

Context:
{context}

Question: {question}

Answer:"""
)

TOP_K = 4


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str, vector_store: FAISS, api_key: str) -> str:
    """
    Retrieve the most relevant chunks and answer the question via OpenAI GPT-4o Mini.

    Raises RuntimeError on chain or API failure.
    """
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
        )

        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _QA_PROMPT
            | llm
            | StrOutputParser()
        )

        return chain.invoke(question)

    except Exception as e:
        raise RuntimeError(f"Failed to generate answer: {e}") from e
