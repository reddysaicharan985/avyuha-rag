import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from monitoring import log_rag_query, save_feedback
st.set_page_config(
    page_title="AVYUHA RAG Assistant",
    page_icon="A",
    layout="centered"
)

base_directory = Path(__file__).parent
database_path = base_directory / "chroma_db"

load_dotenv(base_directory / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY was not found.")
    st.stop()


@st.cache_resource
def load_rag_components():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name="avyuha_knowledge",
        persist_directory=str(database_path),
        embedding_function=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    return vector_store, llm


vector_store, llm = load_rag_components()

st.title("AVYUHA RAG Assistant")
st.caption(
    "Ask questions about AVYUHA, MutualDreamers, "
    "Random Confession and AVYUHA SaaS."
)

st.info(
    "This assistant answers only from the AVYUHA "
    "RAG practice knowledge base."
)

def display_feedback_buttons(log_id):
    """Display feedback buttons for one monitored answer."""

    st.caption("Was this answer helpful?")

    helpful_column, unhelpful_column, empty_column = st.columns(
        [1, 1, 4]
    )

    with helpful_column:
        if st.button(
            "👍 Helpful",
            key=f"helpful_{log_id}"
        ):
            save_feedback(log_id, "helpful")
            st.success("Helpful feedback saved.")

    with unhelpful_column:
        if st.button(
            "👎 Not helpful",
            key=f"unhelpful_{log_id}"
        ):
            save_feedback(log_id, "unhelpful")
            st.warning("Feedback saved for improvement.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("log_id")
        ):
            display_feedback_buttons(
                message["log_id"]
            )

question = st.chat_input("Ask a question about AVYUHA")

if question:
    request_started = time.perf_counter()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    retrieval_started = time.perf_counter()

    retrieved_documents = vector_store.similarity_search(
        query=question,
        k=6
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_started
    ) * 1000

    context_sections = []

    for document in retrieved_documents:
        source = document.metadata["source"]
        page = document.metadata["page"]

        section = (
            f"[Source: {source}, Page: {page}]\n"
            f"{document.page_content}"
        )

        context_sections.append(section)

    context = "\n\n".join(context_sections)

    source_pages = sorted({
        document.metadata.get("page")
        for document in retrieved_documents
        if document.metadata.get("page") is not None
    })

    prompt = f"""
You are the AVYUHA knowledge assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is unavailable, say:
   "I could not find this information in the AVYUHA knowledge base."
4. Give a clear and concise answer.
5. Cite the supporting PDF page.
6. Do not give assumed answers in any way.

Context:
{context}

User question:
{question}

Answer:
"""

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the AVYUHA knowledge base..."
        ):
            generation_started = time.perf_counter()

            response = llm.invoke(prompt)
            answer = response.text

            generation_ms = (
                time.perf_counter() - generation_started
            ) * 1000

        st.markdown(answer)

        with st.expander("View retrieved evidence"):
            for position, document in enumerate(
                retrieved_documents[:3],
                start=1
            ):
                st.markdown(
                    f"**Result {position} - "
                    f"Page {document.metadata['page']}**"
                )
                st.write(document.page_content)
                st.divider()

        total_ms = (
        time.perf_counter() - request_started
    ) * 1000

    log_id = log_rag_query(
        question=question,
        answer=answer,
        source_pages=source_pages,
        retrieved_chunks=len(retrieved_documents),
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
        status="success"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "log_id": log_id
    })

    display_feedback_buttons(log_id)