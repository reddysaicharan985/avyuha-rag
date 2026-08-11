import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

base_directory = Path(__file__).parent
database_path = base_directory / "chroma_db"

load_dotenv(base_directory / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY was not found inside the .env file."
    )

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

question = input("Ask a question about AVYUHA: ")

retrieved_documents = vector_store.similarity_search(
    query=question,
    k=6
)

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

prompt = f"""
You are the AVYUHA knowledge assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is missing, say:
   "I could not find this information in the AVYUHA knowledge base."
4. Give a clear and concise answer.
5. Cite the supporting PDF page in the answer.
6. Do not infer or assume facts that are not explicitly supported
   by the supplied context.

Context:
{context}

User question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nAVYUHA RAG Answer:\n")
print(response.text)

retrieved_pages = sorted({
    document.metadata["page"]
    for document in retrieved_documents
})

print("\nRetrieved PDF pages:")
print(retrieved_pages)