import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

base_directory = Path(__file__).parent
database_path = base_directory / "chroma_db"

load_dotenv(base_directory / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY was not found inside the .env file.")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    collection_name="avyuha_knowledge",
    persist_directory=str(database_path),
    embedding_function=embeddings
)

question = input("Ask a question about AVYUHA: ")

results = vector_store.similarity_search_with_score(
    query=question,
    k=6
)

print("\nTop matching chunks:\n")

for position, result in enumerate(results, start=1):
    document, score = result

    print(f"Result {position}")
    print(f"Source: {document.metadata['source']}")
    print(f"Page: {document.metadata['page']}")
    print(f"Distance score: {score:.4f}")
    print("Content:")
    print(document.page_content)
    print("-" * 70)