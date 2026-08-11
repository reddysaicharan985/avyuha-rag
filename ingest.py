import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

base_directory = Path(__file__).parent
pdf_path = base_directory / "data" / "avyuha_knowledge_base.pdf"
database_path = base_directory / "chroma_db"

load_dotenv(base_directory / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY was not found inside the .env file.")

reader = PdfReader(pdf_path)
pages = []

for page_number, pdf_page in enumerate(reader.pages, start=1):
    text = pdf_page.extract_text() or ""

    document = Document(
        page_content=text,
        metadata={
            "source": pdf_path.name,
            "page": page_number
        }
    )

    pages.append(document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120
)

chunks = text_splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

chunk_ids = [
    f"avyuha-chunk-{index}"
    for index in range(len(chunks))
]

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=chunk_ids,
    collection_name="avyuha_knowledge",
    persist_directory=str(database_path)
)

print(f"Loaded {len(pages)} PDF pages.")
print(f"Created {len(chunks)} chunks.")
print(f"Stored {len(chunks)} embeddings in ChromaDB.")
print(f"Database location: {database_path}")