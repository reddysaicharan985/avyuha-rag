from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

pdf_path = Path(__file__).parent / "data" / "avyuha_knowledge_base.pdf"

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

print(f"Successfully loaded {len(pages)} pages.")
print(f"Created {len(chunks)} chunks.")

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\nFirst chunk metadata:")
print(chunks[0].metadata)